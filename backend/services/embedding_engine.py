"""Embedding training, online updates, and explainability.

This service combines:
1. Metadata priors (cold-start resilience)
2. Learned collaborative signal from interactions (matrix factorization/BPR)
3. Real-time online updates after every user event
4. Human-readable concept decomposition for trust and UX
"""

from __future__ import annotations

import math
import zlib
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from sqlalchemy.orm import Session

from backend.config import (
    EMBEDDING_DIM,
    EMBEDDING_EPOCHS,
    EMBEDDING_LR,
    EMBEDDING_REG,
    ONLINE_UPDATE_ALPHA,
)
from backend.models.interaction import Interaction
from backend.models.user import User
from backend.models.venue import Venue


INTERACTION_WEIGHTS = {
    "view": 0.12,
    "save": 0.85,
    "share": 0.70,
    "checkin": 1.15,
    "intent": 0.95,
    "invite_accept": 1.00,
    "invite_decline": -0.45,
    "unsave": -0.60,
}

ONLINE_WEIGHTS = {
    "view": 0.05,
    "save": 0.25,
    "share": 0.20,
    "checkin": 0.30,
    "intent": 0.24,
    "invite_accept": 0.18,
    "invite_decline": -0.16,
    "unsave": -0.22,
}


CONCEPT_CATALOG = [
    {"id": "culinary_explorer", "label": "Culinary Explorer", "categories": ["restaurant"], "vibes": ["authentic", "adventurous", "trendy"]},
    {"id": "late_night_social", "label": "Late Night Social", "categories": ["bar"], "vibes": ["lively", "loud", "group-friendly"]},
    {"id": "romantic_evenings", "label": "Romantic Evenings", "vibes": ["romantic", "intimate", "upscale"]},
    {"id": "chill_cafe_flow", "label": "Chill Cafe Flow", "categories": ["cafe"], "vibes": ["chill", "aesthetic", "casual"]},
    {"id": "budget_adventurer", "label": "Budget Adventurer", "vibes": ["authentic", "casual"], "price_hint": 1},
    {"id": "luxury_taste", "label": "Luxury Taste", "vibes": ["upscale", "intimate", "aesthetic"], "price_hint": 4},
    {"id": "group_energy", "label": "Group Energy", "categories": ["activity", "bar"], "vibes": ["group-friendly", "fun", "lively"]},
    {"id": "wellness_rhythm", "label": "Wellness Rhythm", "vibes": ["healthy", "chill", "aesthetic"], "cuisines": ["vegan", "juice_bar", "mediterranean"]},
    {"id": "asian_palette", "label": "Asian Palette", "cuisines": ["japanese", "korean", "thai", "vietnamese", "chinese"]},
    {"id": "european_palette", "label": "European Palette", "cuisines": ["italian", "french", "mediterranean"]},
    {"id": "sports_and_games", "label": "Sports & Games", "categories": ["activity", "bar"], "vibes": ["loud", "group-friendly", "fun"]},
    {"id": "hidden_gem_hunter", "label": "Hidden Gem Hunter", "vibes": ["authentic", "chill", "adventurous"]},
]


def _norm(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class EmbeddingEngine:
    def __init__(self):
        self.dim = EMBEDDING_DIM
        self._concept_vectors: dict[str, np.ndarray] = {}
        self._feature_cache: dict[str, np.ndarray] = {}

    def _feature_vector(self, key: str) -> np.ndarray:
        if key in self._feature_cache:
            return self._feature_cache[key]
        seed = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        vec = rng.normal(0.0, 1.0, self.dim)
        vec = _norm(vec)
        self._feature_cache[key] = vec
        return vec

    def venue_prior(self, venue: Venue) -> np.ndarray:
        parts: list[tuple[np.ndarray, float]] = []
        parts.append((self._feature_vector(f"category:{venue.category}"), 1.2))
        parts.append((self._feature_vector(f"price:{venue.price_level}"), 0.55))
        for vibe in (venue.vibe_tags or []):
            parts.append((self._feature_vector(f"vibe:{vibe}"), 0.85))
        for cuisine in (venue.cuisine_tags or []):
            parts.append((self._feature_vector(f"cuisine:{cuisine}"), 0.9))
        if not parts:
            return np.zeros(self.dim)
        vec = sum(p * w for p, w in parts) / max(sum(w for _, w in parts), 1e-8)
        return _norm(vec)

    def user_prior(self, user: User) -> np.ndarray:
        parts: list[tuple[np.ndarray, float]] = []
        parts.append((self._feature_vector(f"user_price:{user.price_preference or 2}"), 0.45))
        for vibe in (user.vibe_preferences or []):
            parts.append((self._feature_vector(f"vibe:{vibe}"), 1.0))
        for cuisine in (user.cuisine_preferences or []):
            parts.append((self._feature_vector(f"cuisine:{cuisine}"), 1.0))
        for slot in (user.preferred_time_slots or []):
            parts.append((self._feature_vector(f"time:{slot}"), 0.65))
        if user.archetype:
            parts.append((self._feature_vector(f"archetype:{user.archetype}"), 0.75))
        if not parts:
            return np.zeros(self.dim)
        vec = sum(p * w for p, w in parts) / max(sum(w for _, w in parts), 1e-8)
        return _norm(vec)

    def _parse_vector(self, value: list | None) -> np.ndarray:
        if not value:
            return np.zeros(self.dim)
        arr = np.array(value, dtype=float)
        if arr.size != self.dim:
            return np.zeros(self.dim)
        return arr

    def _to_list(self, value: np.ndarray) -> list[float]:
        return [round(float(x), 8) for x in value.tolist()]

    def _iter_training_events(self, interactions: Iterable[Interaction]):
        now = datetime.now(timezone.utc)
        for ix in interactions:
            base_weight = INTERACTION_WEIGHTS.get(ix.interaction_type, 0.08)
            days_ago = 0.0
            if ix.created_at:
                created = ix.created_at.replace(tzinfo=timezone.utc)
                days_ago = max(0.0, (now - created).total_seconds() / 86400.0)
            decay = math.exp(-days_ago / 50.0)
            duration_boost = 1.0
            if ix.view_duration_seconds:
                duration_boost += min(0.65, ix.view_duration_seconds / 120.0)
            w = base_weight * decay * duration_boost * max(ix.weight or 1.0, 0.1)
            yield ix.user_id, ix.venue_id, w

    def train_from_synthetic_data(self, db: Session, epochs: int | None = None):
        users = db.query(User).all()
        venues = db.query(Venue).all()
        interactions = db.query(Interaction).order_by(Interaction.created_at.asc()).all()
        if not users or not venues:
            return

        epoch_count = epochs or EMBEDDING_EPOCHS
        rng = np.random.RandomState(42)
        user_ids = [u.id for u in users]
        venue_ids = [v.id for v in venues]
        user_index = {uid: i for i, uid in enumerate(user_ids)}
        venue_index = {vid: i for i, vid in enumerate(venue_ids)}

        user_priors = np.array([self.user_prior(u) for u in users], dtype=float)
        venue_priors = np.array([self.venue_prior(v) for v in venues], dtype=float)

        user_factors = user_priors + rng.normal(0.0, 0.08, (len(users), self.dim))
        venue_factors = venue_priors + rng.normal(0.0, 0.08, (len(venues), self.dim))
        user_factors = np.array([_norm(v) for v in user_factors])
        venue_factors = np.array([_norm(v) for v in venue_factors])

        positives: dict[tuple[int, int], float] = {}
        negatives: dict[tuple[int, int], float] = {}
        user_positive_items: dict[int, set[int]] = {i: set() for i in range(len(users))}

        for uid, vid, w in self._iter_training_events(interactions):
            if uid not in user_index or vid not in venue_index:
                continue
            ui, vi = user_index[uid], venue_index[vid]
            if w >= 0:
                positives[(ui, vi)] = positives.get((ui, vi), 0.0) + w
                user_positive_items[ui].add(vi)
            else:
                negatives[(ui, vi)] = negatives.get((ui, vi), 0.0) + abs(w)

        if not positives:
            for idx, venue in enumerate(venues):
                venue.embedding_vector = self._to_list(_norm(venue_factors[idx]))
            for idx, user in enumerate(users):
                vec = _norm(user_factors[idx])
                user.embedding_vector = self._to_list(vec)
                user.embedding_updated_at = datetime.now(timezone.utc)
            self.compute_concept_vectors(venues, venue_factors)
            self.refresh_all_profiles(db, users, venues)
            db.commit()
            return

        positive_samples = list(positives.items())

        for _ in range(epoch_count):
            rng.shuffle(positive_samples)
            for (ui, vi), weight in positive_samples:
                for _step in range(max(1, min(4, int(round(weight * 1.8))))):
                    candidate = rng.randint(0, len(venues))
                    attempts = 0
                    while candidate in user_positive_items[ui] and attempts < 16:
                        candidate = rng.randint(0, len(venues))
                        attempts += 1
                    vj = candidate

                    u_vec = user_factors[ui]
                    i_vec = venue_factors[vi]
                    j_vec = venue_factors[vj]
                    diff = float(np.dot(u_vec, i_vec - j_vec))
                    grad = _sigmoid(-diff) * min(2.5, 0.5 + weight)
                    lr = EMBEDDING_LR

                    new_u = u_vec + lr * (grad * (i_vec - j_vec) - EMBEDDING_REG * u_vec)
                    new_i = i_vec + lr * (grad * u_vec - EMBEDDING_REG * i_vec)
                    new_j = j_vec + lr * (-grad * u_vec - EMBEDDING_REG * j_vec)

                    user_factors[ui] = _norm(new_u)
                    venue_factors[vi] = _norm(new_i)
                    venue_factors[vj] = _norm(new_j)

            for (ui, vi), weight in negatives.items():
                repel = EMBEDDING_LR * min(0.4, 0.06 * weight)
                user_factors[ui] = _norm(user_factors[ui] - repel * venue_factors[vi])

            user_factors = np.array([_norm(0.88 * u + 0.12 * p) for u, p in zip(user_factors, user_priors)])
            venue_factors = np.array([_norm(0.90 * v + 0.10 * p) for v, p in zip(venue_factors, venue_priors)])

        for idx, venue in enumerate(venues):
            venue.embedding_vector = self._to_list(_norm(venue_factors[idx]))
            venue.embedding_meta = {"version": "v2-trained", "dim": self.dim}
        for idx, user in enumerate(users):
            vec = _norm(user_factors[idx])
            user.embedding_vector = self._to_list(vec)
            user.embedding_meta = {"version": "v2-trained", "dim": self.dim}
            user.embedding_updated_at = datetime.now(timezone.utc)

        self.compute_concept_vectors(venues, venue_factors)
        self.refresh_all_profiles(db, users, venues)
        db.commit()

    def compute_concept_vectors(self, venues: list[Venue], venue_vectors: np.ndarray | None = None):
        vectors = venue_vectors
        if vectors is None:
            vectors = np.array([self._parse_vector(v.embedding_vector) for v in venues], dtype=float)
        self._concept_vectors = {}

        for concept in CONCEPT_CATALOG:
            selected = []
            for idx, venue in enumerate(venues):
                hit = False
                if concept.get("categories") and venue.category in concept["categories"]:
                    hit = True
                if concept.get("vibes") and set(concept["vibes"]) & set(venue.vibe_tags or []):
                    hit = True
                if concept.get("cuisines") and set(concept["cuisines"]) & set(venue.cuisine_tags or []):
                    hit = True
                if concept.get("price_hint") and venue.price_level == concept["price_hint"]:
                    hit = True
                if hit:
                    selected.append(vectors[idx])

            if selected:
                centroid = _norm(np.mean(selected, axis=0))
            else:
                parts = []
                for tag in concept.get("categories", []):
                    parts.append(self._feature_vector(f"category:{tag}"))
                for tag in concept.get("vibes", []):
                    parts.append(self._feature_vector(f"vibe:{tag}"))
                for tag in concept.get("cuisines", []):
                    parts.append(self._feature_vector(f"cuisine:{tag}"))
                if concept.get("price_hint"):
                    parts.append(self._feature_vector(f"price:{concept['price_hint']}"))
                centroid = _norm(np.mean(parts, axis=0)) if parts else np.zeros(self.dim)

            self._concept_vectors[concept["id"]] = centroid

    def concept_profile(self, vec: np.ndarray) -> list[dict]:
        if np.linalg.norm(vec) == 0:
            return []
        raw = []
        for concept in CONCEPT_CATALOG:
            cvec = self._concept_vectors.get(concept["id"])
            if cvec is None:
                continue
            raw.append((concept, _cosine(vec, cvec)))
        if not raw:
            return []

        values = np.array([v for _, v in raw], dtype=float)
        mean = float(values.mean())
        std = float(values.std()) or 1.0
        scored = []
        for concept, value in raw:
            centered = max(0.0, (value - mean) / std)
            sharpened = centered ** 1.3
            if sharpened <= 0:
                continue
            scored.append(
                {
                    "concept_id": concept["id"],
                    "label": concept["label"],
                    "activation": round(float(sharpened), 5),
                }
            )
        scored.sort(key=lambda item: -item["activation"])
        top = scored[:8]
        total = sum(s["activation"] for s in top) or 1.0
        for item in top:
            item["normalized"] = round(item["activation"] / total, 5)
        return top

    def explain_match(self, user_vec: np.ndarray, venue_vec: np.ndarray) -> dict:
        similarity = _cosine(user_vec, venue_vec)
        user_profile = self.concept_profile(user_vec)
        venue_profile = self.concept_profile(venue_vec)

        user_map = {c["concept_id"]: c for c in user_profile}
        venue_map = {c["concept_id"]: c for c in venue_profile}
        reasons = []
        for concept in CONCEPT_CATALOG:
            cid = concept["id"]
            if cid not in user_map or cid not in venue_map:
                continue
            contribution = user_map[cid]["activation"] * venue_map[cid]["activation"]
            if contribution <= 0.015:
                continue
            reasons.append(
                {
                    "concept_id": cid,
                    "label": concept["label"],
                    "strength": round(user_map[cid]["normalized"], 4),
                    "venue_fit": round(venue_map[cid]["normalized"], 4),
                    "contribution": round(contribution, 5),
                }
            )
        reasons.sort(key=lambda item: -item["contribution"])
        top = reasons[:4]
        if top:
            fragments = [f"{item['label']} ({int(item['strength'] * 100)}%)" for item in top[:2]]
            narrative = "Match driven by " + " and ".join(fragments) + "."
        else:
            narrative = "Match comes from overall latent similarity."
        return {
            "similarity": round(similarity, 5),
            "reasons": top,
            "narrative": narrative,
            "user_concepts": user_profile[:6],
            "venue_concepts": venue_profile[:6],
        }

    def ensure_initialized(self, db: Session):
        user = db.query(User).first()
        venue = db.query(Venue).first()
        if not user or not venue:
            return
        user_vec = self._parse_vector(user.embedding_vector)
        venue_vec = self._parse_vector(venue.embedding_vector)
        if np.linalg.norm(user_vec) == 0 or np.linalg.norm(venue_vec) == 0:
            self.train_from_synthetic_data(db)
            return
        venues = db.query(Venue).all()
        self.compute_concept_vectors(venues)

    def refresh_all_profiles(self, db: Session, users: list[User] | None = None, venues: list[Venue] | None = None):
        users = users or db.query(User).all()
        venues = venues or db.query(Venue).all()
        for user in users:
            u_vec = _norm(self._parse_vector(user.embedding_vector))
            profile = self.concept_profile(u_vec)
            top_labels = [c["label"] for c in profile[:3]]
            if top_labels:
                summary = "You lean toward " + ", ".join(top_labels[:-1] + [top_labels[-1]]) + "."
            else:
                summary = "Interact more to unlock your Taste DNA."
            meta = user.embedding_meta or {}
            meta.update({"top_concepts": profile[:6], "narrative": summary, "updated_at": datetime.now(timezone.utc).isoformat()})
            user.embedding_meta = meta

        for venue in venues:
            v_vec = _norm(self._parse_vector(venue.embedding_vector))
            profile = self.concept_profile(v_vec)
            meta = venue.embedding_meta or {}
            meta.update({"top_concepts": profile[:5]})
            venue.embedding_meta = meta

    def online_update(self, db: Session, user_id: str, venue_id: str, interaction_type: str, view_duration: float | None = None):
        user = db.query(User).filter(User.id == user_id).first()
        venue = db.query(Venue).filter(Venue.id == venue_id).first()
        if not user or not venue:
            return {"updated": False, "reason": "user_or_venue_missing"}

        u_old = _norm(self._parse_vector(user.embedding_vector))
        v_vec = _norm(self._parse_vector(venue.embedding_vector))
        if np.linalg.norm(v_vec) == 0:
            v_vec = self.venue_prior(venue)

        base = ONLINE_WEIGHTS.get(interaction_type, 0.06)
        if interaction_type == "view" and view_duration:
            base *= 1.0 + min(0.8, view_duration / 120.0)
        alpha = ONLINE_UPDATE_ALPHA * abs(base)

        if base >= 0:
            u_new = _norm((1 - alpha) * u_old + alpha * v_vec)
        else:
            u_new = _norm((1 + alpha) * u_old - alpha * v_vec)

        user.embedding_vector = self._to_list(u_new)
        user.embedding_updated_at = datetime.now(timezone.utc)

        profile = self.concept_profile(u_new)
        prev = self.concept_profile(u_old)
        prev_map = {p["concept_id"]: p["activation"] for p in prev}
        changes = []
        for item in profile[:8]:
            delta = item["activation"] - prev_map.get(item["concept_id"], 0.0)
            # Lower threshold to make profile drift easier to inspect in demos.
            if abs(delta) > 0.003:
                changes.append(
                    {
                        "concept_id": item["concept_id"],
                        "label": item["label"],
                        "delta": round(delta, 5),
                        "direction": "up" if delta > 0 else "down",
                    }
                )
        changes.sort(key=lambda c: -abs(c["delta"]))
        user.embedding_meta = {
            "top_concepts": profile[:6],
            "narrative": self._narrative_for_profile(profile),
            "latest_drift": round(float(np.linalg.norm(u_new - u_old)), 6),
            "concept_changes": changes[:6],
        }
        db.commit()
        return {
            "updated": True,
            "drift": user.embedding_meta["latest_drift"],
            "concept_changes": user.embedding_meta["concept_changes"],
            "top_concepts": profile[:6],
        }

    def _narrative_for_profile(self, profile: list[dict]) -> str:
        if not profile:
            return "Keep exploring to build a richer preference model."
        if len(profile) == 1:
            return f"Your strongest taste signal is {profile[0]['label']}."
        first = profile[0]["label"]
        second = profile[1]["label"]
        return f"Your current vibe blends {first} with {second}."


engine = EmbeddingEngine()
