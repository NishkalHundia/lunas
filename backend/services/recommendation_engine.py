"""Core recommendation ranking with trained embeddings + social context."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone, date
import re

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.friendship import Friendship
from backend.models.interaction import Interaction
from backend.models.interest import Interest
from backend.models.invitation import Invitation
from backend.models.invitation_group import InvitationGroup
from backend.models.user import User
from backend.models.venue import Venue
from backend.services.embedding_engine import CONCEPT_CATALOG, engine
from backend.services.social_scorer import categorize_relationship
from backend.services.spatial_analyzer import haversine, proximity_score, preference_match_score


WEIGHTS = {
    "embedding": 0.36,
    "social": 0.22,
    "preference": 0.14,
    "distance": 0.11,
    "trend": 0.10,
    "novelty": 0.07,
}


def _vec(payload: list | None) -> np.ndarray:
    if not payload:
        return np.zeros(engine.dim)
    arr = np.array(payload, dtype=float)
    if arr.size != engine.dim:
        return np.zeros(engine.dim)
    return arr


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _embedding_score(user_vec: np.ndarray, venue_vec: np.ndarray) -> float:
    return (_cosine(user_vec, venue_vec) + 1.0) / 2.0


def _novelty_score(view_count_14d: int) -> float:
    if view_count_14d <= 0:
        return 1.0
    return max(0.08, float(np.exp(-0.45 * view_count_14d)))


def _time_slot_match(user: User, slot: str | None) -> float:
    if not slot:
        return 0.5
    preferred = set(user.preferred_time_slots or [])
    if not preferred:
        return 0.6
    return 1.0 if slot in preferred else 0.35


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _fast_compatibility(requester: User, other: User, requester_vec: np.ndarray, other_vec: np.ndarray, relation: str) -> float:
    embed = (_cosine(requester_vec, other_vec) + 1.0) / 2.0
    cuisine = _jaccard(set(requester.cuisine_preferences or []), set(other.cuisine_preferences or []))
    vibe = _jaccard(set(requester.vibe_preferences or []), set(other.vibe_preferences or []))
    social = 1.0 if relation == "friend" else (0.7 if relation == "mutual" else 0.35)
    return float(0.56 * embed + 0.22 * cuisine + 0.12 * vibe + 0.10 * social)


def _build_social_sets(db: Session, user_id: str) -> tuple[set[str], set[str]]:
    rows = db.query(Friendship).filter(
        ((Friendship.user_id == user_id) | (Friendship.friend_id == user_id)),
        Friendship.status == "accepted",
    ).all()
    friend_ids = {row.friend_id if row.user_id == user_id else row.user_id for row in rows}

    if not friend_ids:
        return set(), set()
    fof_rows = db.query(Friendship).filter(
        ((Friendship.user_id.in_(list(friend_ids))) | (Friendship.friend_id.in_(list(friend_ids)))),
        Friendship.status == "accepted",
    ).all()
    mutual_ids = set()
    for row in fof_rows:
        mutual_ids.add(row.user_id)
        mutual_ids.add(row.friend_id)
    mutual_ids -= friend_ids
    mutual_ids.discard(user_id)
    return friend_ids, mutual_ids


def get_recommendations(
    db: Session,
    user_id: str,
    latitude: float | None = None,
    longitude: float | None = None,
    target_date: date | None = None,
    time_slot: str | None = None,
    limit: int = 20,
) -> list[dict]:
    engine.ensure_initialized(db)

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return []

    user_vec = _vec(user.embedding_vector)
    if np.linalg.norm(user_vec) == 0:
        user_vec = engine.user_prior(user)

    friend_ids, mutual_ids = _build_social_sets(db, user_id)
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_14d = now - timedelta(days=14)

    candidates = db.query(Venue).filter(Venue.city == user.city).all()
    nearby = []
    for venue in candidates:
        dist = haversine(user.home_latitude, user.home_longitude, venue.latitude, venue.longitude)
        if dist <= max(2.0, user.max_travel_distance_km or 14.0):
            nearby.append(venue)
    candidates = nearby
    if not candidates:
        return []

    venue_ids = [venue.id for venue in candidates]

    interests = db.query(Interest).filter(
        Interest.venue_id.in_(venue_ids),
        Interest.status == "active",
    ).all()
    interest_by_venue: dict[str, list[str]] = defaultdict(list)
    for interest in interests:
        interest_by_venue[interest.venue_id].append(interest.user_id)

    invite_groups = db.query(InvitationGroup).filter(
        InvitationGroup.venue_id.in_(venue_ids),
        InvitationGroup.status == "collecting",
    ).all()
    group_ids = [group.id for group in invite_groups]
    invites = []
    if group_ids:
        invites = db.query(Invitation).filter(Invitation.group_id.in_(group_ids)).all()
    accepted_by_venue: dict[str, int] = defaultdict(int)
    for invite in invites:
        if invite.status == "accepted":
            accepted_by_venue[invite.venue_id] += 1

    trend_rows = db.query(
        Interaction.venue_id,
        Interaction.interaction_type,
        func.count(Interaction.id),
    ).filter(
        Interaction.venue_id.in_(venue_ids),
        Interaction.created_at >= cutoff_24h.replace(tzinfo=None),
    ).group_by(Interaction.venue_id, Interaction.interaction_type).all()
    trend_by_venue: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for venue_id, interaction_type, count in trend_rows:
        trend_by_venue[venue_id][interaction_type] = count

    views_14d = db.query(
        Interaction.venue_id,
        func.count(Interaction.id),
    ).filter(
        Interaction.user_id == user_id,
        Interaction.venue_id.in_(venue_ids),
        Interaction.interaction_type == "view",
        Interaction.created_at >= cutoff_14d.replace(tzinfo=None),
    ).group_by(Interaction.venue_id).all()
    view_counts = {venue_id: count for venue_id, count in views_14d}

    people_ids = set()
    for venue in candidates:
        people_ids.update(interest_by_venue.get(venue.id, []))
    people_ids.discard(user_id)
    users_map = {}
    if people_ids:
        users = db.query(User).filter(User.id.in_(list(people_ids)[:800])).all()
        users_map = {user_obj.id: user_obj for user_obj in users}
    user_vec_map: dict[str, np.ndarray] = {
        uid: _vec(user_obj.embedding_vector) for uid, user_obj in users_map.items()
    }

    scored = []
    for venue in candidates:
        venue_vec = _vec(venue.embedding_vector)
        if np.linalg.norm(venue_vec) == 0:
            venue_vec = engine.venue_prior(venue)

        emb_score = _embedding_score(user_vec, venue_vec)
        pref = preference_match_score(user, venue)
        dist = proximity_score(user, venue)
        novelty = _novelty_score(view_counts.get(venue.id, 0))

        interested = interest_by_venue.get(venue.id, [])
        friends_here = sum(1 for uid in interested if uid in friend_ids)
        mutuals_here = sum(1 for uid in interested if uid in mutual_ids)
        others_here = sum(1 for uid in interested if uid not in friend_ids and uid not in mutual_ids and uid != user_id)
        accepted = accepted_by_venue.get(venue.id, 0)
        social = (
            0.45 * min(1.0, friends_here / 3.0)
            + 0.25 * min(1.0, mutuals_here / 5.0)
            + 0.20 * min(1.0, others_here / 12.0)
            + 0.10 * min(1.0, accepted / 4.0)
        )

        trends = trend_by_venue.get(venue.id, {})
        trend = (
            0.45 * min(1.0, trends.get("save", 0) / 10.0)
            + 0.20 * min(1.0, trends.get("checkin", 0) / 8.0)
            + 0.20 * min(1.0, trends.get("share", 0) / 5.0)
            + 0.15 * min(1.0, trends.get("view", 0) / 60.0)
        )
        trend = max(0.0, min(1.0, trend + 0.05 * (venue.popularity_prior or 0.0)))

        slot_match = _time_slot_match(user, time_slot)

        score = (
            WEIGHTS["embedding"] * emb_score
            + WEIGHTS["social"] * social
            + WEIGHTS["preference"] * pref
            + WEIGHTS["distance"] * dist
            + WEIGHTS["trend"] * trend
            + WEIGHTS["novelty"] * novelty
        ) * (0.85 + 0.15 * slot_match)

        explanation = engine.explain_match(user_vec, venue_vec)

        people = []
        # Keep this bounded for performance in large synthetic datasets.
        for uid in interested[:80]:
            if uid == user_id:
                continue
            other = users_map.get(uid)
            if not other:
                continue
            relation = "friend" if uid in friend_ids else ("mutual" if uid in mutual_ids else "new_person")
            other_vec = user_vec_map.get(uid)
            if other_vec is None or np.linalg.norm(other_vec) == 0:
                other_vec = engine.user_prior(other)
            people.append(
                {
                    "user": other,
                    "relationship": relation,
                    "compatibility": round(_fast_compatibility(user, other, user_vec, other_vec, relation), 4),
                    "shared_concepts": [r["label"] for r in explanation["reasons"][:2]],
                }
            )

        people.sort(
            key=lambda row: (
                {"friend": 0, "mutual": 1, "new_person": 2}[row["relationship"]],
                -row["compatibility"],
            )
        )

        scored.append(
            {
                "venue": venue,
                "score": round(score, 5),
                "score_breakdown": {
                    "embedding": round(emb_score, 5),
                    "social_proof": round(social, 5),
                    "preference": round(pref, 5),
                    "distance": round(dist, 5),
                    "trending": round(trend, 5),
                    "novelty": round(novelty, 5),
                    "time_slot_match": round(slot_match, 5),
                },
                "people": people[:10],
                "total_interested": len(interested),
                "friend_count": friends_here,
                "mutual_count": mutuals_here,
                "trending": trend > 0.45,
                "explanation": explanation,
            }
        )

    scored.sort(key=lambda row: -row["score"])
    return scored[:limit]


def get_people_recommendations(
    db: Session,
    user_id: str,
    venue_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    engine.ensure_initialized(db)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return []

    friend_ids, mutual_ids = _build_social_sets(db, user_id)
    user_vec = _vec(user.embedding_vector)
    user_profile = [c["label"] for c in engine.concept_profile(user_vec)[:4]]

    query = db.query(User).filter(User.id != user_id)
    if venue_id:
        interested_rows = db.query(Interest.user_id).filter(
            Interest.venue_id == venue_id,
            Interest.status == "active",
            Interest.user_id != user_id,
        ).all()
        interested_ids = [row[0] for row in interested_rows]
        if not interested_ids:
            return []
        query = query.filter(User.id.in_(interested_ids))
    else:
        query = query.limit(250)

    people = query.all()
    results = []
    for other in people:
        other_vec = _vec(other.embedding_vector)
        sim = (_cosine(user_vec, other_vec) + 1.0) / 2.0
        rel = "friend" if other.id in friend_ids else ("mutual" if other.id in mutual_ids else "new_person")
        compat = _fast_compatibility(user, other, user_vec, other_vec, rel)

        other_profile = [c["label"] for c in engine.concept_profile(other_vec)[:4]]
        shared = [label for label in user_profile if label in other_profile]
        narrative = (
            f"Both of you align on {', '.join(shared[:2])}."
            if shared
            else "Taste overlap comes from latent behavior similarity."
        )

        score = 0.58 * sim + 0.42 * compat
        results.append(
            {
                "user": other,
                "relationship": rel,
                "compatibility": round(score, 5),
                "shared_concepts": shared[:4],
                "narrative": narrative,
            }
        )

    results.sort(
        key=lambda row: (
            {"friend": 0, "mutual": 1, "new_person": 2}[row["relationship"]],
            -row["compatibility"],
        )
    )
    return results[:limit]


def get_pair_recommendations(
    db: Session,
    user_id: str,
    other_user_id: str,
    time_slot: str | None = None,
    limit: int = 12,
) -> list[dict]:
    engine.ensure_initialized(db)

    user = db.query(User).filter(User.id == user_id).first()
    other = db.query(User).filter(User.id == other_user_id).first()
    if not user or not other or user.id == other.id:
        return []

    user_vec = _vec(user.embedding_vector)
    other_vec = _vec(other.embedding_vector)
    if np.linalg.norm(user_vec) == 0:
        user_vec = engine.user_prior(user)
    if np.linalg.norm(other_vec) == 0:
        other_vec = engine.user_prior(other)

    pair_vec = _unit(0.52 * user_vec + 0.48 * other_vec)
    relationship = categorize_relationship(db, user.id, other.id)

    base_city = user.city or other.city
    all_candidates = db.query(Venue).filter(Venue.city == base_city).all()
    max_user_distance = max(2.0, user.max_travel_distance_km or 14.0)
    max_other_distance = max(2.0, other.max_travel_distance_km or 14.0)

    filtered: list[tuple[Venue, float, float]] = []
    for venue in all_candidates:
        user_km = haversine(user.home_latitude, user.home_longitude, venue.latitude, venue.longitude)
        other_km = haversine(other.home_latitude, other.home_longitude, venue.latitude, venue.longitude)
        if user_km <= max_user_distance and other_km <= max_other_distance:
            filtered.append((venue, user_km, other_km))
    if not filtered:
        for venue in all_candidates:
            user_km = haversine(user.home_latitude, user.home_longitude, venue.latitude, venue.longitude)
            other_km = haversine(other.home_latitude, other.home_longitude, venue.latitude, venue.longitude)
            if user_km <= max_user_distance * 1.3 or other_km <= max_other_distance * 1.3:
                filtered.append((venue, user_km, other_km))
    if not filtered:
        return []

    venue_ids = [venue.id for venue, _, _ in filtered]
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_14d = now - timedelta(days=14)

    interests = db.query(Interest).filter(
        Interest.venue_id.in_(venue_ids),
        Interest.status == "active",
    ).all()
    interest_by_venue: dict[str, list[str]] = defaultdict(list)
    for interest in interests:
        interest_by_venue[interest.venue_id].append(interest.user_id)

    trend_rows = db.query(
        Interaction.venue_id,
        Interaction.interaction_type,
        func.count(Interaction.id),
    ).filter(
        Interaction.venue_id.in_(venue_ids),
        Interaction.created_at >= cutoff_24h.replace(tzinfo=None),
    ).group_by(Interaction.venue_id, Interaction.interaction_type).all()
    trend_by_venue: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for venue_id, interaction_type, count in trend_rows:
        trend_by_venue[venue_id][interaction_type] = count

    view_rows = db.query(
        Interaction.venue_id,
        func.count(Interaction.id),
    ).filter(
        Interaction.user_id.in_([user.id, other.id]),
        Interaction.venue_id.in_(venue_ids),
        Interaction.interaction_type == "view",
        Interaction.created_at >= cutoff_14d.replace(tzinfo=None),
    ).group_by(Interaction.venue_id).all()
    view_counts = {venue_id: count for venue_id, count in view_rows}

    user_profile = engine.concept_profile(user_vec)
    other_profile = engine.concept_profile(other_vec)
    user_labels = [item["label"] for item in user_profile[:6]]
    other_labels = [item["label"] for item in other_profile[:6]]
    shared_concepts = [label for label in user_labels if label in other_labels][:4]

    scored = []
    for venue, user_km, other_km in filtered:
        venue_vec = _vec(venue.embedding_vector)
        if np.linalg.norm(venue_vec) == 0:
            venue_vec = engine.venue_prior(venue)

        user_match = _embedding_score(user_vec, venue_vec)
        other_match = _embedding_score(other_vec, venue_vec)
        pair_alignment = 0.5 * user_match + 0.5 * other_match

        pref_score = 0.5 * preference_match_score(user, venue) + 0.5 * preference_match_score(other, venue)
        prox_user = proximity_score(user, venue)
        prox_other = proximity_score(other, venue)
        travel_fairness = min(prox_user, prox_other)
        distance_balance = 1.0 - min(1.0, abs(user_km - other_km) / max(1.0, max(user_km, other_km)))

        interested = interest_by_venue.get(venue.id, [])
        user_interested = 1.0 if user.id in interested else 0.0
        other_interested = 1.0 if other.id in interested else 0.0
        social_overlap = (
            0.4 * user_interested
            + 0.4 * other_interested
            + 0.2 * min(1.0, max(0, len(interested) - int(user_interested) - int(other_interested)) / 8.0)
        )

        trends = trend_by_venue.get(venue.id, {})
        trend = (
            0.45 * min(1.0, trends.get("save", 0) / 10.0)
            + 0.2 * min(1.0, trends.get("checkin", 0) / 8.0)
            + 0.2 * min(1.0, trends.get("share", 0) / 5.0)
            + 0.15 * min(1.0, trends.get("view", 0) / 60.0)
        )
        trend = max(0.0, min(1.0, trend + 0.05 * (venue.popularity_prior or 0.0)))
        novelty = _novelty_score(view_counts.get(venue.id, 0))

        slot_match = 0.5 * _time_slot_match(user, time_slot) + 0.5 * _time_slot_match(other, time_slot)
        score = (
            0.38 * pair_alignment
            + 0.2 * pref_score
            + 0.16 * travel_fairness
            + 0.08 * distance_balance
            + 0.08 * social_overlap
            + 0.1 * trend
            + 0.08 * novelty
        ) * (0.82 + 0.18 * slot_match)

        explanation = engine.explain_match(pair_vec, venue_vec)
        if shared_concepts:
            narrative = (
                f"Strong overlap on {', '.join(shared_concepts[:2])}; "
                "this venue balances both of your taste profiles."
            )
        else:
            narrative = "Recommended as a balanced compromise from both embeddings."

        scored.append(
            {
                "venue": venue,
                "score": round(score, 5),
                "relationship": relationship,
                "user_match": round(user_match, 5),
                "other_match": round(other_match, 5),
                "distance_km": {
                    "user": round(float(user_km), 2),
                    "other": round(float(other_km), 2),
                },
                "score_breakdown": {
                    "pair_alignment": round(pair_alignment, 5),
                    "preference": round(pref_score, 5),
                    "travel_fairness": round(travel_fairness, 5),
                    "distance_balance": round(distance_balance, 5),
                    "social_overlap": round(social_overlap, 5),
                    "trending": round(trend, 5),
                    "novelty": round(novelty, 5),
                    "time_slot_match": round(slot_match, 5),
                },
                "shared_concepts": shared_concepts,
                "narrative": narrative,
                "explanation": explanation,
            }
        )

    scored.sort(key=lambda row: -row["score"])
    return scored[:limit]


def _tokenize_query(query: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9_]+", (query or "").lower()) if tok]


def _build_query_embedding(candidates: list[Venue], query: str) -> np.ndarray:
    tokens = _tokenize_query(query)
    if not tokens:
        return np.zeros(engine.dim)

    categories = {((venue.category or "").lower()) for venue in candidates}
    subcategories = {((venue.subcategory or "").lower()) for venue in candidates if venue.subcategory}
    cuisines = {tag.lower() for venue in candidates for tag in (venue.cuisine_tags or [])}
    vibes = {tag.lower() for venue in candidates for tag in (venue.vibe_tags or [])}

    parts: list[tuple[np.ndarray, float]] = []
    for token in tokens:
        if token in categories:
            parts.append((engine._feature_vector(f"category:{token}"), 1.25))
        if token in subcategories:
            parts.append((engine._feature_vector(f"category:{token}"), 0.65))
        if token in cuisines:
            parts.append((engine._feature_vector(f"cuisine:{token}"), 1.0))
        if token in vibes:
            parts.append((engine._feature_vector(f"vibe:{token}"), 0.95))

    # Concept-label bridge for queries like "vegan healthy" or "sports games".
    for concept in CONCEPT_CATALOG:
        label_tokens = set(_tokenize_query(concept["label"]))
        if label_tokens and any(tok in label_tokens for tok in tokens):
            cvec = engine._concept_vectors.get(concept["id"])
            if cvec is not None and np.linalg.norm(cvec) > 0:
                parts.append((cvec, 0.9))

    if not parts:
        return np.zeros(engine.dim)
    vec = sum(v * w for v, w in parts) / max(sum(w for _, w in parts), 1e-8)
    return _unit(vec)


def search_recommendations(
    db: Session,
    user_id: str,
    query: str,
    time_slot: str | None = None,
    limit: int = 24,
) -> dict:
    engine.ensure_initialized(db)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"query": query, "exact_match_found": False, "results": []}

    q = (query or "").strip().lower()
    if not q:
        rows = get_recommendations(db, user_id, time_slot=time_slot, limit=limit)
        return {"query": query, "exact_match_found": True, "results": rows}

    base_rows = get_recommendations(db, user_id, time_slot=time_slot, limit=max(80, limit * 5))
    by_venue_id = {row["venue"].id: row for row in base_rows}

    city_candidates = db.query(Venue).filter(Venue.city == user.city).all()
    if not city_candidates:
        city_candidates = [row["venue"] for row in base_rows]

    def text_blob(venue: Venue) -> str:
        chunks = [
            venue.name or "",
            venue.category or "",
            venue.subcategory or "",
            venue.neighborhood or "",
            venue.description or "",
            " ".join(venue.cuisine_tags or []),
            " ".join(venue.vibe_tags or []),
        ]
        return " ".join(chunks).lower()

    exact_candidates = [venue for venue in city_candidates if q in text_blob(venue)]
    exact_match_found = bool(exact_candidates)

    if exact_match_found:
        ranked = []
        for venue in exact_candidates:
            row = by_venue_id.get(venue.id)
            base_score = row["score"] if row else 0.0
            ranked.append((venue, base_score))
        ranked.sort(key=lambda item: -item[1])

        results = []
        for venue, _score in ranked[:limit]:
            row = by_venue_id.get(venue.id)
            if row:
                payload = dict(row)
                payload["match_type"] = "exact"
                payload["query_similarity"] = 1.0
                results.append(payload)
            else:
                venue_vec = _vec(venue.embedding_vector)
                if np.linalg.norm(venue_vec) == 0:
                    venue_vec = engine.venue_prior(venue)
                explanation = engine.explain_match(_vec(user.embedding_vector), venue_vec)
                results.append(
                    {
                        "venue": venue,
                        "score": 0.0,
                        "score_breakdown": {
                            "embedding": 0.0,
                            "social_proof": 0.0,
                            "preference": 0.0,
                            "distance": 0.0,
                            "trending": 0.0,
                            "novelty": 0.0,
                            "time_slot_match": 0.0,
                        },
                        "people": [],
                        "total_interested": 0,
                        "friend_count": 0,
                        "mutual_count": 0,
                        "trending": False,
                        "explanation": explanation,
                        "match_type": "exact",
                        "query_similarity": 1.0,
                    }
                )
        return {"query": query, "exact_match_found": True, "results": results}

    query_vec = _build_query_embedding(city_candidates, q)
    fallback_ranked = []
    for venue in city_candidates:
        venue_vec = _vec(venue.embedding_vector)
        if np.linalg.norm(venue_vec) == 0:
            venue_vec = engine.venue_prior(venue)
        q_sim = _embedding_score(query_vec, venue_vec) if np.linalg.norm(query_vec) > 0 else 0.0
        personalization = by_venue_id.get(venue.id, {}).get("score", 0.0)
        blended = 0.68 * q_sim + 0.32 * personalization
        fallback_ranked.append((venue, q_sim, blended))
    fallback_ranked.sort(key=lambda row: -row[2])

    results = []
    for venue, q_sim, blended in fallback_ranked[:limit]:
        row = by_venue_id.get(venue.id)
        if row:
            payload = dict(row)
            payload["score"] = round(float(max(payload["score"], blended)), 5)
            payload["match_type"] = "semantic"
            payload["query_similarity"] = round(float(q_sim), 5)
            results.append(payload)
        else:
            venue_vec = _vec(venue.embedding_vector)
            if np.linalg.norm(venue_vec) == 0:
                venue_vec = engine.venue_prior(venue)
            explanation = engine.explain_match(_vec(user.embedding_vector), venue_vec)
            results.append(
                {
                    "venue": venue,
                    "score": round(float(blended), 5),
                    "score_breakdown": {
                        "embedding": 0.0,
                        "social_proof": 0.0,
                        "preference": 0.0,
                        "distance": 0.0,
                        "trending": 0.0,
                        "novelty": 0.0,
                        "time_slot_match": 0.0,
                    },
                    "people": [],
                    "total_interested": 0,
                    "friend_count": 0,
                    "mutual_count": 0,
                    "trending": False,
                    "explanation": explanation,
                    "match_type": "semantic",
                    "query_similarity": round(float(q_sim), 5),
                }
            )

    return {"query": query, "exact_match_found": False, "results": results}
