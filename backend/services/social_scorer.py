"""Social compatibility utilities."""

import numpy as np
from sqlalchemy.orm import Session

from backend.models.friendship import Friendship
from backend.models.interaction import Interaction
from backend.models.user import User


def _parse(vec: list | None) -> np.ndarray:
    if not vec:
        return np.zeros(1)
    return np.array(vec, dtype=float)


def get_friend_ids(db: Session, user_id: str) -> set[str]:
    rows = db.query(Friendship).filter(
        ((Friendship.user_id == user_id) | (Friendship.friend_id == user_id)),
        Friendship.status == "accepted",
    ).all()
    return {r.friend_id if r.user_id == user_id else r.user_id for r in rows}


def get_mutual_ids(db: Session, user_id: str) -> set[str]:
    direct = get_friend_ids(db, user_id)
    if not direct:
        return set()
    rows = db.query(Friendship).filter(
        ((Friendship.user_id.in_(list(direct))) | (Friendship.friend_id.in_(list(direct)))),
        Friendship.status == "accepted",
    ).all()
    mutuals = set()
    for row in rows:
        mutuals.add(row.user_id)
        mutuals.add(row.friend_id)
    mutuals -= direct
    mutuals.discard(user_id)
    return mutuals


def categorize_relationship(db: Session, user_id: str, other_id: str) -> str:
    friends = get_friend_ids(db, user_id)
    if other_id in friends:
        return "friend"
    mutuals = get_mutual_ids(db, user_id)
    if other_id in mutuals:
        return "mutual"
    return "new_person"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compatibility_score(db: Session, user_a: User, user_b: User) -> float:
    vec_a = _parse(user_a.embedding_vector)
    vec_b = _parse(user_b.embedding_vector)
    embed = (_cosine(vec_a, vec_b) + 1.0) / 2.0

    cuisine_a = set(user_a.cuisine_preferences or [])
    cuisine_b = set(user_b.cuisine_preferences or [])
    vibe_a = set(user_a.vibe_preferences or [])
    vibe_b = set(user_b.vibe_preferences or [])

    cuisine_jaccard = len(cuisine_a & cuisine_b) / max(1, len(cuisine_a | cuisine_b))
    vibe_jaccard = len(vibe_a & vibe_b) / max(1, len(vibe_a | vibe_b))

    direct_friends = get_friend_ids(db, user_a.id)
    social = 1.0 if user_b.id in direct_friends else 0.0
    if social == 0.0:
        mutual = get_mutual_ids(db, user_a.id)
        social = 0.55 if user_b.id in mutual else 0.15

    shared_checkins = db.query(Interaction).filter(
        Interaction.user_id.in_([user_a.id, user_b.id]),
        Interaction.interaction_type == "checkin",
    ).all()
    by_venue: dict[str, set[str]] = {}
    for row in shared_checkins:
        by_venue.setdefault(row.venue_id, set()).add(row.user_id)
    co_attendance = sum(1 for users in by_venue.values() if len(users) == 2)
    co_score = min(1.0, co_attendance / 4.0)

    return float(0.45 * embed + 0.20 * cuisine_jaccard + 0.15 * vibe_jaccard + 0.10 * social + 0.10 * co_score)
