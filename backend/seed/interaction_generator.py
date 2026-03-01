"""Generate realistic user-venue interactions over a 90-day period.

Key design choices for rigor:
  - Temporal patterns: weekday vs weekend browsing distributions
  - Bimodal view duration: quick scrolls (2-5s) vs engaged reads (15-60s)
  - Preference-biased venue selection: users interact more with venues matching their tastes
  - Social correlation: if user A checks in, 30% chance a friend also checked in same day
  - Interaction type distribution mirrors real app analytics funnels
"""

import random
import uuid
from datetime import datetime, timedelta, timezone, date

from backend.models.user import User
from backend.models.venue import Venue
from backend.models.interaction import Interaction
from backend.models.interest import Interest
from backend.models.friendship import Friendship
from backend.services.spatial_analyzer import haversine, preference_match_score

HOURLY_WEIGHTS_WEEKDAY = [
    0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07,
    0.08, 0.09, 0.10, 0.08, 0.06, 0.05, 0.04, 0.05, 0.06, 0.05,
    0.04, 0.03, 0.02, 0.01,
]
HOURLY_WEIGHTS_WEEKEND = [
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07,
    0.09, 0.10, 0.10, 0.09, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04,
    0.03, 0.03, 0.02, 0.01,
]

INTERACTION_PROBS = {
    "view": 0.55,
    "save": 0.15,
    "share": 0.05,
    "checkin": 0.08,
    "filter_click": 0.12,
    "unsave": 0.05,
}

ACTIVITY_LEVEL = {
    "foodie": 5.0,
    "nightlife": 5.0,
    "casual": 3.0,
    "date_night": 2.5,
    "budget_explorer": 4.5,
    "wellness": 3.0,
}

SOURCES = ["feed", "search", "friend_activity", "trending", "notification"]


def _pick_interaction_type() -> str:
    r = random.random()
    cumulative = 0.0
    for itype, prob in INTERACTION_PROBS.items():
        cumulative += prob
        if r < cumulative:
            return itype
    return "view"


def _view_duration(user: User, venue: Venue) -> float:
    """Bimodal: quick scroll or engaged view, biased by preference match."""
    match = preference_match_score(user, venue)
    engaged_prob = 0.3 + 0.4 * match
    if random.random() < engaged_prob:
        return max(5.0, random.gauss(30, 15))
    else:
        return max(1.0, random.gauss(3, 1.5))


def _pick_venue(user: User, venues: list[Venue], friend_checkin_venues: set[str]) -> Venue:
    """Weighted venue selection biased by preference match, distance, and friend activity."""
    weights = []
    for v in venues:
        dist = haversine(user.home_latitude, user.home_longitude, v.latitude, v.longitude)
        if dist > (user.max_travel_distance_km or 15):
            weights.append(0.01)
            continue
        match = preference_match_score(user, v)
        dist_factor = max(0.1, 1.0 - dist / 20.0)
        friend_boost = 2.0 if v.id in friend_checkin_venues else 1.0
        weights.append((0.3 + match) * dist_factor * friend_boost)

    total = sum(weights)
    if total == 0:
        return random.choice(venues)
    weights = [w / total for w in weights]
    return random.choices(venues, weights=weights, k=1)[0]


def generate_interactions(
    users: list[User],
    venues: list[Venue],
    friendships: list[Friendship],
    n_days: int = 90,
    rng_seed: int = 42,
) -> tuple[list[Interaction], list[Interest]]:
    random.seed(rng_seed)

    friend_map: dict[str, set[str]] = {u.id: set() for u in users}
    for f in friendships:
        if f.status == "accepted":
            friend_map.setdefault(f.user_id, set()).add(f.friend_id)
            friend_map.setdefault(f.friend_id, set()).add(f.user_id)

    user_map = {u.id: u for u in users}
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=n_days)

    interactions: list[Interaction] = []
    interests: list[Interest] = []
    daily_checkins: dict[str, dict[str, set[str]]] = {}

    for day_offset in range(n_days):
        current_date = start_date + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5
        hourly_weights = HOURLY_WEIGHTS_WEEKEND if is_weekend else HOURLY_WEIGHTS_WEEKDAY
        date_str = current_date.strftime("%Y-%m-%d")
        daily_checkins[date_str] = {}

        friend_venue_today: set[str] = set()

        for user in users:
            arch = user.archetype or "casual"
            base_activity = ACTIVITY_LEVEL.get(arch, 3.0)
            weekend_boost = 1.4 if is_weekend else 1.0
            n_actions = max(0, int(random.gauss(base_activity * weekend_boost, 1.5)))

            for _ in range(n_actions):
                hour = random.choices(range(24), weights=hourly_weights, k=1)[0]
                minute = random.randint(0, 59)
                ts = current_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))

                venue = _pick_venue(user, venues, friend_venue_today)
                itype = _pick_interaction_type()

                view_dur = None
                scroll_dep = None
                if itype == "view":
                    view_dur = round(_view_duration(user, venue), 1)
                    scroll_dep = round(min(1.0, random.betavariate(2, 3)), 2)

                interactions.append(Interaction(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    venue_id=venue.id,
                    interaction_type=itype,
                    view_duration_seconds=view_dur,
                    scroll_depth=scroll_dep,
                    source=random.choice(SOURCES),
                    created_at=ts,
                ))

                if itype == "checkin":
                    daily_checkins[date_str].setdefault(venue.id, set()).add(user.id)
                    friend_venue_today.add(venue.id)

                    for fid in friend_map.get(user.id, set()):
                        if random.random() < 0.30 and fid in user_map:
                            interactions.append(Interaction(
                                id=str(uuid.uuid4()),
                                user_id=fid,
                                venue_id=venue.id,
                                interaction_type="checkin",
                                source="friend_activity",
                                created_at=ts + timedelta(minutes=random.randint(5, 120)),
                            ))
                            daily_checkins[date_str].setdefault(venue.id, set()).add(fid)

                if itype == "save" and random.random() < 0.4:
                    pref_date = (current_date + timedelta(days=random.randint(1, 14))).date() if hasattr(current_date, 'date') else current_date.date() + timedelta(days=random.randint(1, 14))
                    interests.append(Interest(
                        id=str(uuid.uuid4()),
                        user_id=user.id,
                        venue_id=venue.id,
                        preferred_date=pref_date,
                        preferred_time_slot=random.choice(["morning", "afternoon", "evening", "night"]),
                        flexible_date=random.random() < 0.7,
                        min_group_size=random.choice([2, 2, 3]),
                        max_group_size=random.choice([4, 6, 8]),
                        status="active" if day_offset > n_days - 14 else random.choice(["active", "expired"]),
                        created_at=ts,
                        expires_at=ts + timedelta(days=7),
                    ))

    return interactions, interests
