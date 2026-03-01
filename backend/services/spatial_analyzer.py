"""Spatial scoring and metadata preference matching."""

import math

from backend.models.user import User
from backend.models.venue import Venue


EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def proximity_score(user: User, venue: Venue) -> float:
    dist = haversine(user.home_latitude, user.home_longitude, venue.latitude, venue.longitude)
    max_distance = max(1.0, user.max_travel_distance_km or 14.0)
    if dist > max_distance:
        return 0.0
    relative = dist / max_distance
    return max(0.0, math.exp(-2.2 * relative))


def preference_match_score(user: User, venue: Venue) -> float:
    user_cuisines = set(user.cuisine_preferences or [])
    user_vibes = set(user.vibe_preferences or [])
    venue_cuisines = set(venue.cuisine_tags or [])
    venue_vibes = set(venue.vibe_tags or [])

    cuisine_overlap = len(user_cuisines & venue_cuisines) / max(len(user_cuisines), 1)
    vibe_overlap = len(user_vibes & venue_vibes) / max(len(user_vibes), 1)
    price_delta = abs((user.price_preference or 2) - (venue.price_level or 2))
    price_score = max(0.0, 1.0 - 0.33 * price_delta)

    return 0.42 * cuisine_overlap + 0.36 * vibe_overlap + 0.22 * price_score
