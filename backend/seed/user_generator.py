"""Generate realistic synthetic users across Austin, TX neighborhoods.

Uses archetype-driven generation: each user is assigned an archetype that
drives correlated preferences (a foodie who likes Japanese also tends to
like ramen, a nightlife person prefers bars in lively areas, etc.).

Location is sampled from a Gaussian centered on the user's assigned
neighborhood, producing realistic spatial clustering.
"""

import random
import uuid
from datetime import datetime, timedelta, timezone
from faker import Faker
from backend.models.user import User

fake = Faker()
Faker.seed(42)

NEIGHBORHOODS = [
    {"name": "Downtown", "center": (30.2672, -97.7431), "radius_km": 2, "weight": 0.20},
    {"name": "East Austin", "center": (30.2621, -97.7195), "radius_km": 3, "weight": 0.18},
    {"name": "South Congress", "center": (30.2460, -97.7487), "radius_km": 2, "weight": 0.15},
    {"name": "North Loop", "center": (30.3172, -97.7231), "radius_km": 2, "weight": 0.10},
    {"name": "Zilker", "center": (30.2670, -97.7730), "radius_km": 2, "weight": 0.08},
    {"name": "Mueller", "center": (30.2988, -97.7056), "radius_km": 2, "weight": 0.08},
    {"name": "West Campus", "center": (30.2849, -97.7414), "radius_km": 1.5, "weight": 0.12},
    {"name": "Hyde Park", "center": (30.3060, -97.7270), "radius_km": 1.5, "weight": 0.09},
]

ARCHETYPES = {
    "foodie": {
        "cuisine_pool": ["japanese", "italian", "thai", "mexican", "korean", "vietnamese", "french", "mediterranean"],
        "vibe_pool": ["trendy", "intimate", "adventurous", "aesthetic", "authentic"],
        "price_range": [2, 3, 4],
        "age_range": (23, 38),
    },
    "nightlife": {
        "cuisine_pool": ["american", "mexican", "bar_food", "korean", "pizza"],
        "vibe_pool": ["lively", "loud", "group-friendly", "trendy", "fun"],
        "price_range": [2, 3],
        "age_range": (21, 32),
    },
    "casual": {
        "cuisine_pool": ["american", "italian", "mexican", "chinese", "pizza", "burgers"],
        "vibe_pool": ["chill", "casual", "family-friendly", "group-friendly"],
        "price_range": [1, 2],
        "age_range": (22, 45),
    },
    "date_night": {
        "cuisine_pool": ["italian", "french", "japanese", "mediterranean", "thai"],
        "vibe_pool": ["romantic", "intimate", "upscale", "aesthetic"],
        "price_range": [3, 4],
        "age_range": (25, 40),
    },
    "budget_explorer": {
        "cuisine_pool": ["mexican", "vietnamese", "indian", "ethiopian", "thai", "chinese"],
        "vibe_pool": ["authentic", "adventurous", "casual", "chill"],
        "price_range": [1, 2],
        "age_range": (20, 30),
    },
    "wellness": {
        "cuisine_pool": ["vegan", "mediterranean", "juice_bar", "cafe", "healthy"],
        "vibe_pool": ["chill", "healthy", "aesthetic", "intimate"],
        "price_range": [2, 3],
        "age_range": (22, 35),
    },
}

ARCHETYPE_WEIGHTS = {
    "foodie": 0.20, "nightlife": 0.18, "casual": 0.25,
    "date_night": 0.12, "budget_explorer": 0.15, "wellness": 0.10,
}

ALL_CUISINES = list({c for a in ARCHETYPES.values() for c in a["cuisine_pool"]})
ALL_VIBES = list({v for a in ARCHETYPES.values() for v in a["vibe_pool"]})


def _sample_location(center: tuple[float, float], radius_km: float) -> tuple[float, float]:
    """Sample lat/lon from Gaussian around center. ~1 deg lat ≈ 111 km."""
    sigma = radius_km / 111.0
    lat = random.gauss(center[0], sigma)
    lon = random.gauss(center[1], sigma / max(0.01, abs(center[0]) * 0.0175))
    return round(lat, 6), round(lon, 6)


def generate_users(n: int = 1000, rng_seed: int = 42) -> list[User]:
    random.seed(rng_seed)
    Faker.seed(rng_seed)

    neighborhood_names = [nb["name"] for nb in NEIGHBORHOODS]
    neighborhood_weights = [nb["weight"] for nb in NEIGHBORHOODS]
    nb_map = {nb["name"]: nb for nb in NEIGHBORHOODS}

    archetype_names = list(ARCHETYPE_WEIGHTS.keys())
    archetype_weights = list(ARCHETYPE_WEIGHTS.values())

    users: list[User] = []
    usernames_seen: set[str] = set()
    now = datetime.now(timezone.utc)

    for _ in range(n):
        nb_name = random.choices(neighborhood_names, weights=neighborhood_weights, k=1)[0]
        nb = nb_map[nb_name]
        archetype_name = random.choices(archetype_names, weights=archetype_weights, k=1)[0]
        arch = ARCHETYPES[archetype_name]

        lat, lon = _sample_location(nb["center"], nb["radius_km"])

        n_cuisine = random.randint(3, 5)
        cuisines = random.sample(arch["cuisine_pool"], min(n_cuisine, len(arch["cuisine_pool"])))
        extra_cuisine = random.sample([c for c in ALL_CUISINES if c not in cuisines], min(random.randint(0, 2), len(ALL_CUISINES) - len(cuisines)))
        cuisines += extra_cuisine

        n_vibes = random.randint(2, 3)
        vibes = random.sample(arch["vibe_pool"], min(n_vibes, len(arch["vibe_pool"])))
        extra_vibes = random.sample([v for v in ALL_VIBES if v not in vibes], min(random.randint(0, 1), len(ALL_VIBES) - len(vibes)))
        vibes += extra_vibes

        age = random.randint(*arch["age_range"])
        price = random.choice(arch["price_range"])

        username = fake.user_name()
        while username in usernames_seen:
            username = fake.user_name() + str(random.randint(0, 9999))
        usernames_seen.add(username)

        email = f"{username}@luna-demo.com"

        created = now - timedelta(days=random.randint(1, 180))
        last_active = created + timedelta(days=random.randint(0, (now - created).days or 1))

        users.append(User(
            id=str(uuid.uuid4()),
            username=username,
            display_name=fake.name(),
            email=email,
            bio=fake.sentence() if random.random() < 0.6 else None,
            avatar_url=f"https://api.dicebear.com/7.x/avataaars/svg?seed={username}",
            home_latitude=lat,
            home_longitude=lon,
            city="Austin",
            neighborhood=nb_name,
            max_travel_distance_km=round(random.uniform(5.0, 20.0), 1),
            cuisine_preferences=cuisines,
            vibe_preferences=vibes,
            price_preference=price,
            age=age,
            archetype=archetype_name,
            created_at=created,
            last_active=last_active,
            total_checkins=0,
        ))

    return users
