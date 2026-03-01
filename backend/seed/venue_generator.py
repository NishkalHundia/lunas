"""Generate realistic synthetic venues across Austin, TX.

Venues are distributed across neighborhoods with category-appropriate
clustering (more bars Downtown, more cafes in Hyde Park, etc.).
Names are generated from template patterns to feel authentic.
"""

import random
import uuid
from datetime import datetime, timezone
from backend.models.venue import Venue

NEIGHBORHOODS = [
    {"name": "Downtown", "center": (30.2672, -97.7431), "radius_km": 2,
     "bias": {"bar": 1.5, "restaurant": 1.2, "activity": 1.3, "cafe": 0.8}},
    {"name": "East Austin", "center": (30.2621, -97.7195), "radius_km": 3,
     "bias": {"bar": 1.3, "restaurant": 1.1, "cafe": 1.4, "activity": 0.9}},
    {"name": "South Congress", "center": (30.2460, -97.7487), "radius_km": 2,
     "bias": {"restaurant": 1.4, "cafe": 1.3, "bar": 1.0, "activity": 0.8}},
    {"name": "North Loop", "center": (30.3172, -97.7231), "radius_km": 2,
     "bias": {"cafe": 1.5, "restaurant": 1.0, "bar": 0.8, "activity": 0.7}},
    {"name": "Zilker", "center": (30.2670, -97.7730), "radius_km": 2,
     "bias": {"activity": 1.5, "restaurant": 1.0, "cafe": 1.0, "bar": 0.7}},
    {"name": "Mueller", "center": (30.2988, -97.7056), "radius_km": 2,
     "bias": {"restaurant": 1.2, "cafe": 1.1, "activity": 1.0, "bar": 0.6}},
    {"name": "West Campus", "center": (30.2849, -97.7414), "radius_km": 1.5,
     "bias": {"bar": 1.4, "restaurant": 1.0, "cafe": 1.2, "activity": 0.8}},
    {"name": "Hyde Park", "center": (30.3060, -97.7270), "radius_km": 1.5,
     "bias": {"cafe": 1.6, "restaurant": 1.1, "bar": 0.6, "activity": 0.5}},
]

VENUE_TEMPLATES = {
    "restaurant": {
        "count": 120,
        "subcategories": {
            "italian": {"vibes": ["romantic", "intimate", "casual"], "price": [2, 3, 4], "capacity": (30, 120)},
            "japanese": {"vibes": ["trendy", "intimate", "authentic"], "price": [2, 3, 4], "capacity": (20, 80)},
            "mexican": {"vibes": ["lively", "casual", "authentic"], "price": [1, 2, 3], "capacity": (40, 150)},
            "american": {"vibes": ["casual", "group-friendly", "lively"], "price": [1, 2, 3], "capacity": (50, 200)},
            "thai": {"vibes": ["casual", "authentic", "chill"], "price": [1, 2], "capacity": (25, 70)},
            "korean": {"vibes": ["trendy", "group-friendly", "authentic"], "price": [2, 3], "capacity": (30, 80)},
            "french": {"vibes": ["romantic", "upscale", "intimate"], "price": [3, 4], "capacity": (20, 60)},
            "indian": {"vibes": ["casual", "authentic", "adventurous"], "price": [1, 2], "capacity": (30, 80)},
            "mediterranean": {"vibes": ["chill", "healthy", "aesthetic"], "price": [2, 3], "capacity": (30, 90)},
            "vietnamese": {"vibes": ["authentic", "casual", "chill"], "price": [1, 2], "capacity": (25, 60)},
            "ethiopian": {"vibes": ["authentic", "adventurous", "group-friendly"], "price": [1, 2], "capacity": (25, 60)},
            "vegan": {"vibes": ["healthy", "aesthetic", "chill"], "price": [2, 3], "capacity": (20, 50)},
        },
    },
    "bar": {
        "count": 60,
        "subcategories": {
            "cocktail_bar": {"vibes": ["trendy", "intimate", "upscale"], "price": [3, 4], "capacity": (30, 80)},
            "dive_bar": {"vibes": ["casual", "lively", "authentic"], "price": [1, 2], "capacity": (40, 100)},
            "sports_bar": {"vibes": ["loud", "group-friendly", "lively"], "price": [1, 2], "capacity": (60, 200)},
            "wine_bar": {"vibes": ["romantic", "chill", "intimate"], "price": [3, 4], "capacity": (20, 50)},
            "rooftop": {"vibes": ["trendy", "aesthetic", "lively"], "price": [3, 4], "capacity": (50, 150)},
        },
    },
    "cafe": {
        "count": 40,
        "subcategories": {
            "coffee_shop": {"vibes": ["chill", "aesthetic", "casual"], "price": [1, 2], "capacity": (15, 40)},
            "brunch_spot": {"vibes": ["trendy", "casual", "aesthetic"], "price": [2, 3], "capacity": (30, 80)},
            "juice_bar": {"vibes": ["healthy", "aesthetic", "chill"], "price": [2, 3], "capacity": (10, 25)},
        },
    },
    "activity": {
        "count": 30,
        "subcategories": {
            "bowling": {"vibes": ["group-friendly", "lively", "casual"], "price": [2, 3], "capacity": (50, 150)},
            "mini_golf": {"vibes": ["group-friendly", "casual", "fun"], "price": [2], "capacity": (30, 80)},
            "escape_room": {"vibes": ["adventurous", "group-friendly"], "price": [2, 3], "capacity": (6, 12)},
            "karaoke": {"vibes": ["lively", "group-friendly", "fun"], "price": [2, 3], "capacity": (10, 30)},
            "arcade": {"vibes": ["casual", "fun", "group-friendly"], "price": [1, 2], "capacity": (40, 100)},
        },
    },
}

ADJECTIVES = ["Golden", "Silver", "Red", "Blue", "Hidden", "Little", "Grand", "Lucky", "Electric", "Velvet",
              "Wild", "Rustic", "Urban", "Sunny", "Cosmic", "Sage", "Ember", "Neon", "Copper", "Ivory"]
NOUNS = ["Oak", "Fox", "Lantern", "Garden", "Table", "Kitchen", "House", "Room", "Cellar", "Den",
         "Porch", "Corner", "Plate", "Vine", "Stone", "Flame", "Leaf", "Bell", "Moon", "Pearl"]
OWNERS = ["Marco", "Sofia", "Kai", "Luna", "Jasper", "Mila", "Theo", "Aria", "Finn", "Nora",
          "Dante", "Rosa", "Luca", "Maya", "Sato", "Priya", "Omar", "Elena", "Jin", "Leila"]

HOURS_TEMPLATES = {
    "restaurant": {"open": "11:00", "close": "22:00", "weekend_close": "23:00"},
    "bar": {"open": "16:00", "close": "02:00", "weekend_close": "02:00"},
    "cafe": {"open": "06:00", "close": "17:00", "weekend_close": "18:00"},
    "activity": {"open": "10:00", "close": "22:00", "weekend_close": "23:00"},
}

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _generate_name(category: str, subcategory: str) -> str:
    pattern = random.choice(["adj_noun", "owner", "the_noun", "sub_adj"])
    if pattern == "adj_noun":
        return f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
    elif pattern == "owner":
        suffix = {"restaurant": "Kitchen", "bar": "Bar", "cafe": "Cafe", "activity": "Place"}.get(category, "Spot")
        return f"{random.choice(OWNERS)}'s {suffix}"
    elif pattern == "the_noun":
        return f"The {random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
    else:
        return f"{random.choice(ADJECTIVES)} {subcategory.replace('_', ' ').title()}"


def _generate_hours(category: str) -> dict:
    tmpl = HOURS_TEMPLATES.get(category, HOURS_TEMPLATES["restaurant"])
    hours = {}
    for day in DAYS:
        close = tmpl["weekend_close"] if day in ("friday", "saturday") else tmpl["close"]
        hours[day] = {"open": tmpl["open"], "close": close}
    return hours


def _sample_location(center: tuple[float, float], radius_km: float) -> tuple[float, float]:
    sigma = radius_km / 111.0
    lat = random.gauss(center[0], sigma)
    lon = random.gauss(center[1], sigma / max(0.01, abs(center[0]) * 0.0175))
    return round(lat, 6), round(lon, 6)


def _generate_address(neighborhood: str) -> str:
    num = random.randint(100, 9999)
    streets = ["Congress Ave", "6th St", "Lamar Blvd", "South 1st St", "Rainey St",
               "Manor Rd", "E Cesar Chavez St", "Burnet Rd", "Guadalupe St", "Red River St",
               "S Lamar Blvd", "N Lamar Blvd", "E 7th St", "W 5th St", "Barton Springs Rd"]
    return f"{num} {random.choice(streets)}, Austin, TX"


def generate_venues(n: int = 250, rng_seed: int = 42) -> list[Venue]:
    random.seed(rng_seed)

    nb_names = [nb["name"] for nb in NEIGHBORHOODS]
    nb_map = {nb["name"]: nb for nb in NEIGHBORHOODS}

    venue_plan: list[tuple[str, str]] = []
    for cat, spec in VENUE_TEMPLATES.items():
        subs = list(spec["subcategories"].keys())
        for _ in range(spec["count"]):
            sub = random.choice(subs)
            venue_plan.append((cat, sub))

    random.shuffle(venue_plan)
    venue_plan = venue_plan[:n]

    names_seen: set[str] = set()
    venues: list[Venue] = []

    for cat, sub in venue_plan:
        nb_weights = [nb_map[name].get("bias", {}).get(cat, 1.0) for name in nb_names]
        nb_name = random.choices(nb_names, weights=nb_weights, k=1)[0]
        nb = nb_map[nb_name]

        lat, lon = _sample_location(nb["center"], nb["radius_km"])
        sub_spec = VENUE_TEMPLATES[cat]["subcategories"][sub]

        name = _generate_name(cat, sub)
        while name in names_seen:
            name = _generate_name(cat, sub)
        names_seen.add(name)

        vibes = list(sub_spec["vibes"])
        price = random.choice(sub_spec["price"])
        capacity = random.randint(*sub_spec["capacity"])
        rating = round(max(2.5, min(5.0, random.gauss(4.1, 0.5))), 1)

        cuisine_tags = [sub] if cat == "restaurant" else []
        if cat == "bar":
            cuisine_tags = ["bar_food", sub]
        elif cat == "cafe":
            cuisine_tags = ["cafe", sub]

        venues.append(Venue(
            id=str(uuid.uuid4()),
            name=name,
            description=f"A {random.choice(vibes)} {sub.replace('_', ' ')} spot in {nb_name}.",
            latitude=lat,
            longitude=lon,
            address=_generate_address(nb_name),
            city="Austin",
            neighborhood=nb_name,
            category=cat,
            subcategory=sub,
            cuisine_tags=cuisine_tags,
            vibe_tags=vibes,
            price_level=price,
            capacity=capacity,
            avg_rating=rating,
            hours=_generate_hours(cat),
            accepts_reservations=cat in ("restaurant", "activity") or random.random() < 0.3,
            min_party_size=1,
            max_party_size=min(20, capacity // 2),
            image_urls=[f"https://picsum.photos/seed/{name.replace(' ', '')}{i}/400/300" for i in range(3)],
            created_at=datetime.now(timezone.utc),
        ))

    return venues
