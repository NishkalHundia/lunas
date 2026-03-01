"""Quick smoke test for Luna v2 API."""

import json

import httpx


BASE = "http://localhost:8000"
client = httpx.Client(timeout=60)


def show(name: str, response: httpx.Response):
    payload = response.json()
    print(f"\n--- {name} ---")
    print(f"Status: {response.status_code}")
    if isinstance(payload, list):
        print(f"Items: {len(payload)}")
        if payload:
            print(json.dumps(payload[0], indent=2, default=str)[:500])
    else:
        print(json.dumps(payload, indent=2, default=str)[:800])
    return payload


def main():
    users = show("Users", client.get(f"{BASE}/api/users?limit=3"))
    user_id = users[0]["id"]

    venues = show("Venues", client.get(f"{BASE}/api/venues?limit=3"))
    venue_id = venues[0]["id"]

    show("Recommendations", client.get(f"{BASE}/api/recommendations/{user_id}?limit=5"))
    show("Taste Profile", client.get(f"{BASE}/api/embeddings/taste-profile/{user_id}"))
    show("Trending", client.get(f"{BASE}/api/analytics/trending?limit=5"))

    show(
        "Track Save",
        client.post(
            f"{BASE}/api/analytics/track",
            json={
                "user_id": user_id,
                "venue_id": venue_id,
                "interaction_type": "save",
                "source": "test",
                "context": {"from": "test_api"},
            },
        ),
    )

    show(
        "Create Interest",
        client.post(
            f"{BASE}/api/interests",
            json={
                "user_id": user_id,
                "venue_id": venue_id,
                "preferred_date": "2026-03-03",
                "preferred_time_slot": "evening",
                "intent_strength": 0.8,
                "source": "test",
            },
        ),
    )

    show("Bookings", client.get(f"{BASE}/api/bookings/user/{user_id}"))
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
