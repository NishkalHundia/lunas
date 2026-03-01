"""Embedding-focused smoke tests."""

import httpx


BASE = "http://localhost:8000"
client = httpx.Client(timeout=90)


def main():
    users = client.get(f"{BASE}/api/users?limit=2").json()
    user_id = users[0]["id"]
    print(f"Testing embeddings for {users[0]['display_name']}")

    profile = client.get(f"{BASE}/api/embeddings/taste-profile/{user_id}").json()
    print(f"Top concepts: {[c['label'] for c in profile.get('top_concepts', [])[:3]]}")

    recs = client.get(f"{BASE}/api/embeddings/recommend/{user_id}?limit=5").json()
    print(f"Recommendations: {len(recs)}")
    if recs:
        top = recs[0]
        print(f"Top venue: {top['venue']['name']} similarity={top['embedding_similarity']:.3f}")

    people = client.get(f"{BASE}/api/embeddings/people/{user_id}?limit=5").json()
    print(f"Compatible people: {len(people)}")

    if recs:
        venue_id = recs[0]["venue"]["id"]
        update = client.post(
            f"{BASE}/api/embeddings/interact",
            params={
                "user_id": user_id,
                "venue_id": venue_id,
                "interaction_type": "save",
                "view_duration": 42,
            },
        ).json()
        print(f"Drift after save: {update['drift']['drift']}")

    concepts = client.get(f"{BASE}/api/embeddings/concepts").json()
    print(f"Concept catalog size: {len(concepts)}")
    print("Embedding test complete.")


if __name__ == "__main__":
    main()
