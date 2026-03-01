"""Invitation lifecycle smoke test for v2 auto-booking."""

import httpx


BASE = "http://localhost:8000"
client = httpx.Client(base_url=BASE, timeout=45)


def main():
    users = client.get("/api/users?limit=6").json()
    assert len(users) >= 3, "need at least 3 users"
    organizer = users[0]
    invitee_a = users[1]
    invitee_b = users[2]

    venue = client.get("/api/venues?limit=1").json()[0]

    sent = client.post(
        "/api/invitations/send",
        json={
            "from_user_id": organizer["id"],
            "to_user_ids": [invitee_a["id"], invitee_b["id"]],
            "venue_id": venue["id"],
            "proposed_time_slot": "evening",
            "threshold_count": 3,
            "message": "Test auto booking flow",
        },
    ).json()
    print("Group:", sent["group_id"])

    inbox_a = client.get(f"/api/invitations/incoming/{invitee_a['id']}").json()
    inbox_b = client.get(f"/api/invitations/incoming/{invitee_b['id']}").json()
    inv_a = next(i for i in inbox_a if i["group_id"] == sent["group_id"])
    inv_b = next(i for i in inbox_b if i["group_id"] == sent["group_id"])

    r1 = client.post(f"/api/invitations/{inv_a['id']}/respond", json={"status": "accepted"}).json()
    print("After first response:", r1["auto_booking_state"]["status"])
    r2 = client.post(f"/api/invitations/{inv_b['id']}/respond", json={"status": "accepted"}).json()
    print("After second response:", r2["auto_booking_state"]["status"])

    outgoing = client.get(f"/api/invitations/outgoing/{organizer['id']}").json()
    group = next(g for g in outgoing if g["group_id"] == sent["group_id"])
    print("Group status:", group["group_status"])

    bookings = client.get(f"/api/bookings/user/{organizer['id']}").json()
    print("Organizer bookings:", len(bookings))
    print("Invitation smoke test complete.")


if __name__ == "__main__":
    main()
