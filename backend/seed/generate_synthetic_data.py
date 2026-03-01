"""Generate synthetic graph, interactions, invite state, and trained embeddings.

Usage:
    python -m backend.seed.generate_synthetic_data
"""

from __future__ import annotations

import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import NUM_SEED_INTERACTIONS, NUM_SEED_USERS, NUM_SEED_VENUES, SEED_RANDOM_STATE
from backend.database import Base, SessionLocal, engine
from backend.models import Booking, Friendship, Interaction, Interest, Invitation, InvitationGroup, User, Venue
from backend.seed.interaction_generator import generate_interactions
from backend.seed.social_graph_generator import generate_social_graph
from backend.seed.user_generator import generate_users
from backend.seed.venue_generator import generate_venues
from backend.services.booking_agent import run_booking_agent
from backend.services.embedding_engine import engine as embedding_engine


def _generate_invites(db, users: list[User], venues: list[Venue], friendships: list[Friendship], rng: random.Random):
    friend_map: dict[str, set[str]] = {user.id: set() for user in users}
    for row in friendships:
        friend_map.setdefault(row.user_id, set()).add(row.friend_id)
        friend_map.setdefault(row.friend_id, set()).add(row.user_id)

    groups = []
    invites = []
    for _ in range(max(60, len(users) // 10)):
        organizer = rng.choice(users)
        friends = list(friend_map.get(organizer.id, []))
        if len(friends) < 2:
            continue
        venue = rng.choice(venues)
        selected = rng.sample(friends, k=min(len(friends), rng.randint(2, 5)))
        group = InvitationGroup(
            id=str(uuid.uuid4()),
            organizer_id=organizer.id,
            venue_id=venue.id,
            proposed_date=(datetime.now(timezone.utc) + timedelta(days=rng.randint(1, 10))).date(),
            proposed_time_slot=rng.choice(["morning", "afternoon", "evening", "night"]),
            threshold_count=rng.choice([2, 3, 3, 4]),
            status="collecting",
        )
        groups.append(group)
        for target_id in selected:
            status = rng.choices(["pending", "accepted", "declined"], weights=[0.45, 0.40, 0.15], k=1)[0]
            invite = Invitation(
                id=str(uuid.uuid4()),
                group_id=group.id,
                from_user_id=organizer.id,
                to_user_id=target_id,
                venue_id=venue.id,
                message=rng.choice(
                    [
                        "You in for this?",
                        "This place looks perfect for us.",
                        "Let's make this happen.",
                        "I think you'll love this spot.",
                    ]
                ),
                status=status,
                responded_at=(datetime.now(timezone.utc) - timedelta(hours=rng.randint(1, 48))) if status != "pending" else None,
            )
            invites.append(invite)
    db.add_all(groups)
    db.add_all(invites)
    db.commit()


def seed_database():
    rng = random.Random(SEED_RANDOM_STATE)
    print("=" * 70)
    print("Luna Social v2 - Synthetic Data + Trained Embeddings")
    print("=" * 70)

    print("\n[1/7] Resetting schema...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("  OK schema rebuilt")

    db = SessionLocal()
    try:
        print(f"\n[2/7] Generating users ({NUM_SEED_USERS})...")
        t0 = time.time()
        users = generate_users(NUM_SEED_USERS, SEED_RANDOM_STATE)
        for user in users:
            if not user.preferred_time_slots:
                user.preferred_time_slots = rng.sample(["morning", "afternoon", "evening", "night"], k=rng.randint(1, 2))
        db.add_all(users)
        db.commit()
        print(f"  OK users in {time.time()-t0:.1f}s")

        print(f"\n[3/7] Generating venues ({NUM_SEED_VENUES})...")
        t0 = time.time()
        venues = generate_venues(NUM_SEED_VENUES, SEED_RANDOM_STATE)
        for venue in venues:
            venue.popularity_prior = round(rng.uniform(0.0, 1.0), 3)
        db.add_all(venues)
        db.commit()
        print(f"  OK venues in {time.time()-t0:.1f}s")

        print("\n[4/7] Generating friendship graph...")
        t0 = time.time()
        friendships = generate_social_graph(users, SEED_RANDOM_STATE)
        batch = 1000
        for i in range(0, len(friendships), batch):
            db.add_all(friendships[i : i + batch])
            db.commit()
        print(f"  OK friendships={len(friendships)} in {time.time()-t0:.1f}s")

        print("\n[5/7] Generating interactions + interests...")
        t0 = time.time()
        interactions_target = max(30000, NUM_SEED_INTERACTIONS)
        actions_per_user_day = 3.2
        estimated_days = max(28, min(60, int(interactions_target / max(1, NUM_SEED_USERS * actions_per_user_day))))
        interactions, interests = generate_interactions(
            users,
            venues,
            friendships,
            n_days=estimated_days,
            rng_seed=SEED_RANDOM_STATE,
        )
        for i in range(0, len(interactions), batch):
            db.add_all(interactions[i : i + batch])
            db.commit()
        for i in range(0, len(interests), batch):
            db.add_all(interests[i : i + batch])
            db.commit()
        print(f"  OK interactions={len(interactions)} interests={len(interests)} in {time.time()-t0:.1f}s")

        print("\n[6/7] Generating invitation groups + partial responses...")
        _generate_invites(db, users, venues, friendships, rng)
        agent_results = run_booking_agent(db)
        print(f"  OK invitations seeded, auto-booked/exception groups processed={len(agent_results)}")

        print("\n[7/7] Training embedding model and computing explainability profiles...")
        t0 = time.time()
        embedding_engine.train_from_synthetic_data(db)
        print(f"  OK trained in {time.time()-t0:.1f}s")

        for user in users:
            checkins = db.query(Interaction).filter(
                Interaction.user_id == user.id,
                Interaction.interaction_type == "checkin",
            ).count()
            user.total_checkins = checkins
        db.commit()

        print("\n" + "-" * 70)
        print(f"Users: {db.query(User).count()}")
        print(f"Venues: {db.query(Venue).count()}")
        print(f"Friendships: {db.query(Friendship).count()}")
        print(f"Interactions: {db.query(Interaction).count()}")
        print(f"Interests: {db.query(Interest).count()}")
        print(f"Invitation Groups: {db.query(InvitationGroup).count()}")
        print(f"Bookings: {db.query(Booking).count()}")
        print("-" * 70)
        print("Seed + training complete.")
        print("=" * 70)
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
