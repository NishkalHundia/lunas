"""Interaction tracking and engagement analytics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.friendship import Friendship
from backend.models.interaction import Interaction
from backend.models.user import User
from backend.models.venue import Venue
from backend.services.embedding_engine import engine


def track_interaction(
    db: Session,
    user_id: str,
    venue_id: str,
    interaction_type: str,
    view_duration_seconds: float | None = None,
    scroll_depth: float | None = None,
    source: str = "feed",
    session_id: str | None = None,
    context: dict | None = None,
) -> tuple[Interaction, dict]:
    row = Interaction(
        user_id=user_id,
        venue_id=venue_id,
        interaction_type=interaction_type,
        view_duration_seconds=view_duration_seconds,
        scroll_depth=scroll_depth,
        source=source,
        session_id=session_id,
        context=context or {},
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_active = datetime.now(timezone.utc)
        if interaction_type == "checkin":
            user.total_checkins = (user.total_checkins or 0) + 1
        db.commit()

    if interaction_type == "checkin":
        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
        friend_rows = db.query(Friendship).filter(
            ((Friendship.user_id == user_id) | (Friendship.friend_id == user_id)),
            Friendship.status == "accepted",
        ).all()
        friend_ids = {f.friend_id if f.user_id == user_id else f.user_id for f in friend_rows}
        if friend_ids:
            nearby_friend_checkins = db.query(Interaction).filter(
                Interaction.user_id.in_(list(friend_ids)),
                Interaction.venue_id == venue_id,
                Interaction.interaction_type == "checkin",
                Interaction.created_at >= cutoff.replace(tzinfo=None),
            ).all()
            if nearby_friend_checkins:
                for row_friendship in friend_rows:
                    row_friendship.interaction_score = min(5.0, (row_friendship.interaction_score or 0.0) + 0.08)
                db.commit()

    update = engine.online_update(db, user_id, venue_id, interaction_type, view_duration_seconds)
    return row, update


def get_venue_engagement_summary(db: Session, venue_id: str) -> dict:
    base = db.query(Interaction).filter(Interaction.venue_id == venue_id)
    total_views = base.filter(Interaction.interaction_type == "view").count()
    total_saves = base.filter(Interaction.interaction_type == "save").count()
    total_checkins = base.filter(Interaction.interaction_type == "checkin").count()
    total_shares = base.filter(Interaction.interaction_type == "share").count()

    avg_dur = db.query(func.avg(Interaction.view_duration_seconds)).filter(
        Interaction.venue_id == venue_id,
        Interaction.interaction_type == "view",
        Interaction.view_duration_seconds.isnot(None),
    ).scalar()

    now = datetime.now(timezone.utc)
    last_week = now - timedelta(days=7)
    prev_week = now - timedelta(days=14)
    recent = base.filter(Interaction.created_at >= last_week.replace(tzinfo=None)).count()
    prev = base.filter(
        Interaction.created_at < last_week.replace(tzinfo=None),
        Interaction.created_at >= prev_week.replace(tzinfo=None),
    ).count()
    if prev == 0:
        trend_direction = "up" if recent > 0 else "stable"
    else:
        ratio = recent / max(prev, 1)
        if ratio > 1.12:
            trend_direction = "up"
        elif ratio < 0.88:
            trend_direction = "down"
        else:
            trend_direction = "stable"

    return {
        "venue_id": venue_id,
        "total_views": total_views,
        "total_saves": total_saves,
        "total_checkins": total_checkins,
        "total_shares": total_shares,
        "avg_view_duration": round(float(avg_dur), 2) if avg_dur else None,
        "trend_direction": trend_direction,
    }


def get_user_activity_summary(db: Session, user_id: str) -> dict:
    events = db.query(Interaction).filter(Interaction.user_id == user_id).all()
    if not events:
        return {
            "user_id": user_id,
            "total_interactions": 0,
            "top_categories": [],
            "most_active_hours": [],
            "favorite_venues": [],
            "friend_interaction_count": 0,
        }

    venue_ids = list({event.venue_id for event in events})
    venues = db.query(Venue).filter(Venue.id.in_(venue_ids)).all()
    venue_map = {venue.id: venue for venue in venues}

    category_counter = Counter()
    venue_counter = Counter()
    hour_counter = Counter()
    for event in events:
        venue_counter[event.venue_id] += 1
        if event.created_at:
            hour_counter[event.created_at.hour] += 1
        venue = venue_map.get(event.venue_id)
        if venue:
            category_counter[venue.category] += 1

    favorite_venues = []
    for venue_id, count in venue_counter.most_common(5):
        venue = venue_map.get(venue_id)
        if venue:
            favorite_venues.append({"venue_id": venue_id, "name": venue.name, "visits": count})

    friendships = db.query(Friendship).filter(
        ((Friendship.user_id == user_id) | (Friendship.friend_id == user_id)),
        Friendship.status == "accepted",
    ).all()
    friend_interaction_count = int(sum(friend.interaction_score or 0 for friend in friendships))

    return {
        "user_id": user_id,
        "total_interactions": len(events),
        "top_categories": [{"category": cat, "count": count} for cat, count in category_counter.most_common(5)],
        "most_active_hours": [hour for hour, _count in hour_counter.most_common(5)],
        "favorite_venues": favorite_venues,
        "friend_interaction_count": friend_interaction_count,
    }
