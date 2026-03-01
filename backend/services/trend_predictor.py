"""Trend forecasting over recent momentum and social intent."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone, date

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.interaction import Interaction
from backend.models.interest import Interest
from backend.models.invitation import Invitation
from backend.models.invitation_group import InvitationGroup
from backend.models.venue import Venue


def predict_trending(db: Session, city: str, target_date: date, time_slot: str, top_n: int = 10) -> list[dict]:
    venues = db.query(Venue).filter(Venue.city == city).all()
    if not venues:
        return []
    venue_ids = [venue.id for venue in venues]

    now = datetime.now(timezone.utc)
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)
    prev_7d = now - timedelta(days=14)

    recent = db.query(
        Interaction.venue_id,
        Interaction.interaction_type,
        func.count(Interaction.id),
    ).filter(
        Interaction.venue_id.in_(venue_ids),
        Interaction.created_at >= last_24h.replace(tzinfo=None),
    ).group_by(Interaction.venue_id, Interaction.interaction_type).all()

    weekly = db.query(
        Interaction.venue_id,
        func.count(Interaction.id),
    ).filter(
        Interaction.venue_id.in_(venue_ids),
        Interaction.created_at >= last_7d.replace(tzinfo=None),
    ).group_by(Interaction.venue_id).all()

    prev_week = db.query(
        Interaction.venue_id,
        func.count(Interaction.id),
    ).filter(
        Interaction.venue_id.in_(venue_ids),
        Interaction.created_at < last_7d.replace(tzinfo=None),
        Interaction.created_at >= prev_7d.replace(tzinfo=None),
    ).group_by(Interaction.venue_id).all()

    active_interests = db.query(
        Interest.venue_id,
        func.count(Interest.id),
    ).filter(
        Interest.venue_id.in_(venue_ids),
        Interest.status == "active",
    ).group_by(Interest.venue_id).all()

    groups = db.query(InvitationGroup).filter(
        InvitationGroup.venue_id.in_(venue_ids),
        InvitationGroup.status == "collecting",
    ).all()
    group_ids = [group.id for group in groups]
    accepted_invites = defaultdict(int)
    if group_ids:
        accepted_rows = db.query(Invitation.venue_id, func.count(Invitation.id)).filter(
            Invitation.group_id.in_(group_ids),
            Invitation.status == "accepted",
        ).group_by(Invitation.venue_id).all()
        for venue_id, count in accepted_rows:
            accepted_invites[venue_id] = count

    by_recent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for venue_id, interaction_type, count in recent:
        by_recent[venue_id][interaction_type] = count
    by_week = {venue_id: count for venue_id, count in weekly}
    by_prev_week = {venue_id: count for venue_id, count in prev_week}
    by_interests = {venue_id: count for venue_id, count in active_interests}

    predictions = []
    for venue in venues:
        rec = by_recent.get(venue.id, {})
        saves = rec.get("save", 0)
        views = rec.get("view", 0)
        shares = rec.get("share", 0)
        checkins = rec.get("checkin", 0)
        intents = rec.get("intent", 0)

        interests = by_interests.get(venue.id, 0)
        accepted = accepted_invites.get(venue.id, 0)
        week_count = by_week.get(venue.id, 0)
        prev_count = by_prev_week.get(venue.id, 0)
        if prev_count == 0:
            momentum = 0.35 if week_count > 0 else 0.0
        else:
            momentum = (week_count - prev_count) / max(prev_count, 1)
        momentum = max(-1.0, min(1.5, momentum))

        score = (
            0.22 * min(1.0, saves / 12.0)
            + 0.16 * min(1.0, checkins / 8.0)
            + 0.12 * min(1.0, shares / 6.0)
            + 0.12 * min(1.0, intents / 10.0)
            + 0.15 * min(1.0, interests / 6.0)
            + 0.08 * min(1.0, accepted / 5.0)
            + 0.08 * ((momentum + 1.0) / 2.0)
            + 0.07 * ((venue.avg_rating or 4.0) / 5.0)
        )
        score = max(0.0, min(1.0, score + 0.03 * (venue.popularity_prior or 0.0)))

        predictions.append(
            {
                "venue": venue,
                "trend_score": round(score, 5),
                "momentum": round(momentum, 5),
                "active_interests": interests + accepted,
                "predicted_attendance": int(max(1, 1.7 * interests + 1.2 * accepted + 0.4 * saves + 0.7 * checkins)),
            }
        )

    predictions.sort(key=lambda item: -item["trend_score"])
    return predictions[:top_n]
