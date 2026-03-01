"""Invitation orchestration and automated booking agent."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone, time, date

from sqlalchemy.orm import Session

from backend.config import AUTO_BOOKING_MIN_ACCEPTED
from backend.models.booking import Booking
from backend.models.interaction import Interaction
from backend.models.invitation import Invitation
from backend.models.invitation_group import InvitationGroup
from backend.models.user import User
from backend.models.venue import Venue
from backend.services.embedding_engine import engine


SLOT_HOURS = {
    "morning": 10,
    "afternoon": 13,
    "evening": 19,
    "night": 21,
}


def _slot_from_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _resolve_requested_time(value: str | None) -> tuple[str, time]:
    default_slot = "evening"
    if not value:
        return default_slot, time(SLOT_HOURS[default_slot], 0)

    raw = value.strip().lower()
    if raw in SLOT_HOURS:
        return raw, time(SLOT_HOURS[raw], 0)

    parts = raw.split(":")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        hour = int(parts[0])
        minute = int(parts[1])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return _slot_from_hour(hour), time(hour, minute)

    return default_slot, time(SLOT_HOURS[default_slot], 0)


def _make_mock_reservation(venue: Venue, party_size: int) -> tuple[bool, str | None, dict]:
    reliability = venue.booking_reliability or 0.92
    success = random.random() < reliability
    code = f"LUNA-{venue.id[:4].upper()}-{random.randint(1000, 9999)}" if success else None
    payload = {
        "provider": venue.booking_provider or "mock",
        "simulated_latency_ms": random.randint(220, 920),
        "availability_score": round(random.uniform(0.52, 0.99), 3),
    }
    return success, code, payload


def _group_snapshot(invites: list[Invitation]) -> dict:
    accepted = [inv for inv in invites if inv.status == "accepted"]
    declined = [inv for inv in invites if inv.status == "declined"]
    pending = [inv for inv in invites if inv.status == "pending"]
    return {
        "accepted": len(accepted),
        "declined": len(declined),
        "pending": len(pending),
        "all_responded": len(pending) == 0,
    }


def _choose_time_slot(group: InvitationGroup, invites: list[Invitation]) -> tuple[str, time]:
    _ = invites
    return _resolve_requested_time(group.proposed_time_slot)


def _log(group_id: str, step: str, payload: dict | None = None) -> dict:
    entry = {
        "id": str(uuid.uuid4()),
        "group_id": group_id,
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if payload:
        entry.update(payload)
    return entry


def evaluate_group_for_auto_booking(db: Session, group_id: str) -> dict:
    group = db.query(InvitationGroup).filter(InvitationGroup.id == group_id).first()
    if not group:
        return {"status": "missing_group"}
    if group.status in {"auto_booked", "cancelled"}:
        return {"status": group.status}

    invites = db.query(Invitation).filter(Invitation.group_id == group.id).all()
    snapshot = _group_snapshot(invites)
    accepted_invites = [inv for inv in invites if inv.status == "accepted"]
    accepted_attendees = [group.organizer_id] + [inv.to_user_id for inv in accepted_invites]
    accepted_attendees = list(dict.fromkeys(accepted_attendees))
    threshold = max(2, group.threshold_count or AUTO_BOOKING_MIN_ACCEPTED)

    if not snapshot["all_responded"]:
        return {
            "status": "collecting",
            "snapshot": snapshot,
            "threshold": threshold,
            "accepted_attendees": len(accepted_attendees),
        }

    if len(accepted_attendees) < threshold:
        group.status = "exception"
        group.exception_reason = "threshold_not_met"
        db.commit()
        return {
            "status": "exception",
            "reason": group.exception_reason,
            "snapshot": snapshot,
            "threshold": threshold,
            "accepted_attendees": len(accepted_attendees),
        }

    venue = db.query(Venue).filter(Venue.id == group.venue_id).first()
    if not venue:
        group.status = "exception"
        group.exception_reason = "venue_missing"
        db.commit()
        return {"status": "exception", "reason": "venue_missing"}

    slot, reservation_time = _choose_time_slot(group, invites)
    reservation_date = group.proposed_date or (datetime.now(timezone.utc) + timedelta(days=1)).date()
    success, code, provider_payload = _make_mock_reservation(venue, len(accepted_attendees))
    now = datetime.now(timezone.utc).isoformat()
    agent_log = [
        _log(group.id, "all_responses_collected", snapshot),
        _log(group.id, "threshold_evaluated", {"threshold": threshold, "accepted_attendees": len(accepted_attendees)}),
        _log(group.id, "reservation_attempted", {"slot": slot, "reservation_date": str(reservation_date), "party_size": len(accepted_attendees)}),
        _log(group.id, "reservation_result", {"success": success, "confirmation_code": code, "provider_payload": provider_payload}),
    ]

    booking = Booking(
        id=str(uuid.uuid4()),
        invitation_group_id=group.id,
        venue_id=group.venue_id,
        organizer_id=group.organizer_id,
        attendee_ids=accepted_attendees,
        attendee_status_map={uid: "accepted" for uid in accepted_attendees},
        reservation_date=reservation_date,
        reservation_time=reservation_time,
        party_size=len(accepted_attendees),
        status="confirmed" if success else "needs_manual_review",
        confirmation_code=code,
        provider=venue.booking_provider or "mock",
        provider_payload=provider_payload,
        booking_method="auto_agent",
        manual_review_required=not success,
        exception_reason=None if success else "provider_failure",
        agent_log=agent_log,
    )
    db.add(booking)
    db.flush()

    group.booking_id = booking.id
    if success:
        group.status = "auto_booked"
        group.exception_reason = None
    else:
        group.status = "exception"
        group.exception_reason = "provider_failure"

    for attendee in accepted_attendees:
        interaction = Interaction(
            user_id=attendee,
            venue_id=group.venue_id,
            interaction_type="intent",
            weight=1.1,
            source="booking_agent",
            context={"group_id": group.id, "booking_id": booking.id},
        )
        db.add(interaction)
        engine.online_update(db, attendee, group.venue_id, "intent")

    db.commit()
    db.refresh(booking)
    db.refresh(group)

    return {
        "status": "auto_booked" if success else "exception",
        "group_id": group.id,
        "booking_id": booking.id,
        "booking_status": booking.status,
        "confirmation_code": booking.confirmation_code,
        "snapshot": snapshot,
        "threshold": threshold,
        "accepted_attendees": len(accepted_attendees),
    }


def run_booking_agent(db: Session) -> list[dict]:
    groups = db.query(InvitationGroup).filter(InvitationGroup.status == "collecting").all()
    results = []
    for group in groups:
        result = evaluate_group_for_auto_booking(db, group.id)
        if result.get("status") in {"auto_booked", "exception"}:
            results.append(result)
    return results


def resolve_exception(db: Session, group_id: str, action: str = "retry") -> dict:
    group = db.query(InvitationGroup).filter(InvitationGroup.id == group_id).first()
    if not group:
        return {"status": "missing_group"}
    if group.status != "exception":
        return {"status": "not_exception", "group_status": group.status}

    if action == "cancel":
        group.status = "cancelled"
        db.commit()
        return {"status": "cancelled", "group_id": group_id}

    group.status = "collecting"
    group.exception_reason = None
    db.commit()
    return evaluate_group_for_auto_booking(db, group_id)


def create_solo_booking(
    db: Session,
    user_id: str,
    venue_id: str,
    reservation_date: date | None = None,
    time_slot: str | None = None,
) -> dict:
    user = db.query(User).filter(User.id == user_id).first()
    venue = db.query(Venue).filter(Venue.id == venue_id).first()

    if not user:
        return {"status": "missing_user"}
    if not venue:
        return {"status": "missing_venue"}

    slot, reservation_time = _resolve_requested_time(time_slot)
    reservation_date = reservation_date or (datetime.now(timezone.utc) + timedelta(days=1)).date()

    attendees = [user.id]
    success, code, payload = _make_mock_reservation(venue, len(attendees))
    booking_id = str(uuid.uuid4())

    booking = Booking(
        id=booking_id,
        invitation_group_id=None,
        venue_id=venue.id,
        organizer_id=user.id,
        attendee_ids=attendees,
        attendee_status_map={user.id: "accepted"},
        reservation_date=reservation_date,
        reservation_time=reservation_time,
        party_size=1,
        status="confirmed" if success else "needs_manual_review",
        confirmation_code=code,
        provider=venue.booking_provider or "mock",
        provider_payload=payload,
        booking_method="direct_solo",
        manual_review_required=not success,
        exception_reason=None if success else "provider_failure",
        agent_log=[
            _log(
                booking_id,
                "direct_solo_booking_requested",
                {
                    "user_id": user.id,
                    "time_slot": slot,
                    "reservation_time": reservation_time.strftime("%H:%M"),
                    "reservation_date": str(reservation_date),
                },
            ),
            _log(
                booking_id,
                "direct_solo_booking_result",
                {"success": success, "confirmation_code": code, "provider_payload": payload},
            ),
        ],
    )
    db.add(booking)

    db.add(
        Interaction(
            user_id=user.id,
            venue_id=venue.id,
            interaction_type="intent",
            weight=1.0,
            source="direct_solo_booking",
            context={"booking_id": booking.id},
        )
    )

    db.commit()
    db.refresh(booking)

    # Apply online update after booking persistence.
    engine.online_update(db, user.id, venue.id, "intent")
    return {"status": booking.status, "booking": booking}


def create_direct_booking(
    db: Session,
    organizer_id: str,
    friend_id: str,
    venue_id: str,
    reservation_date: date | None = None,
    time_slot: str | None = None,
) -> dict:
    organizer = db.query(User).filter(User.id == organizer_id).first()
    friend = db.query(User).filter(User.id == friend_id).first()
    venue = db.query(Venue).filter(Venue.id == venue_id).first()

    if not organizer or not friend:
        return {"status": "missing_user"}
    if not venue:
        return {"status": "missing_venue"}
    if organizer.id == friend.id:
        return {"status": "invalid_pair"}

    slot, reservation_time = _resolve_requested_time(time_slot)
    reservation_date = reservation_date or (datetime.now(timezone.utc) + timedelta(days=1)).date()

    attendees = list(dict.fromkeys([organizer.id, friend.id]))
    success, code, payload = _make_mock_reservation(venue, len(attendees))
    booking_id = str(uuid.uuid4())

    booking = Booking(
        id=booking_id,
        invitation_group_id=None,
        venue_id=venue.id,
        organizer_id=organizer.id,
        attendee_ids=attendees,
        attendee_status_map={uid: "accepted" for uid in attendees},
        reservation_date=reservation_date,
        reservation_time=reservation_time,
        party_size=len(attendees),
        status="confirmed" if success else "needs_manual_review",
        confirmation_code=code,
        provider=venue.booking_provider or "mock",
        provider_payload=payload,
        booking_method="direct_pair",
        manual_review_required=not success,
        exception_reason=None if success else "provider_failure",
        agent_log=[
            _log(
                booking_id,
                "direct_pair_booking_requested",
                {
                    "organizer_id": organizer.id,
                    "friend_id": friend.id,
                    "time_slot": slot,
                    "reservation_time": reservation_time.strftime("%H:%M"),
                    "reservation_date": str(reservation_date),
                },
            ),
            _log(
                booking_id,
                "direct_pair_booking_result",
                {"success": success, "confirmation_code": code, "provider_payload": payload},
            ),
        ],
    )
    db.add(booking)

    for attendee_id in attendees:
        db.add(
            Interaction(
                user_id=attendee_id,
                venue_id=venue.id,
                interaction_type="intent",
                weight=1.05,
                source="direct_pair_booking",
                context={"booking_id": booking.id, "initiator_id": organizer.id, "friend_id": friend.id},
            )
        )

    db.commit()
    db.refresh(booking)

    # Apply online updates after persistence so the booking state is durable even if updates fail.
    for attendee_id in attendees:
        engine.online_update(db, attendee_id, venue.id, "intent")

    return {"status": booking.status, "booking": booking}


def handle_manual_review(
    db: Session,
    booking_id: str,
    actor_user_id: str,
    action: str,
    confirmation_code: str | None = None,
) -> dict:
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        return {"status": "missing_booking"}
    if booking.organizer_id != actor_user_id:
        return {"status": "not_organizer"}

    valid_actions = {"confirm", "retry_provider", "cancel"}
    if action not in valid_actions:
        return {"status": "invalid_action", "allowed": sorted(valid_actions)}

    if action != "cancel" and booking.status == "cancelled":
        return {"status": "invalid_state", "booking_status": booking.status}

    group = None
    if booking.invitation_group_id:
        group = db.query(InvitationGroup).filter(InvitationGroup.id == booking.invitation_group_id).first()

    if action == "confirm":
        booking.status = "confirmed"
        booking.manual_review_required = False
        booking.exception_reason = None
        if confirmation_code:
            booking.confirmation_code = confirmation_code.strip()
        elif not booking.confirmation_code:
            booking.confirmation_code = f"MAN-{booking.id[:6].upper()}-{random.randint(1000, 9999)}"
        booking.agent_log = (booking.agent_log or []) + [
            _log(
                booking.invitation_group_id or booking.id,
                "manual_review_confirmed",
                {"actor_user_id": actor_user_id, "confirmation_code": booking.confirmation_code},
            )
        ]
        if group:
            group.status = "auto_booked"
            group.exception_reason = None
        db.commit()
        return {
            "status": "confirmed",
            "booking_id": booking.id,
            "booking_status": booking.status,
            "confirmation_code": booking.confirmation_code,
        }

    if action == "retry_provider":
        venue = db.query(Venue).filter(Venue.id == booking.venue_id).first()
        if not venue:
            return {"status": "invalid_state", "reason": "venue_missing"}
        success, code, payload = _make_mock_reservation(venue, booking.party_size)
        booking.provider_payload = payload
        booking.agent_log = (booking.agent_log or []) + [
            _log(booking.invitation_group_id or booking.id, "manual_review_retry_requested", {"actor_user_id": actor_user_id}),
            _log(
                booking.invitation_group_id or booking.id,
                "manual_review_retry_result",
                {"success": success, "confirmation_code": code, "provider_payload": payload},
            ),
        ]
        if success:
            booking.status = "confirmed"
            booking.manual_review_required = False
            booking.exception_reason = None
            booking.confirmation_code = code or booking.confirmation_code
            if group:
                group.status = "auto_booked"
                group.exception_reason = None
            db.commit()
            return {
                "status": "confirmed",
                "booking_id": booking.id,
                "booking_status": booking.status,
                "confirmation_code": booking.confirmation_code,
            }

        booking.status = "needs_manual_review"
        booking.manual_review_required = True
        booking.exception_reason = "provider_failure"
        if group:
            group.status = "exception"
            group.exception_reason = "provider_failure"
        db.commit()
        return {
            "status": "still_requires_manual_review",
            "booking_id": booking.id,
            "booking_status": booking.status,
            "exception_reason": booking.exception_reason,
        }

    booking.status = "cancelled"
    booking.manual_review_required = False
    booking.exception_reason = "cancelled_by_organizer"
    booking.agent_log = (booking.agent_log or []) + [
        _log(booking.invitation_group_id or booking.id, "manual_review_cancelled", {"actor_user_id": actor_user_id})
    ]
    if group:
        group.status = "cancelled"
        group.exception_reason = "cancelled_by_organizer"
    db.commit()
    return {"status": "cancelled", "booking_id": booking.id, "booking_status": booking.status}
