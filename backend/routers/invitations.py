"""Invitation lifecycle endpoints with automatic booking trigger."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.invitation import Invitation
from backend.models.invitation_group import InvitationGroup
from backend.models.user import User
from backend.models.venue import Venue
from backend.schemas.booking import (
    BookingExceptionResolve,
    InvitationRespond,
    InvitationResponse,
    InvitationSend,
)
from backend.services.booking_agent import evaluate_group_for_auto_booking, resolve_exception, run_booking_agent
from backend.services.embedding_engine import engine


router = APIRouter(prefix="/api/invitations", tags=["invitations"])


@router.post("/send")
def send_invitations(payload: InvitationSend, db: Session = Depends(get_db)):
    if not payload.to_user_ids:
        raise HTTPException(400, "to_user_ids_required")
    organizer = db.query(User).filter(User.id == payload.from_user_id).first()
    venue = db.query(Venue).filter(Venue.id == payload.venue_id).first()
    if not organizer or not venue:
        raise HTTPException(404, "organizer_or_venue_not_found")

    unique_targets = [uid for uid in dict.fromkeys(payload.to_user_ids) if uid != payload.from_user_id]
    if not unique_targets:
        raise HTTPException(400, "no_valid_invitees")

    group = InvitationGroup(
        id=str(uuid.uuid4()),
        organizer_id=payload.from_user_id,
        venue_id=payload.venue_id,
        proposed_date=payload.proposed_date,
        proposed_time_slot=payload.proposed_time_slot,
        threshold_count=max(2, payload.threshold_count),
        status="collecting",
    )
    db.add(group)
    db.flush()

    invites = []
    for to_user in unique_targets:
        invite = Invitation(
            id=str(uuid.uuid4()),
            group_id=group.id,
            from_user_id=payload.from_user_id,
            to_user_id=to_user,
            venue_id=payload.venue_id,
            message=payload.message,
            status="pending",
        )
        db.add(invite)
        invites.append(invite)

    db.commit()
    for invite in invites:
        db.refresh(invite)

    engine.ensure_initialized(db)
    engine.online_update(db, payload.from_user_id, payload.venue_id, "intent")

    return {
        "group_id": group.id,
        "invitations_sent": len(invites),
        "threshold_count": group.threshold_count,
        "from_user": organizer.display_name,
        "venue_name": venue.name,
        "auto_booking_rule": "all_responded_and_threshold_met",
        "invitations": [InvitationResponse.model_validate(invite).model_dump() for invite in invites],
    }


@router.get("/incoming/{user_id}")
def incoming(user_id: str, db: Session = Depends(get_db)):
    invites = db.query(Invitation).filter(
        Invitation.to_user_id == user_id,
        Invitation.status == "pending",
    ).order_by(Invitation.created_at.desc()).all()
    results = []
    for invite in invites:
        sender = db.query(User).filter(User.id == invite.from_user_id).first()
        venue = db.query(Venue).filter(Venue.id == invite.venue_id).first()
        group = db.query(InvitationGroup).filter(InvitationGroup.id == invite.group_id).first()
        group_invites = db.query(Invitation).filter(Invitation.group_id == invite.group_id).all()
        accepted = sum(1 for row in group_invites if row.status == "accepted")
        declined = sum(1 for row in group_invites if row.status == "declined")
        pending = sum(1 for row in group_invites if row.status == "pending")
        results.append(
            {
                **InvitationResponse.model_validate(invite).model_dump(),
                "from_display_name": sender.display_name if sender else "Unknown",
                "from_avatar": sender.avatar_url if sender else None,
                "venue_name": venue.name if venue else "Unknown",
                "venue_category": venue.category if venue else None,
                "venue_neighborhood": venue.neighborhood if venue else None,
                "group_status": group.status if group else "collecting",
                "group_threshold": group.threshold_count if group else 2,
                "group_accepted": accepted,
                "group_declined": declined,
                "group_pending": pending,
                "group_total_invitees": len(group_invites),
            }
        )
    return results


@router.get("/outgoing/{user_id}")
def outgoing(user_id: str, db: Session = Depends(get_db)):
    groups = db.query(InvitationGroup).filter(InvitationGroup.organizer_id == user_id).order_by(InvitationGroup.created_at.desc()).all()
    results = []
    for group in groups:
        venue = db.query(Venue).filter(Venue.id == group.venue_id).first()
        invites = db.query(Invitation).filter(Invitation.group_id == group.id).all()
        people = []
        for invite in invites:
            target = db.query(User).filter(User.id == invite.to_user_id).first()
            people.append(
                {
                    "invitation_id": invite.id,
                    "user_id": invite.to_user_id,
                    "display_name": target.display_name if target else "Unknown",
                    "avatar_url": target.avatar_url if target else None,
                    "status": invite.status,
                    "responded_at": invite.responded_at.isoformat() if invite.responded_at else None,
                }
            )
        accepted = sum(1 for person in people if person["status"] == "accepted")
        declined = sum(1 for person in people if person["status"] == "declined")
        pending = sum(1 for person in people if person["status"] == "pending")
        results.append(
            {
                "group_id": group.id,
                "group_status": group.status,
                "exception_reason": group.exception_reason,
                "booking_id": group.booking_id,
                "venue_id": group.venue_id,
                "venue_name": venue.name if venue else "Unknown",
                "venue_category": venue.category if venue else "",
                "proposed_date": str(group.proposed_date) if group.proposed_date else None,
                "proposed_time_slot": group.proposed_time_slot,
                "threshold_count": group.threshold_count,
                "created_at": group.created_at.isoformat() if group.created_at else None,
                "people": people,
                "summary": {"accepted": accepted, "declined": declined, "pending": pending, "total": len(people)},
                "auto_booking_ready": pending == 0 and (accepted + 1) >= group.threshold_count,
            }
        )
    return results


@router.post("/{invitation_id}/respond")
def respond(invitation_id: str, payload: InvitationRespond, db: Session = Depends(get_db)):
    invite = db.query(Invitation).filter(Invitation.id == invitation_id).first()
    if not invite:
        raise HTTPException(404, "invitation_not_found")
    if invite.status != "pending":
        raise HTTPException(400, f"invitation_already_{invite.status}")
    if payload.status not in {"accepted", "declined"}:
        raise HTTPException(400, "status_must_be_accepted_or_declined")

    invite.status = payload.status
    invite.responded_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(invite)

    if payload.status == "accepted":
        engine.ensure_initialized(db)
        engine.online_update(db, invite.to_user_id, invite.venue_id, "invite_accept")
    else:
        engine.ensure_initialized(db)
        engine.online_update(db, invite.to_user_id, invite.venue_id, "invite_decline")

    evaluation = evaluate_group_for_auto_booking(db, invite.group_id)
    group_invites = db.query(Invitation).filter(Invitation.group_id == invite.group_id).all()
    accepted = sum(1 for row in group_invites if row.status == "accepted")
    declined = sum(1 for row in group_invites if row.status == "declined")
    pending = sum(1 for row in group_invites if row.status == "pending")

    return {
        "invitation_id": invite.id,
        "new_status": invite.status,
        "group_id": invite.group_id,
        "group_summary": {"accepted": accepted, "declined": declined, "pending": pending},
        "all_responded": pending == 0,
        "auto_booking_state": evaluation,
    }


@router.post("/agent/run")
def manual_agent_run(db: Session = Depends(get_db)):
    results = run_booking_agent(db)
    return {"groups_processed": len(results), "results": results}


@router.post("/groups/{group_id}/resolve-exception")
def resolve_group_exception(group_id: str, payload: BookingExceptionResolve, db: Session = Depends(get_db)):
    result = resolve_exception(db, group_id, action=payload.action)
    return result
