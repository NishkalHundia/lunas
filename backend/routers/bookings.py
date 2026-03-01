from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.booking import Booking
from backend.models.invitation_group import InvitationGroup
from backend.schemas.booking import BookingResponse, DirectBookingCreate, ManualReviewAction, SoloBookingCreate
from backend.services.booking_agent import (
    create_direct_booking,
    create_solo_booking,
    handle_manual_review,
    resolve_exception,
    run_booking_agent,
)


router = APIRouter(prefix="/api/bookings", tags=["bookings"])


@router.get("/user/{user_id}", response_model=list[BookingResponse])
def user_bookings(user_id: str, db: Session = Depends(get_db)):
    rows = db.query(Booking).order_by(Booking.created_at.desc()).all()
    return [row for row in rows if user_id == row.organizer_id or user_id in (row.attendee_ids or [])]


@router.get("/{booking_id}", response_model=BookingResponse)
def get_booking(booking_id: str, db: Session = Depends(get_db)):
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        raise HTTPException(404, "booking_not_found")
    return booking


@router.post("/{booking_id}/cancel")
def cancel_booking(
    booking_id: str,
    user_id: str = Query(...),
    db: Session = Depends(get_db),
):
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        raise HTTPException(404, "booking_not_found")
    if user_id != booking.organizer_id and user_id not in (booking.attendee_ids or []):
        raise HTTPException(403, "not_allowed")

    booking.status = "cancelled"
    booking.manual_review_required = False
    booking.exception_reason = None
    db.commit()

    if booking.invitation_group_id:
        group = db.query(InvitationGroup).filter(InvitationGroup.id == booking.invitation_group_id).first()
        if group and group.status != "cancelled":
            group.status = "cancelled"
            db.commit()
    return {"status": "cancelled", "booking_id": booking_id}


@router.post("/agent/run")
def run_agent(db: Session = Depends(get_db)):
    results = run_booking_agent(db)
    return {"groups_processed": len(results), "results": results}


@router.post("/groups/{group_id}/retry")
def retry_group(group_id: str, db: Session = Depends(get_db)):
    result = resolve_exception(db, group_id, action="retry")
    return result


@router.post("/direct")
def direct_booking(
    payload: DirectBookingCreate,
    db: Session = Depends(get_db),
):
    result = create_direct_booking(
        db=db,
        organizer_id=payload.organizer_id,
        friend_id=payload.friend_id,
        venue_id=payload.venue_id,
        reservation_date=payload.reservation_date,
        time_slot=payload.time_slot,
    )
    status = result.get("status")
    if status == "missing_user":
        raise HTTPException(404, "organizer_or_friend_not_found")
    if status == "missing_venue":
        raise HTTPException(404, "venue_not_found")
    if status == "invalid_pair":
        raise HTTPException(400, "invalid_pair")

    booking = result.get("booking")
    return {
        "status": status,
        "booking": BookingResponse.model_validate(booking).model_dump(),
    }


@router.post("/solo")
def solo_booking(
    payload: SoloBookingCreate,
    db: Session = Depends(get_db),
):
    result = create_solo_booking(
        db=db,
        user_id=payload.user_id,
        venue_id=payload.venue_id,
        reservation_date=payload.reservation_date,
        time_slot=payload.time_slot,
    )
    status = result.get("status")
    if status == "missing_user":
        raise HTTPException(404, "user_not_found")
    if status == "missing_venue":
        raise HTTPException(404, "venue_not_found")

    booking = result.get("booking")
    return {
        "status": status,
        "booking": BookingResponse.model_validate(booking).model_dump(),
    }


@router.post("/{booking_id}/manual-review")
def manual_review_booking(
    booking_id: str,
    payload: ManualReviewAction,
    db: Session = Depends(get_db),
):
    result = handle_manual_review(
        db=db,
        booking_id=booking_id,
        actor_user_id=payload.user_id,
        action=payload.action,
        confirmation_code=payload.confirmation_code,
    )
    status = result.get("status")
    if status == "missing_booking":
        raise HTTPException(404, "booking_not_found")
    if status == "not_organizer":
        raise HTTPException(403, "only_organizer_can_review")
    if status in {"invalid_state", "invalid_action"}:
        raise HTTPException(400, result)
    return result
