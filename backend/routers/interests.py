from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.interest import Interest
from backend.models.user import User
from backend.models.venue import Venue
from backend.schemas.booking import InterestCreate, InterestResponse
from backend.services.booking_agent import run_booking_agent
from backend.services.embedding_engine import engine


router = APIRouter(prefix="/api/interests", tags=["interests"])


@router.post("", response_model=InterestResponse)
def create_interest(payload: InterestCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    venue = db.query(Venue).filter(Venue.id == payload.venue_id).first()
    if not user or not venue:
        raise HTTPException(404, "user_or_venue_not_found")

    interest = Interest(**payload.model_dump())
    db.add(interest)
    db.commit()
    db.refresh(interest)

    engine.ensure_initialized(db)
    engine.online_update(db, payload.user_id, payload.venue_id, "intent")
    run_booking_agent(db)
    return interest


@router.delete("/{interest_id}")
def retract_interest(interest_id: str, db: Session = Depends(get_db)):
    interest = db.query(Interest).filter(Interest.id == interest_id).first()
    if not interest:
        raise HTTPException(404, "interest_not_found")
    interest.status = "cancelled"
    db.commit()
    return {"status": "cancelled", "interest_id": interest_id}


@router.get("/user/{user_id}", response_model=list[InterestResponse])
def get_user_interests(user_id: str, db: Session = Depends(get_db)):
    return db.query(Interest).filter(
        Interest.user_id == user_id,
        Interest.status == "active",
    ).order_by(Interest.created_at.desc()).all()
