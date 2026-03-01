from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.interest import Interest
from backend.models.user import User
from backend.models.venue import Venue
from backend.schemas.venue import VenueResponse
from backend.services.social_scorer import categorize_relationship, compatibility_score


router = APIRouter(prefix="/api/venues", tags=["venues"])


@router.get("", response_model=list[VenueResponse])
def list_venues(
    city: str = "Austin",
    category: str | None = None,
    search: str | None = None,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    query = db.query(Venue).filter(Venue.city == city)
    if category:
        query = query.filter(Venue.category == category)
    if search:
        pattern = f"%{search}%"
        query = query.filter(
            (Venue.name.ilike(pattern))
            | (Venue.category.ilike(pattern))
            | (Venue.neighborhood.ilike(pattern))
            | (Venue.subcategory.ilike(pattern))
        )
    return query.order_by(Venue.popularity_prior.desc(), Venue.avg_rating.desc()).offset(skip).limit(limit).all()


@router.get("/{venue_id}", response_model=VenueResponse)
def get_venue(venue_id: str, db: Session = Depends(get_db)):
    venue = db.query(Venue).filter(Venue.id == venue_id).first()
    if not venue:
        raise HTTPException(404, "venue_not_found")
    return venue


@router.get("/{venue_id}/interested")
def interested_users(
    venue_id: str,
    user_id: str = Query(..., description="Requesting user id"),
    db: Session = Depends(get_db),
):
    requester = db.query(User).filter(User.id == user_id).first()
    if not requester:
        raise HTTPException(404, "requesting_user_not_found")

    interests = db.query(Interest).filter(
        Interest.venue_id == venue_id,
        Interest.status == "active",
    ).all()

    grouped = {"friends": [], "mutuals": [], "new_people": []}
    for interest in interests:
        if interest.user_id == user_id:
            continue
        other = db.query(User).filter(User.id == interest.user_id).first()
        if not other:
            continue
        rel = categorize_relationship(db, user_id, other.id)
        entry = {
            "user_id": other.id,
            "username": other.username,
            "display_name": other.display_name,
            "avatar_url": other.avatar_url,
            "neighborhood": other.neighborhood,
            "compatibility": round(compatibility_score(db, requester, other), 4),
            "preferred_time_slot": interest.preferred_time_slot,
            "preferred_date": str(interest.preferred_date) if interest.preferred_date else None,
        }
        if rel == "friend":
            grouped["friends"].append(entry)
        elif rel == "mutual":
            grouped["mutuals"].append(entry)
        else:
            grouped["new_people"].append(entry)

    for value in grouped.values():
        value.sort(key=lambda item: -item["compatibility"])

    return grouped
