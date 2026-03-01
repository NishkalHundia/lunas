from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.booking import TrackingEvent
from backend.schemas.venue import VenueResponse
from backend.services.engagement_tracker import (
    get_user_activity_summary,
    get_venue_engagement_summary,
    track_interaction,
)
from backend.services.trend_predictor import predict_trending


router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/trending")
def trending(
    city: str = "Austin",
    target_date: date | None = None,
    time_slot: str = "evening",
    limit: int = Query(default=10, ge=1, le=40),
    db: Session = Depends(get_db),
):
    chosen_date = target_date or datetime.now(timezone.utc).date()
    rows = predict_trending(db, city=city, target_date=chosen_date, time_slot=time_slot, top_n=limit)
    response = []
    for row in rows:
        response.append(
            {
                "venue": VenueResponse.model_validate(row["venue"]).model_dump(),
                "trend_score": row["trend_score"],
                "momentum": row["momentum"],
                "active_interests": row["active_interests"],
                "predicted_attendance": row["predicted_attendance"],
            }
        )
    return response


@router.post("/track")
def track(payload: TrackingEvent, db: Session = Depends(get_db)):
    row, update = track_interaction(
        db=db,
        user_id=payload.user_id,
        venue_id=payload.venue_id,
        interaction_type=payload.interaction_type,
        view_duration_seconds=payload.view_duration_seconds,
        scroll_depth=payload.scroll_depth,
        source=payload.source,
        session_id=payload.session_id,
        context=payload.context,
    )
    return {"id": row.id, "status": "tracked", "embedding_update": update}


@router.get("/venue/{venue_id}/engagement")
def venue_engagement(venue_id: str, db: Session = Depends(get_db)):
    return get_venue_engagement_summary(db, venue_id)


@router.get("/user/{user_id}/activity")
def user_activity(user_id: str, db: Session = Depends(get_db)):
    return get_user_activity_summary(db, user_id)
