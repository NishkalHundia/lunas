"""Embedding-focused endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.user import User
from backend.models.venue import Venue
from backend.services.embedding_engine import CONCEPT_CATALOG, engine
from backend.services.recommendation_engine import get_people_recommendations, get_recommendations


router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


def _get_user_or_404(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "user_not_found")
    return user


@router.post("/train")
def train_embeddings(db: Session = Depends(get_db)):
    engine.train_from_synthetic_data(db)
    return {"status": "trained", "dim": engine.dim}


@router.get("/taste-profile/{user_id}")
def get_taste_profile(user_id: str, db: Session = Depends(get_db)):
    engine.ensure_initialized(db)
    user = _get_user_or_404(db, user_id)
    vector = user.embedding_vector or []
    profile = engine.concept_profile(engine._parse_vector(vector))
    meta = user.embedding_meta or {}
    return {
        "user_id": user.id,
        "display_name": user.display_name,
        "archetype": user.archetype,
        "embedding_dim": engine.dim,
        "concepts": profile,
        "top_concepts": profile[:6],
        "narrative": meta.get("narrative", "Interact more to build your profile."),
        "latest_drift": meta.get("latest_drift", 0.0),
        "concept_changes": meta.get("concept_changes", []),
        "updated_at": user.embedding_updated_at.isoformat() if user.embedding_updated_at else None,
    }


@router.get("/venue-profile/{venue_id}")
def venue_profile(venue_id: str, db: Session = Depends(get_db)):
    engine.ensure_initialized(db)
    venue = db.query(Venue).filter(Venue.id == venue_id).first()
    if not venue:
        raise HTTPException(404, "venue_not_found")
    profile = engine.concept_profile(engine._parse_vector(venue.embedding_vector))
    return {
        "venue_id": venue.id,
        "venue_name": venue.name,
        "category": venue.category,
        "concepts": profile,
        "top_concepts": profile[:5],
    }


@router.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=60),
    time_slot: str | None = None,
    db: Session = Depends(get_db),
):
    rows = get_recommendations(db, user_id, time_slot=time_slot, limit=limit)
    response = []
    for row in rows:
        response.append(
            {
                "venue": {
                    "id": row["venue"].id,
                    "name": row["venue"].name,
                    "description": row["venue"].description,
                    "category": row["venue"].category,
                    "subcategory": row["venue"].subcategory,
                    "city": row["venue"].city,
                    "neighborhood": row["venue"].neighborhood,
                    "price_level": row["venue"].price_level,
                    "avg_rating": row["venue"].avg_rating,
                    "image_urls": row["venue"].image_urls,
                    "cuisine_tags": row["venue"].cuisine_tags,
                    "vibe_tags": row["venue"].vibe_tags,
                },
                "embedding_similarity": row["score_breakdown"]["embedding"],
                "score": row["score"],
                "score_breakdown": row["score_breakdown"],
                "total_interested": row["total_interested"],
                "friend_count": row["friend_count"],
                "mutual_count": row["mutual_count"],
                "trending": row["trending"],
                "explanation": row["explanation"],
            }
        )
    return response


@router.get("/people/{user_id}")
def people(
    user_id: str,
    venue_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=60),
    db: Session = Depends(get_db),
):
    rows = get_people_recommendations(db, user_id, venue_id=venue_id, limit=limit)
    response = []
    for row in rows:
        user = row["user"]
        response.append(
            {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "display_name": user.display_name,
                    "avatar_url": user.avatar_url,
                    "neighborhood": user.neighborhood,
                    "age": user.age,
                    "archetype": user.archetype,
                },
                "similarity": row["compatibility"],
                "relationship": row["relationship"],
                "shared_concepts": row["shared_concepts"],
                "narrative": row["narrative"],
            }
        )
    return response


@router.post("/interact")
def interact(
    user_id: str,
    venue_id: str,
    interaction_type: str,
    view_duration: float | None = None,
    db: Session = Depends(get_db),
):
    engine.ensure_initialized(db)
    result = engine.online_update(db, user_id, venue_id, interaction_type, view_duration)
    if not result.get("updated"):
        raise HTTPException(404, result.get("reason", "update_failed"))
    return {
        "user_id": user_id,
        "venue_id": venue_id,
        "interaction_type": interaction_type,
        "embedding_updated": True,
        "drift": {
            "drift": result["drift"],
            "concept_changes": result["concept_changes"],
        },
        "new_top_concepts": result["top_concepts"],
    }


@router.get("/concepts")
def concepts():
    return [{"id": concept["id"], "label": concept["label"]} for concept in CONCEPT_CATALOG]
