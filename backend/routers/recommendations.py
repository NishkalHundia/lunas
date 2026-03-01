from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.venue import VenueResponse
from backend.services.recommendation_engine import (
    get_pair_recommendations,
    get_people_recommendations,
    get_recommendations,
    search_recommendations,
)


router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.get("/{user_id}")
def recommend_venues(
    user_id: str,
    latitude: float | None = None,
    longitude: float | None = None,
    target_date: date | None = None,
    time_slot: str | None = None,
    limit: int = Query(default=20, ge=1, le=60),
    db: Session = Depends(get_db),
):
    rows = get_recommendations(
        db=db,
        user_id=user_id,
        latitude=latitude,
        longitude=longitude,
        target_date=target_date,
        time_slot=time_slot,
        limit=limit,
    )
    response = []
    for row in rows:
        people = []
        for person in row["people"]:
            user = person["user"]
            people.append(
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
                    "relationship": person["relationship"],
                    "compatibility": person["compatibility"],
                    "shared_concepts": person.get("shared_concepts", []),
                }
            )
        response.append(
            {
                "venue": VenueResponse.model_validate(row["venue"]).model_dump(),
                "score": row["score"],
                "score_breakdown": row["score_breakdown"],
                "people": people,
                "total_interested": row["total_interested"],
                "friend_count": row["friend_count"],
                "mutual_count": row["mutual_count"],
                "trending": row["trending"],
                "explanation": row["explanation"],
            }
        )
    return response


@router.get("/{user_id}/people")
def recommend_people(
    user_id: str,
    venue_id: str | None = Query(default=None),
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
                "relationship": row["relationship"],
                "compatibility": row["compatibility"],
                "shared_concepts": row["shared_concepts"],
                "narrative": row["narrative"],
            }
        )
    return response


@router.get("/{user_id}/search")
def search_venues(
    user_id: str,
    query: str = Query(..., min_length=1),
    time_slot: str | None = None,
    limit: int = Query(default=24, ge=1, le=80),
    db: Session = Depends(get_db),
):
    payload = search_recommendations(
        db=db,
        user_id=user_id,
        query=query,
        time_slot=time_slot,
        limit=limit,
    )
    rows = payload.get("results", [])
    response = []
    for row in rows:
        people = []
        for person in row.get("people", []):
            user = person["user"]
            people.append(
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
                    "relationship": person["relationship"],
                    "compatibility": person["compatibility"],
                    "shared_concepts": person.get("shared_concepts", []),
                }
            )
        response.append(
            {
                "venue": VenueResponse.model_validate(row["venue"]).model_dump(),
                "score": row["score"],
                "score_breakdown": row["score_breakdown"],
                "people": people,
                "total_interested": row["total_interested"],
                "friend_count": row["friend_count"],
                "mutual_count": row["mutual_count"],
                "trending": row["trending"],
                "explanation": row["explanation"],
                "match_type": row.get("match_type", "exact"),
                "query_similarity": row.get("query_similarity"),
            }
        )
    return {
        "query": payload.get("query", query),
        "exact_match_found": payload.get("exact_match_found", False),
        "results": response,
    }


@router.get("/{user_id}/pair/{other_user_id}")
def recommend_pair_venues(
    user_id: str,
    other_user_id: str,
    time_slot: str | None = None,
    limit: int = Query(default=12, ge=1, le=40),
    db: Session = Depends(get_db),
):
    rows = get_pair_recommendations(
        db=db,
        user_id=user_id,
        other_user_id=other_user_id,
        time_slot=time_slot,
        limit=limit,
    )
    response = []
    for row in rows:
        response.append(
            {
                "venue": VenueResponse.model_validate(row["venue"]).model_dump(),
                "score": row["score"],
                "relationship": row["relationship"],
                "user_match": row["user_match"],
                "other_match": row["other_match"],
                "distance_km": row["distance_km"],
                "score_breakdown": row["score_breakdown"],
                "shared_concepts": row["shared_concepts"],
                "narrative": row["narrative"],
                "explanation": row["explanation"],
            }
        )
    return response
