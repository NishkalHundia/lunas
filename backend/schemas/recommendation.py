from typing import Optional

from pydantic import BaseModel

from backend.schemas.user import UserBrief
from backend.schemas.venue import VenueResponse


class ExplainConcept(BaseModel):
    concept_id: str
    label: str
    strength: float
    venue_fit: float
    contribution: float


class PersonWithRelationship(BaseModel):
    user: UserBrief
    relationship: str
    compatibility: float
    shared_concepts: list[str]


class VenueRecommendation(BaseModel):
    venue: VenueResponse
    score: float
    score_breakdown: dict
    people: list[PersonWithRelationship]
    total_interested: int
    friend_count: int
    mutual_count: int
    trending: bool
    explanation: dict


class PeopleRecommendation(BaseModel):
    user: UserBrief
    relationship: str
    compatibility: float
    shared_concepts: list[str]
    narrative: str


class TrendingVenue(BaseModel):
    venue: VenueResponse
    trend_score: float
    momentum: float
    active_interests: int
    predicted_attendance: int


class EngagementSummary(BaseModel):
    venue_id: str
    total_views: int
    total_saves: int
    total_checkins: int
    total_shares: int
    avg_view_duration: Optional[float]
    trend_direction: str


class UserActivitySummary(BaseModel):
    user_id: str
    total_interactions: int
    top_categories: list[dict]
    most_active_hours: list[int]
    favorite_venues: list[dict]
    friend_interaction_count: int
