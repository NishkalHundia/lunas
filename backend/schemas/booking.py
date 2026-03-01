from datetime import date, time, datetime
from typing import Optional

from pydantic import BaseModel, Field


class InterestCreate(BaseModel):
    user_id: str
    venue_id: str
    preferred_date: Optional[date] = None
    preferred_time_slot: Optional[str] = "19:00"
    flexible_date: bool = True
    flexible_group_size: bool = True
    min_group_size: int = 2
    max_group_size: int = 8
    intent_strength: float = 0.65
    source: str = "manual"


class InterestResponse(BaseModel):
    id: str
    user_id: str
    venue_id: str
    preferred_date: Optional[date]
    preferred_time_slot: Optional[str]
    flexible_date: bool
    flexible_group_size: bool
    min_group_size: int
    max_group_size: int
    intent_strength: float
    source: str
    status: str
    created_at: datetime
    expires_at: datetime

    model_config = {"from_attributes": True}


class BookingResponse(BaseModel):
    id: str
    invitation_group_id: Optional[str]
    venue_id: str
    organizer_id: str
    attendee_ids: list[str]
    attendee_status_map: dict
    reservation_date: date
    reservation_time: time
    party_size: int
    status: str
    confirmation_code: Optional[str]
    provider: str
    provider_payload: dict
    booking_method: str
    manual_review_required: bool
    exception_reason: Optional[str]
    agent_log: list[dict]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TrackingEvent(BaseModel):
    user_id: str
    venue_id: str
    interaction_type: str
    source: str = "feed"
    view_duration_seconds: Optional[float] = None
    scroll_depth: Optional[float] = None
    session_id: Optional[str] = None
    context: dict = Field(default_factory=dict)


class InvitationSend(BaseModel):
    from_user_id: str
    to_user_ids: list[str]
    venue_id: str
    proposed_date: Optional[date] = None
    proposed_time_slot: Optional[str] = "19:00"
    threshold_count: int = 2
    message: Optional[str] = None


class InvitationRespond(BaseModel):
    status: str  # accepted, declined


class InvitationResponse(BaseModel):
    id: str
    group_id: str
    from_user_id: str
    to_user_id: str
    venue_id: str
    message: Optional[str]
    status: str
    created_at: datetime
    responded_at: Optional[datetime]

    model_config = {"from_attributes": True}


class BookingExceptionResolve(BaseModel):
    action: str = "retry"  # retry, cancel


class ManualReviewAction(BaseModel):
    user_id: str
    action: str  # confirm, retry_provider, cancel
    confirmation_code: Optional[str] = None


class DirectBookingCreate(BaseModel):
    organizer_id: str
    friend_id: str
    venue_id: str
    reservation_date: Optional[date] = None
    time_slot: Optional[str] = "19:00"


class SoloBookingCreate(BaseModel):
    user_id: str
    venue_id: str
    reservation_date: Optional[date] = None
    time_slot: Optional[str] = "19:00"
