from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    display_name: str = Field(min_length=2, max_length=80)
    email: str
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    home_latitude: float
    home_longitude: float
    city: str = "Austin"
    neighborhood: Optional[str] = None
    max_travel_distance_km: float = 14.0
    cuisine_preferences: list[str] = Field(default_factory=list)
    vibe_preferences: list[str] = Field(default_factory=list)
    preferred_time_slots: list[str] = Field(default_factory=list)
    price_preference: int = 2
    age: int = 25


class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    max_travel_distance_km: Optional[float] = None
    cuisine_preferences: Optional[list[str]] = None
    vibe_preferences: Optional[list[str]] = None
    preferred_time_slots: Optional[list[str]] = None
    price_preference: Optional[int] = None


class UserResponse(BaseModel):
    id: str
    username: str
    display_name: str
    email: str
    bio: Optional[str]
    avatar_url: Optional[str]
    city: str
    neighborhood: Optional[str]
    home_latitude: float
    home_longitude: float
    max_travel_distance_km: float
    cuisine_preferences: list[str]
    vibe_preferences: list[str]
    preferred_time_slots: list[str]
    price_preference: int
    age: int
    archetype: Optional[str]
    embedding_updated_at: datetime
    created_at: datetime
    last_active: datetime
    total_checkins: int

    model_config = {"from_attributes": True}


class UserBrief(BaseModel):
    id: str
    username: str
    display_name: str
    avatar_url: Optional[str]
    neighborhood: Optional[str]
    age: int
    archetype: Optional[str]

    model_config = {"from_attributes": True}
