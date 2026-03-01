import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Index

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    bio = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)

    oauth_provider = Column(String, nullable=True)
    oauth_subject = Column(String, nullable=True)

    home_latitude = Column(Float, nullable=False)
    home_longitude = Column(Float, nullable=False)
    city = Column(String, nullable=False, default="Austin")
    neighborhood = Column(String, nullable=True)
    max_travel_distance_km = Column(Float, default=14.0)

    cuisine_preferences = Column(JSON, default=list)
    vibe_preferences = Column(JSON, default=list)
    preferred_time_slots = Column(JSON, default=list)
    price_preference = Column(Integer, default=2)
    age = Column(Integer, nullable=False)
    archetype = Column(String, nullable=True)

    embedding_vector = Column(JSON, default=list)
    embedding_meta = Column(JSON, default=dict)
    embedding_updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_active = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    total_checkins = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_users_oauth", "oauth_provider", "oauth_subject"),
    )
