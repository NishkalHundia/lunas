import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Date, Index, Float

from backend.database import Base


def _default_expiry():
    return datetime.now(timezone.utc) + timedelta(days=10)


class Interest(Base):
    __tablename__ = "interests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    venue_id = Column(String, nullable=False, index=True)

    preferred_date = Column(Date, nullable=True)
    preferred_time_slot = Column(String, nullable=True)
    flexible_date = Column(Boolean, default=True)
    flexible_group_size = Column(Boolean, default=True)
    min_group_size = Column(Integer, default=2)
    max_group_size = Column(Integer, default=8)
    intent_strength = Column(Float, default=0.6)
    source = Column(String, default="manual")

    status = Column(String, default="active")

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, default=_default_expiry)

    __table_args__ = (
        Index("ix_interests_venue_status", "venue_id", "status"),
        Index("ix_interests_user_status", "user_id", "status"),
    )
