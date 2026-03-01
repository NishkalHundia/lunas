import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, DateTime, Index, JSON

from backend.database import Base


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    venue_id = Column(String, nullable=False, index=True)

    interaction_type = Column(String, nullable=False, index=True)
    weight = Column(Float, default=1.0)
    view_duration_seconds = Column(Float, nullable=True)
    scroll_depth = Column(Float, nullable=True)
    source = Column(String, default="feed")
    session_id = Column(String, nullable=True)
    context = Column(JSON, default=dict)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_interactions_user_time", "user_id", "created_at"),
        Index("ix_interactions_venue_time", "venue_id", "created_at"),
        Index("ix_interactions_user_venue_type", "user_id", "venue_id", "interaction_type"),
    )
