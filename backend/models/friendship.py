import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, DateTime, Index

from backend.database import Base


class Friendship(Base):
    __tablename__ = "friendships"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    friend_id = Column(String, nullable=False, index=True)

    status = Column(String, default="accepted")
    relationship_type = Column(String, default="friend")
    interaction_score = Column(Float, default=0.0)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_friendships_pair", "user_id", "friend_id", unique=True),
    )
