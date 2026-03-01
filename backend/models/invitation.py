import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Index

from backend.database import Base


class Invitation(Base):
    __tablename__ = "invitations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    group_id = Column(String, nullable=False, index=True)
    from_user_id = Column(String, nullable=False, index=True)
    to_user_id = Column(String, nullable=False, index=True)
    venue_id = Column(String, nullable=False, index=True)
    message = Column(String, nullable=True)

    status = Column(String, default="pending")  # pending, accepted, declined, expired

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    responded_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_invites_group_target", "group_id", "to_user_id"),
    )
