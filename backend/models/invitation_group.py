import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Date, Integer, Index

from backend.database import Base


class InvitationGroup(Base):
    __tablename__ = "invitation_groups"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organizer_id = Column(String, nullable=False, index=True)
    venue_id = Column(String, nullable=False, index=True)
    proposed_date = Column(Date, nullable=True)
    proposed_time_slot = Column(String, nullable=True)
    threshold_count = Column(Integer, default=2)

    status = Column(String, default="collecting")  # collecting, auto_booked, exception, cancelled
    booking_id = Column(String, nullable=True, index=True)
    exception_reason = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_inv_group_status_created", "status", "created_at"),
    )
