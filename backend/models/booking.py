import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, DateTime, Date, Time, JSON, Boolean, Index

from backend.database import Base


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    invitation_group_id = Column(String, nullable=True, index=True)
    venue_id = Column(String, nullable=False, index=True)

    organizer_id = Column(String, nullable=False, index=True)
    attendee_ids = Column(JSON, default=list)
    attendee_status_map = Column(JSON, default=dict)

    reservation_date = Column(Date, nullable=False)
    reservation_time = Column(Time, nullable=False)
    party_size = Column(Integer, nullable=False)

    status = Column(String, default="pending")  # pending, confirmed, failed, cancelled, needs_manual_review
    confirmation_code = Column(String, nullable=True)
    provider = Column(String, default="mock")
    provider_payload = Column(JSON, default=dict)

    booking_method = Column(String, default="auto_agent")
    manual_review_required = Column(Boolean, default=False)
    exception_reason = Column(String, nullable=True)
    agent_log = Column(JSON, default=list)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_bookings_status_date", "status", "reservation_date"),
    )
