import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Boolean

from backend.database import Base


class Venue(Base):
    __tablename__ = "venues"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(String, nullable=False)
    city = Column(String, nullable=False, default="Austin")
    neighborhood = Column(String, nullable=True)

    category = Column(String, nullable=False, index=True)
    subcategory = Column(String, nullable=True)
    cuisine_tags = Column(JSON, default=list)
    vibe_tags = Column(JSON, default=list)
    price_level = Column(Integer, nullable=False)
    capacity = Column(Integer, nullable=False)
    avg_rating = Column(Float, default=4.1)

    hours = Column(JSON, default=dict)
    accepts_reservations = Column(Boolean, default=True)
    reservation_url = Column(String, nullable=True)
    booking_provider = Column(String, default="mock")
    booking_reliability = Column(Float, default=0.92)
    min_party_size = Column(Integer, default=1)
    max_party_size = Column(Integer, default=12)

    image_urls = Column(JSON, default=list)
    embedding_vector = Column(JSON, default=list)
    embedding_meta = Column(JSON, default=dict)
    popularity_prior = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
