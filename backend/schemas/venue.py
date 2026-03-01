from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class VenueResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    latitude: float
    longitude: float
    address: str
    city: str
    neighborhood: Optional[str]
    category: str
    subcategory: Optional[str]
    cuisine_tags: list[str]
    vibe_tags: list[str]
    price_level: int
    capacity: int
    avg_rating: float
    hours: dict
    accepts_reservations: bool
    image_urls: list[str]
    popularity_prior: float
    created_at: datetime

    model_config = {"from_attributes": True}
