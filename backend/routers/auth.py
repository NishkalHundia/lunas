"""Lightweight OAuth-ready auth shim.

For take-home scope this provides provider-based sign-in without handling
full OAuth redirects/secrets on the backend. The frontend can pass verified
identity payloads from an OAuth provider later with the same contract.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.user import User
from backend.services.embedding_engine import engine


router = APIRouter(prefix="/api/auth", tags=["auth"])


class OAuthSignIn(BaseModel):
    provider: str = "google"
    provider_subject: str
    email: str
    display_name: str
    avatar_url: Optional[str] = None
    city: str = "Austin"
    neighborhood: Optional[str] = None


@router.post("/oauth-signin")
def oauth_signin(payload: OAuthSignIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        User.oauth_provider == payload.provider,
        User.oauth_subject == payload.provider_subject,
    ).first()
    if not user:
        user = db.query(User).filter(User.email == payload.email).first()

    if user:
        user.oauth_provider = payload.provider
        user.oauth_subject = payload.provider_subject
        if payload.avatar_url:
            user.avatar_url = payload.avatar_url
        db.commit()
        db.refresh(user)
        return {"status": "signed_in", "user_id": user.id}

    username = payload.email.split("@")[0].lower().replace(".", "_")
    base = username[:30]
    username = base
    suffix = 1
    while db.query(User).filter(User.username == username).first():
        suffix += 1
        username = f"{base[:24]}_{suffix}"

    user = User(
        id=str(uuid.uuid4()),
        username=username,
        display_name=payload.display_name,
        email=payload.email,
        avatar_url=payload.avatar_url,
        home_latitude=30.2672,
        home_longitude=-97.7431,
        city=payload.city,
        neighborhood=payload.neighborhood,
        cuisine_preferences=[],
        vibe_preferences=[],
        preferred_time_slots=["evening"],
        price_preference=2,
        age=26,
        archetype="newcomer",
        oauth_provider=payload.provider,
        oauth_subject=payload.provider_subject,
    )
    user.embedding_vector = engine.user_prior(user).tolist()
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"status": "created", "user_id": user.id}
