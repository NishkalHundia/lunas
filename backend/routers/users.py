import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.friendship import Friendship
from backend.models.user import User
from backend.schemas.user import UserBrief, UserCreate, UserResponse, UserUpdate
from backend.services.embedding_engine import engine


router = APIRouter(prefix="/api/users", tags=["users"])


@router.post("", response_model=UserResponse)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter((User.username == payload.username) | (User.email == payload.email)).first()
    if existing:
        raise HTTPException(409, "username_or_email_already_exists")
    user = User(**payload.model_dump())
    user.embedding_vector = engine.user_prior(user).tolist()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "user_not_found")
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: str, payload: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "user_not_found")
    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(user, key, value)
    prior = engine.user_prior(user)
    current = engine._parse_vector(user.embedding_vector)
    if current.size == engine.dim:
        merged = 0.75 * current + 0.25 * prior
    else:
        merged = prior
    norm = np.linalg.norm(merged)
    if norm > 0:
        merged = merged / norm
    user.embedding_vector = [float(x) for x in merged.tolist()]
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}/friends", response_model=list[UserBrief])
def get_friends(user_id: str, db: Session = Depends(get_db)):
    rows = db.query(Friendship).filter(
        ((Friendship.user_id == user_id) | (Friendship.friend_id == user_id)),
        Friendship.status == "accepted",
    ).all()
    friend_ids = [row.friend_id if row.user_id == user_id else row.user_id for row in rows]
    if not friend_ids:
        return []
    return db.query(User).filter(User.id.in_(friend_ids)).limit(150).all()


@router.post("/{user_id}/friends/{friend_id}")
def add_friend(user_id: str, friend_id: str, db: Session = Depends(get_db)):
    if user_id == friend_id:
        raise HTTPException(400, "cannot_friend_self")

    user = db.query(User).filter(User.id == user_id).first()
    friend = db.query(User).filter(User.id == friend_id).first()
    if not user or not friend:
        raise HTTPException(404, "user_or_friend_not_found")

    a, b = sorted([user_id, friend_id])
    existing = db.query(Friendship).filter(
        Friendship.user_id == a,
        Friendship.friend_id == b,
    ).first()
    if existing:
        if existing.status != "accepted":
            existing.status = "accepted"
            existing.relationship_type = "friend"
            db.commit()
        return {"status": "already_friends", "user_id": user_id, "friend_id": friend_id}

    row = Friendship(
        user_id=a,
        friend_id=b,
        status="accepted",
        relationship_type="friend",
        interaction_score=0.2,
    )
    db.add(row)
    db.commit()
    return {"status": "friend_added", "user_id": user_id, "friend_id": friend_id}


@router.delete("/{user_id}/friends/{friend_id}")
def remove_friend(user_id: str, friend_id: str, db: Session = Depends(get_db)):
    if user_id == friend_id:
        raise HTTPException(400, "cannot_unfriend_self")

    a, b = sorted([user_id, friend_id])
    row = db.query(Friendship).filter(
        Friendship.user_id == a,
        Friendship.friend_id == b,
    ).first()
    if not row:
        return {"status": "not_friends", "user_id": user_id, "friend_id": friend_id}
    db.delete(row)
    db.commit()
    return {"status": "friend_removed", "user_id": user_id, "friend_id": friend_id}


@router.get("", response_model=list[UserBrief])
def list_users(
    city: str | None = Query(default=None),
    search: str | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=2000),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if city:
        query = query.filter(User.city == city)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(
            (User.display_name.ilike(pattern))
            | (User.username.ilike(pattern))
            | (User.neighborhood.ilike(pattern))
        )
    return query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
