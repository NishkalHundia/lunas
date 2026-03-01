"""Luna Social API v2."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import CORS_ORIGINS
from backend.database import init_db
from backend.routers import analytics, auth, bookings, embeddings, interests, invitations, recommendations, users, venues


app = FastAPI(
    title="Luna Social API",
    description="Social meetup intelligence platform with trainable embeddings and auto-booking agents.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(venues.router)
app.include_router(recommendations.router)
app.include_router(embeddings.router)
app.include_router(interests.router)
app.include_router(invitations.router)
app.include_router(bookings.router)
app.include_router(analytics.router)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/")
def root():
    return {
        "name": "Luna Social API",
        "version": "2.0.0",
        "docs": "/docs",
        "frontend": "/app",
        "highlights": [
            "Trained user+venue embeddings",
            "Real-time online embedding updates",
            "Explainable taste decomposition",
            "Invitation orchestration with auto booking",
        ],
    }


@app.on_event("startup")
def on_startup():
    init_db()
