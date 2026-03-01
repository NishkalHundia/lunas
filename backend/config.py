import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'luna.db'}")

_default_origins = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", _default_origins).split(",") if origin.strip()]

SEED_RANDOM_STATE = int(os.getenv("SEED_RANDOM_STATE", "42"))
NUM_SEED_USERS = int(os.getenv("NUM_SEED_USERS", "700"))
NUM_SEED_VENUES = int(os.getenv("NUM_SEED_VENUES", "220"))
NUM_SEED_INTERACTIONS = int(os.getenv("NUM_SEED_INTERACTIONS", "90000"))

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "48"))
EMBEDDING_EPOCHS = int(os.getenv("EMBEDDING_EPOCHS", "6"))
EMBEDDING_LR = float(os.getenv("EMBEDDING_LR", "0.045"))
EMBEDDING_REG = float(os.getenv("EMBEDDING_REG", "0.0008"))
ONLINE_UPDATE_ALPHA = float(os.getenv("ONLINE_UPDATE_ALPHA", "0.14"))

AUTO_BOOKING_MIN_ACCEPTED = int(os.getenv("AUTO_BOOKING_MIN_ACCEPTED", "2"))
