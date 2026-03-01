"""Re-train embeddings from existing seeded interaction data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.database import SessionLocal
from backend.services.embedding_engine import engine


def seed_embeddings():
    db = SessionLocal()
    try:
        engine.train_from_synthetic_data(db)
        print("Embeddings retrained and profiles refreshed.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_embeddings()
