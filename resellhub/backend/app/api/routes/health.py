import os
from fastapi import APIRouter
from sqlalchemy import text

from app.database import engine
from app.config import get_settings

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
def health():
    """Liveness/readiness probe: checks DB connectivity and whether a trained model exists."""
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    model_path = os.path.join(settings.model_artifact_dir, "artifacts.pkl")
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "up" if db_ok else "down",
        "model_trained": os.path.exists(model_path),
        "environment": settings.environment,
    }
