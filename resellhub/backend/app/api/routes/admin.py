from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.deps import require_role
from app.models import UserRole
from app.schemas import TrainingSummary
from app.core.cache import invalidate

router = APIRouter(prefix="/admin", tags=["admin"])


def _train_and_invalidate():
    from app.recommender.train import train_all
    train_all()
    invalidate("reco:")


@router.post("/retrain", status_code=202)
def trigger_retrain(background_tasks: BackgroundTasks, _=Depends(require_role(UserRole.ADMIN))):
    """
    Kicks off retraining as a background task so the request returns
    immediately. In the docker-compose stack this is instead handled by the
    scheduled Celery beat task in app/services/tasks.py - this endpoint is
    for on-demand retrains (e.g. after a big data backfill).
    """
    background_tasks.add_task(_train_and_invalidate)
    return {"status": "retrain_started"}


@router.get("/training-summary", response_model=TrainingSummary | None)
def training_summary(db: Session = Depends(get_db), _=Depends(require_role(UserRole.ADMIN))):
    from app.services.recommendation_service import _load_artifacts
    artifacts = _load_artifacts()
    if artifacts is None:
        return None
    return TrainingSummary(
        products=len(artifacts["content_model"].product_ids),
        customer_interactions=len(artifacts["customer_cf"].user_index) if artifacts["customer_cf"].is_fitted else 0,
        reseller_events=len(artifacts["reseller_cf"].user_index) if artifacts["reseller_cf"].is_fitted else 0,
        customer_cf_coverage=artifacts["customer_cf"].coverage(),
        reseller_cf_coverage=artifacts["reseller_cf"].coverage(),
    )
