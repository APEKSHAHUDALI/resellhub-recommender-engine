"""
Celery worker for scheduled, off-request-path work: nightly model retraining.
Run with:
  celery -A app.services.tasks worker --loglevel=info
  celery -A app.services.tasks beat --loglevel=info
"""
from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

celery_app = Celery("resellhub", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.beat_schedule = {
    "nightly-model-retrain": {
        "task": "app.services.tasks.retrain_models",
        "schedule": crontab(hour=3, minute=0),  # 3am daily - low traffic window
    },
}
celery_app.conf.timezone = "UTC"


@celery_app.task(name="app.services.tasks.retrain_models")
def retrain_models():
    from app.recommender.train import train_all
    from app.core.cache import invalidate

    summary = train_all()
    invalidate("reco:")
    return summary
