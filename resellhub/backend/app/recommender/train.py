"""
Orchestrates a full model training run:
  1. Pull interactions/products/stock-events from the DB
  2. Fit customer-side CF, reseller-side CF, content-based, and popularity
  3. Persist everything to disk as pickles under settings.model_artifact_dir

Run manually with `python -m app.recommender.train`, or on a schedule via the
Celery beat task defined in app/services/tasks.py. Keeping training as an
offline batch job (rather than fitting on every request) is what makes the
online recommendation endpoint fast: the API only ever does matrix lookups
and vector similarity against pre-fitted models, never trains live.
"""
from __future__ import annotations
import os
import pickle
import logging
from collections import defaultdict

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal
from app.models import Product, CustomerInteraction, ResellerStockEvent
from app.recommender.collaborative import CollaborativeRecommender
from app.recommender.content_based import ContentBasedRecommender
from app.recommender.popularity import compute_popularity_scores

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_products(db: Session) -> list[dict]:
    products = db.query(Product).all()
    return [{"id": p.id, "text_blob": p.text_blob(), "price": p.price} for p in products]


def _load_customer_interactions(db: Session) -> list[tuple[int, int, float]]:
    rows = db.query(CustomerInteraction).all()
    agg = defaultdict(float)
    for r in rows:
        agg[(r.user_id, r.product_id)] += r.weight
    return [(u, p, w) for (u, p), w in agg.items()]


def _load_reseller_events(db: Session) -> list[tuple[int, int, float]]:
    rows = db.query(ResellerStockEvent).all()
    agg = defaultdict(float)
    for r in rows:
        # Confidence = volume stocked scaled by how well it actually sold,
        # so a reseller who over-stocked a dud contributes less signal than
        # one who stocked a smaller batch that sold through completely.
        confidence = r.units_stocked * max(r.sell_through_rate, 0.05)
        agg[(r.reseller_id, r.product_id)] += confidence
    return [(u, p, w) for (u, p), w in agg.items()]


def train_all() -> dict:
    db = SessionLocal()
    try:
        products = _load_products(db)
        customer_events = _load_customer_interactions(db)
        reseller_events = _load_reseller_events(db)

        content_model = ContentBasedRecommender().fit(products)

        customer_cf = CollaborativeRecommender(
            factors=settings.collaborative_factors,
            regularization=settings.collaborative_regularization,
            iterations=settings.collaborative_iterations,
        ).fit(customer_events)

        reseller_cf = CollaborativeRecommender(
            factors=settings.collaborative_factors,
            regularization=settings.collaborative_regularization,
            iterations=settings.collaborative_iterations,
        ).fit(reseller_events)

        customer_interaction_rows = db.query(CustomerInteraction).all()
        popularity_events = [
            {"product_id": r.product_id, "weight": r.weight, "created_at": r.created_at}
            for r in customer_interaction_rows
        ]
        popularity_scores = compute_popularity_scores(popularity_events)

        os.makedirs(settings.model_artifact_dir, exist_ok=True)
        artifacts = {
            "content_model": content_model,
            "customer_cf": customer_cf,
            "reseller_cf": reseller_cf,
            "popularity_scores": popularity_scores,
        }
        with open(os.path.join(settings.model_artifact_dir, "artifacts.pkl"), "wb") as f:
            pickle.dump(artifacts, f)

        summary = {
            "products": len(products),
            "customer_interactions": len(customer_events),
            "reseller_events": len(reseller_events),
            "customer_cf_coverage": customer_cf.coverage(),
            "reseller_cf_coverage": reseller_cf.coverage(),
        }
        logger.info("Training complete: %s", summary)
        return summary
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(train_all())
