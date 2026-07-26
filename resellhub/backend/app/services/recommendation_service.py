"""
The layer the API routes actually call. Responsible for:
  - lazily loading trained model artifacts from disk (hot-reloadable after retrain)
  - checking Redis cache before doing any computation
  - assembling the right signals (CF + content + popularity) per surface
  - applying the hybrid blend and, for resellers, a margin-aware business rule
"""
from __future__ import annotations
import os
import pickle
import threading

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.cache import get_cached, set_cached
from app.models import Product, CustomerInteraction, ResellerStockEvent
from app.recommender.hybrid import HybridRecommender

settings = get_settings()
_lock = threading.Lock()
_artifacts_cache = {"mtime": None, "data": None}


def _artifacts_path() -> str:
    return os.path.join(settings.model_artifact_dir, "artifacts.pkl")


def _load_artifacts():
    """Reloads from disk only if the file changed since last load (cheap hot-reload)."""
    path = _artifacts_path()
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    with _lock:
        if _artifacts_cache["mtime"] != mtime:
            with open(path, "rb") as f:
                _artifacts_cache["data"] = pickle.load(f)
            _artifacts_cache["mtime"] = mtime
        return _artifacts_cache["data"]


class RecommendationService:
    def __init__(self, db: Session):
        self.db = db
        self.hybrid = HybridRecommender(
            w_cf=settings.hybrid_weight_collaborative,
            w_content=settings.hybrid_weight_content,
            w_pop=settings.hybrid_weight_popularity,
            min_interactions_for_cf=settings.min_interactions_for_cf,
        )

    def _product_map(self, ids: list[int]) -> dict[int, Product]:
        if not ids:
            return {}
        rows = self.db.query(Product).filter(Product.id.in_(ids)).all()
        return {p.id: p for p in rows}

    def _serialize(self, ranked: list[dict]) -> list[dict]:
        products = self._product_map([r["product_id"] for r in ranked])
        out = []
        for r in ranked:
            p = products.get(r["product_id"])
            if not p:
                continue
            out.append({
                "product_id": p.id,
                "sku": p.sku,
                "title": p.title,
                "category": p.category,
                "price": p.price,
                "condition": p.condition,
                "score": r["score"],
                "reason": r["explanation"],
            })
        return out

    # ---------------------------------------------------------------- customer
    def recommend_for_customer(self, user_id: int, n: int = 10) -> list[dict]:
        cache_key = f"reco:customer:{user_id}:{n}"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached

        artifacts = _load_artifacts()
        if artifacts is None:
            return []

        history = self.db.query(CustomerInteraction).filter(CustomerInteraction.user_id == user_id).all()
        liked_ids = [h.product_id for h in history]
        exclude = set(liked_ids)

        cf_scores = artifacts["customer_cf"].recommend(user_id, n=n * 3)
        content_scores = artifacts["content_model"].similar_to_profile(liked_ids, n=n * 3, exclude=exclude)
        pop_scores = list(artifacts["popularity_scores"].items())
        pop_scores = [(pid, s) for pid, s in pop_scores if pid not in exclude][: n * 3]

        ranked = self.hybrid.recommend(
            n_interactions=len(history),
            cf_scores=cf_scores,
            content_scores=content_scores,
            popularity_scores=pop_scores,
            n=n,
        )
        result = self._serialize(ranked)
        set_cached(cache_key, result)
        return result

    # ---------------------------------------------------------------- reseller
    def recommend_for_reseller(self, reseller_id: int, n: int = 10) -> list[dict]:
        cache_key = f"reco:reseller:{reseller_id}:{n}"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached

        artifacts = _load_artifacts()
        if artifacts is None:
            return []

        history = self.db.query(ResellerStockEvent).filter(ResellerStockEvent.reseller_id == reseller_id).all()
        stocked_ids = [h.product_id for h in history]
        exclude = set(stocked_ids)

        cf_scores = artifacts["reseller_cf"].recommend(reseller_id, n=n * 3)
        content_scores = artifacts["content_model"].similar_to_profile(stocked_ids, n=n * 3, exclude=exclude)
        pop_scores = list(artifacts["popularity_scores"].items())
        pop_scores = [(pid, s) for pid, s in pop_scores if pid not in exclude][: n * 3]

        # Business rule specific to the reseller surface: boost candidates
        # with a healthy margin, since "what should I stock" should weigh
        # profitability, not just what's trending or similar.
        candidate_ids = {pid for pid, _ in cf_scores} | {pid for pid, _ in content_scores} | {pid for pid, _ in pop_scores}
        products = self._product_map(list(candidate_ids))
        margin_boost = {}
        for pid, p in products.items():
            if p.price > 0:
                margin = (p.price - p.wholesale_cost) / p.price
                margin_boost[pid] = max(margin - 0.3, 0.0)  # only boost above a 30% margin baseline

        ranked = self.hybrid.recommend(
            n_interactions=len(history),
            cf_scores=cf_scores,
            content_scores=content_scores,
            popularity_scores=pop_scores,
            n=n,
            margin_boost=margin_boost,
        )
        result = self._serialize(ranked)
        set_cached(cache_key, result)
        return result

    # ------------------------------------------------------------ similar items
    def similar_products(self, product_id: int, n: int = 8) -> list[dict]:
        cache_key = f"reco:similar:{product_id}:{n}"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached
        artifacts = _load_artifacts()
        if artifacts is None:
            return []
        content_scores = artifacts["content_model"].similar_to_product(product_id, n=n)
        ranked = [{"product_id": pid, "score": s, "explanation": "similar attributes"} for pid, s in content_scores]
        result = self._serialize(ranked)
        set_cached(cache_key, result)
        return result
