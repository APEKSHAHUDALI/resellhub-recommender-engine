"""
Popularity fallback. Used when we have neither interaction history nor a
content profile for a user (e.g. a brand-new signup with zero activity).

Popularity is computed with a recency-weighted count so the catalog doesn't
calcify around whatever sold well when the store first launched.
"""
from __future__ import annotations
from datetime import datetime, timezone


def compute_popularity_scores(events: list[dict], half_life_days: float = 14.0) -> dict[int, float]:
    """
    events: list of {product_id, weight, created_at (datetime)}
    Returns {product_id: decayed_score}, higher = more popular right now.
    """
    now = datetime.now(timezone.utc)
    scores: dict[int, float] = {}
    for e in events:
        created = e["created_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max((now - created).total_seconds() / 86400.0, 0.0)
        decay = 0.5 ** (age_days / half_life_days)
        scores[e["product_id"]] = scores.get(e["product_id"], 0.0) + e["weight"] * decay
    return scores


def top_n_popular(scores: dict[int, float], n: int = 10, exclude: set[int] | None = None,
                   category_filter: set[int] | None = None) -> list[tuple[int, float]]:
    exclude = exclude or set()
    items = [(pid, s) for pid, s in scores.items() if pid not in exclude]
    if category_filter:
        items = [(pid, s) for pid, s in items if pid in category_filter]
    return sorted(items, key=lambda x: -x[1])[:n]
