"""
Offline evaluation harness. Run against a time-based train/test split:
train on interactions before cutoff_date, evaluate whether the model's
top-N recommendations for each user contain what they *actually* interacted
with after cutoff_date. This is the standard way to get an honest, reportable
accuracy number instead of eyeballing recommendations.
"""
from __future__ import annotations
import math


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(top_k) if pid in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    recommend_fn,
    test_ground_truth: dict[int, set[int]],
    k: int = 10,
) -> dict:
    """
    recommend_fn: callable(user_id) -> list[int] of recommended product_ids
    test_ground_truth: {user_id: set(product_ids the user actually interacted
                        with after the train/test cutoff)}
    """
    precisions, recalls, ndcgs = [], [], []
    users_with_hits = 0

    for user_id, relevant in test_ground_truth.items():
        if not relevant:
            continue
        recommended = recommend_fn(user_id)
        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        g = ndcg_at_k(recommended, relevant, k)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(g)
        if p > 0:
            users_with_hits += 1

    n = len(precisions) or 1
    return {
        "k": k,
        "users_evaluated": len(precisions),
        f"precision_at_{k}": round(sum(precisions) / n, 4),
        f"recall_at_{k}": round(sum(recalls) / n, 4),
        f"ndcg_at_{k}": round(sum(ndcgs) / n, 4),
        "hit_rate": round(users_with_hits / n, 4),
    }
