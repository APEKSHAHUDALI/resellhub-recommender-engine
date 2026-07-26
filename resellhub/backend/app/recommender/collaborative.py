"""
Collaborative filtering via implicit-feedback matrix factorization (ALS).

Used twice with the same code path:
  - customer_id x product_id  -> "customers who behaved like you also liked..."
  - reseller_id x product_id  -> "resellers with a similar stocking profile also stocked..."

Weighted by the confidence of the interaction (view < wishlist < cart < purchase,
or units_stocked * sell_through_rate for resellers), which is the standard
Hu/Koren/Volinsky implicit-feedback formulation ALS is built for.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class CollaborativeRecommender:
    def __init__(self, factors: int = 32, regularization: float = 0.05, iterations: int = 20, random_state: int = 42):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.user_index: dict[int, int] = {}   # external user_id -> matrix row
        self.item_index: dict[int, int] = {}   # external product_id -> matrix col
        self.index_item: dict[int, int] = {}   # matrix col -> external product_id
        self.user_items: sp.csr_matrix | None = None
        self.is_fitted = False

    def fit(self, interactions: list[tuple[int, int, float]]):
        """
        interactions: list of (user_id, product_id, confidence_weight)
        """
        if not interactions:
            self.is_fitted = False
            return self

        users = sorted({u for u, _, _ in interactions})
        items = sorted({i for _, i, _ in interactions})
        self.user_index = {u: idx for idx, u in enumerate(users)}
        self.item_index = {i: idx for idx, i in enumerate(items)}
        self.index_item = {idx: i for i, idx in self.item_index.items()}

        rows = [self.user_index[u] for u, _, _ in interactions]
        cols = [self.item_index[i] for _, i, _ in interactions]
        data = [max(w, 0.0) for _, _, w in interactions]

        self.user_items = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(users), len(items))
        )
        self.model.fit(self.user_items)
        self.is_fitted = True
        return self

    def recommend(self, user_id: int, n: int = 10, filter_already_interacted: bool = True) -> list[tuple[int, float]]:
        """Returns [(product_id, score), ...] sorted descending. Empty if unknown/cold-start user."""
        if not self.is_fitted or user_id not in self.user_index:
            return []
        row = self.user_index[user_id]
        ids, scores = self.model.recommend(
            row,
            self.user_items[row],
            N=n,
            filter_already_liked_items=filter_already_interacted,
        )
        return [(self.index_item[i], float(s)) for i, s in zip(ids, scores) if s > 0]

    def similar_items(self, product_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Item-item similarity in latent space - powers 'similar products' widgets."""
        if not self.is_fitted or product_id not in self.item_index:
            return []
        col = self.item_index[product_id]
        ids, scores = self.model.similar_items(col, N=n + 1)
        return [(self.index_item[i], float(s)) for i, s in zip(ids, scores) if self.index_item[i] != product_id][:n]

    def coverage(self) -> dict:
        """Diagnostic: how many users/items the model actually learned factors for."""
        if not self.is_fitted:
            return {"users": 0, "items": 0}
        return {"users": len(self.user_index), "items": len(self.item_index)}
