"""
Content-based recommender: represents each product as a TF-IDF vector over its
text fields (title/brand/category/condition/description) plus a normalized
price feature, then ranks by cosine similarity.

This is what makes cold-start possible: a brand-new product with zero
interactions still has a content vector the moment it's added to the catalog,
so it can be recommended based on similarity to what a user already likes -
no interaction history required.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentBasedRecommender:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
        self.scaler = MinMaxScaler()
        self.product_ids: list[int] = []
        self.id_to_row: dict[int, int] = {}
        self.feature_matrix = None  # kept sparse - see fit()
        self.is_fitted = False

    def fit(self, products: list[dict]):
        """products: list of {id, text_blob, price}"""
        if not products:
            self.is_fitted = False
            return self

        self.product_ids = [p["id"] for p in products]
        self.id_to_row = {pid: i for i, pid in enumerate(self.product_ids)}
        text_matrix = self.vectorizer.fit_transform([p["text_blob"] for p in products])  # sparse
        prices = np.array([[p["price"]] for p in products], dtype=float)
        price_scaled = sp.csr_matrix(self.scaler.fit_transform(prices))

        # Stay sparse: a dense array here is what previously bloated a
        # ~4K-product catalog's model artifact to 150+ MB for no benefit,
        # since cosine_similarity accepts sparse input natively.
        self.feature_matrix = sp.hstack([text_matrix, price_scaled]).tocsr()
        self.is_fitted = True
        return self

    def _row_for(self, product_id: int):
        row_idx = self.id_to_row.get(product_id)
        if row_idx is None:
            return None
        return self.feature_matrix[row_idx]

    def similar_to_product(self, product_id: int, n: int = 10) -> list[tuple[int, float]]:
        if not self.is_fitted:
            return []
        row = self._row_for(product_id)
        if row is None:
            return []
        sims = cosine_similarity(row, self.feature_matrix)[0]
        ranked = sorted(zip(self.product_ids, sims), key=lambda x: -x[1])
        return [(pid, float(score)) for pid, score in ranked if pid != product_id][:n]

    def similar_to_profile(self, liked_product_ids: list[int], n: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """
        Builds a user "taste vector" as the mean of their liked products'
        content vectors, then ranks the whole catalog against it. This is how
        a customer or reseller with a short history still gets a personalized
        (not just generic-popular) list.
        """
        if not self.is_fitted or not liked_product_ids:
            return []
        rows = [self._row_for(pid) for pid in liked_product_ids]
        rows = [r for r in rows if r is not None]
        if not rows:
            return []
        profile = sp.vstack(rows).mean(axis=0)
        profile = sp.csr_matrix(profile)
        sims = cosine_similarity(profile, self.feature_matrix)[0]
        exclude = exclude or set()
        ranked = sorted(zip(self.product_ids, sims), key=lambda x: -x[1])
        return [(pid, float(score)) for pid, score in ranked if pid not in exclude][:n]
