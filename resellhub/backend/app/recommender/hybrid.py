"""
Hybrid scoring layer. This is the piece that actually gets called by the API.

Strategy (this is the part worth explaining in an interview):
  1. Warm users (>= min_interactions_for_cf interactions): blend collaborative
     filtering, content-based similarity, and popularity with configured weights.
  2. Cold-start users (some history, below the CF threshold): drop CF entirely
     (it's unreliable with too few signals) and reweight content + popularity
     to fill the gap - this avoids the classic "cold start gives garbage
     recommendations" failure mode.
  3. Brand-new users (zero history): popularity only, optionally filtered to
     a declared category preference.

Scores from each sub-model are min-max normalized per user before blending,
since CF scores, cosine similarities, and popularity counts live on
incomparable scales.
"""
from __future__ import annotations


def _normalize(scored: list[tuple[int, float]]) -> dict[int, float]:
    if not scored:
        return {}
    values = [s for _, s in scored]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {pid: 1.0 for pid, _ in scored}
    return {pid: (s - lo) / (hi - lo) for pid, s in scored}


class HybridRecommender:
    def __init__(self, w_cf: float, w_content: float, w_pop: float, min_interactions_for_cf: int):
        self.w_cf = w_cf
        self.w_content = w_content
        self.w_pop = w_pop
        self.min_interactions_for_cf = min_interactions_for_cf

    def recommend(
        self,
        *,
        n_interactions: int,
        cf_scores: list[tuple[int, float]],
        content_scores: list[tuple[int, float]],
        popularity_scores: list[tuple[int, float]],
        n: int = 10,
        margin_boost: dict[int, float] | None = None,
    ) -> list[dict]:
        cf_norm = _normalize(cf_scores)
        content_norm = _normalize(content_scores)
        pop_norm = _normalize(popularity_scores)

        if n_interactions >= self.min_interactions_for_cf and cf_norm:
            weights = {"cf": self.w_cf, "content": self.w_content, "pop": self.w_pop}
        elif n_interactions > 0:
            # cold-start-but-not-empty: redistribute the CF weight proportionally
            total = self.w_content + self.w_pop
            weights = {"cf": 0.0, "content": self.w_content / total, "pop": self.w_pop / total}
        else:
            weights = {"cf": 0.0, "content": 0.0, "pop": 1.0}

        candidates = set(cf_norm) | set(content_norm) | set(pop_norm)
        results = []
        for pid in candidates:
            score = (
                weights["cf"] * cf_norm.get(pid, 0.0)
                + weights["content"] * content_norm.get(pid, 0.0)
                + weights["pop"] * pop_norm.get(pid, 0.0)
            )
            if margin_boost:
                score *= (1.0 + margin_boost.get(pid, 0.0))
            results.append({
                "product_id": pid,
                "score": round(score, 6),
                "explanation": self._explain(pid, cf_norm, content_norm, pop_norm, weights),
            })

        return sorted(results, key=lambda r: -r["score"])[:n]

    @staticmethod
    def _explain(pid, cf_norm, content_norm, pop_norm, weights) -> str:
        """Human-readable reason, shown in the UI - a small but real touch of explainability."""
        contributions = []
        if weights["cf"] > 0 and pid in cf_norm:
            contributions.append("similar users' behavior")
        if weights["content"] > 0 and pid in content_norm:
            contributions.append("similar to items you've engaged with")
        if weights["pop"] > 0 and pid in pop_norm:
            contributions.append("currently trending")
        return " + ".join(contributions) if contributions else "catalog match"
