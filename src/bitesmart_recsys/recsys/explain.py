"""Human readable reasons attached to each recommendation."""

from __future__ import annotations

from .config import RecommendationConfig
from .scoring import ScoredCandidate

_TEMPLATES = {
    "similarity": "Very similar to the product you chose",
    "brand_diversity": "Adds variety in brands",
    "category_diversity": "Broadens the category coverage",
}


class ExplanationBuilder:
    def __init__(self, config: RecommendationConfig) -> None:
        self.cfg = config

    def _ranked_factors(self, item: ScoredCandidate) -> list[str]:
        w, f = self.cfg.scoring, item.features
        contributions = {
            "similarity": f.similarity * w.similarity,
            "brand_diversity": f.brand_diversity * w.brand_diversity,
            "category_diversity": f.category_diversity * w.category_diversity,
        }
        return sorted(contributions, key=lambda name: contributions[name], reverse=True)

    def build(self, items: list[ScoredCandidate]) -> list[ScoredCandidate]:
        if not self.cfg.enable_explanations:
            return items
        for item in items:
            reasons = [_TEMPLATES[name] for name in self._ranked_factors(item)]
            nutr = item.payload.get("nutriments") or {}
            if item.penalties.get("category_transition"):
                reasons.append("Comes from a slightly different product category")
            if nutr.get("sugars_100g") is not None and nutr["sugars_100g"] <= 5:
                reasons.append("Low in sugar")
            item.debug["reasons"] = reasons[: self.cfg.max_reasons]
        return items
