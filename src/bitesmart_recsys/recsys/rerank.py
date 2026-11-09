"""Reranking: soft category transition penalty and per-brand/category diversity penalty."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .config import RecommendationConfig
from .scoring import ScoredCandidate, brand_key, category_key

logger = logging.getLogger(__name__)


def category_hierarchy(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("categories_hierarchy")
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if isinstance(c, str) and c.strip()]
    cat = payload.get("category")
    return [cat.strip()] if isinstance(cat, str) and cat.strip() else []


def hierarchy_transition_score(src: list[str], cand: list[str]) -> tuple[float, str]:
    """Order-agnostic overlap of category sets mapped to [0, 1], with a reason label."""
    if not src or not cand:
        return 0.0, "missing_hierarchy"
    a, b = set(src), set(cand)
    if a == b:
        return 1.0, "exact_match"
    shared = a & b
    if not shared:
        return 0.0, "no_shared_category"
    return round(0.35 + 0.65 * len(shared) / max(len(a), len(b)), 6), "partial_overlap"


class Reranker:
    def __init__(self, config: RecommendationConfig) -> None:
        self.cfg = config

    def apply(
        self,
        scored: list[ScoredCandidate],
        source_payload: Mapping[str, Any] | None = None,
    ) -> list[ScoredCandidate]:
        if not scored:
            return []
        scored = self._category_transition(scored, source_payload)
        return self._diversity_penalty(scored)

    def _category_transition(
        self, scored: list[ScoredCandidate], source_payload: Mapping[str, Any] | None
    ) -> list[ScoredCandidate]:
        cfg = self.cfg.category_transition
        if not cfg.enabled or not source_payload:
            return scored
        src = category_hierarchy(source_payload)
        if not src and cfg.fallback_missing == "skip":
            return scored

        penalized = 0
        for item in scored:
            cand = category_hierarchy(item.payload)
            if not cand and cfg.fallback_missing == "skip":
                continue
            t_score, reason = hierarchy_transition_score(src, cand)
            penalty = 0.0
            if t_score < cfg.threshold:
                gap = (cfg.threshold - t_score) / max(cfg.threshold, 1e-9)
                penalty = cfg.alpha * gap**cfg.gamma
            item.debug.update(
                category_transition_score=round(t_score, 4), category_transition_reason=reason
            )
            if penalty:
                item.score = max(0.0, item.score - penalty)
                item.penalties["category_transition"] = round(penalty, 4)
                penalized += 1
        scored.sort(key=lambda s: s.score, reverse=True)
        logger.info("category transition applied", extra={"penalized": penalized})
        return scored

    def _diversity_penalty(self, scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
        d = self.cfg.diversity
        brands: Counter[str] = Counter()
        cats: Counter[str] = Counter()
        for item in scored:
            penalty = 0.0
            b, c = brand_key(item.payload), category_key(item.payload)
            if b:
                brands[b] += 1
                penalty += d.brand_penalty * max(0, brands[b] - d.max_per_brand)
            if c:
                cats[c] += 1
                penalty += d.category_penalty * max(0, cats[c] - d.max_per_category)
            if penalty:
                item.score = max(0.0, item.score - penalty)
                item.penalties["diversity"] = round(penalty, 4)
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored
