"""Candidate scoring: similarity plus brand/category diversity (ported)."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from .config import RecommendationConfig

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    id: str
    similarity: float
    payload: dict[str, Any]


class ScoringFeatures(BaseModel):
    similarity: float
    brand_diversity: float
    category_diversity: float


class ScoredCandidate(BaseModel):
    id: str
    score: float
    features: ScoringFeatures
    payload: dict[str, Any]
    penalties: dict[str, float] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


def first_scalar(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, int | float | bool):
        return str(value)
    if isinstance(value, list | tuple | set):
        for item in value:
            found = first_scalar(item)
            if found:
                return found
    return None


def brand_key(payload: Mapping[str, Any]) -> str | None:
    return first_scalar(payload.get("brand")) or first_scalar(payload.get("brands_tags"))


def category_key(payload: Mapping[str, Any]) -> str | None:
    """Leaf category, the most specific entry of the hierarchy (root to leaf order)."""
    cats = payload.get("categories_hierarchy")
    if isinstance(cats, list) and cats:
        return first_scalar(cats[-1])
    return first_scalar(payload.get("category"))


def normalized_entropy(values: Iterable[str | None]) -> float:
    counts = Counter(v for v in values if v)
    total = sum(counts.values())
    if len(counts) <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    return entropy / math.log(len(counts))


def normalize_scores(items: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Min-max normalise scores to [0, 1] in place."""
    if not items:
        return items
    lo = min(i.score for i in items)
    hi = max(i.score for i in items)
    for item in items:
        item.score = 1.0 if hi <= lo else (item.score - lo) / (hi - lo)
    return items


class CandidateScorer:
    def __init__(self, config: RecommendationConfig) -> None:
        self.cfg = config

    @staticmethod
    def _inverse_frequency(keys: list[str | None]) -> dict[str, float]:
        total = max(len(keys), 1)
        return {k: 1.0 - c / total for k, c in Counter(k for k in keys if k).items()}

    def score(self, candidates: list[Candidate]) -> list[ScoredCandidate]:
        if not candidates:
            return []
        w = self.cfg.scoring
        brands = [brand_key(c.payload) for c in candidates]
        cats = [category_key(c.payload) for c in candidates]
        brand_inv = self._inverse_frequency(brands)
        cat_inv = self._inverse_frequency(cats)

        scored: list[ScoredCandidate] = []
        for cand, b, c in zip(candidates, brands, cats, strict=True):
            feats = ScoringFeatures(
                similarity=cand.similarity,
                brand_diversity=brand_inv.get(b, 0.0) if b else 0.0,
                category_diversity=cat_inv.get(c, 0.0) if c else 0.0,
            )
            value = (
                feats.similarity * w.similarity
                + feats.brand_diversity * w.brand_diversity
                + feats.category_diversity * w.category_diversity
            )
            scored.append(
                ScoredCandidate(id=cand.id, score=value, features=feats, payload=cand.payload)
            )
        scored.sort(key=lambda s: s.score, reverse=True)

        if self.cfg.diversity.incremental_enabled:
            self._apply_incremental_bonus(scored)

        top_k = self.cfg.candidate_search.top_k
        logger.info(
            "scoring done",
            extra={"candidates": len(candidates), "kept": min(top_k, len(scored))},
        )
        return scored[:top_k]

    def _apply_incremental_bonus(self, scored: list[ScoredCandidate]) -> None:
        """Items from brands/categories not yet seen higher in the list get a decaying bonus."""
        d, w = self.cfg.diversity, self.cfg.scoring
        brand_seen: Counter[str] = Counter()
        cat_seen: Counter[str] = Counter()
        for item in scored:
            bonus = 0.0
            b, c = brand_key(item.payload), category_key(item.payload)
            if b:
                bonus += w.brand_diversity * 0.15 * math.exp(-d.brand_alpha * brand_seen[b])
                brand_seen[b] += 1
            if c:
                bonus += w.category_diversity * 0.15 * math.exp(-d.category_alpha * cat_seen[c])
                cat_seen[c] += 1
            if bonus > 0:
                item.score += bonus
                item.debug["diversity_bonus"] = round(bonus, 6)
        scored.sort(key=lambda s: s.score, reverse=True)
