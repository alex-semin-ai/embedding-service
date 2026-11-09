"""End to end substitution pipeline.

Index search (constraints pushed down) -> hard constraint verification -> scoring ->
rerank (category transition, diversity) -> explanations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from bitesmart_recsys.indexes.base import Matrix, VectorIndex

from .catalog import Catalog
from .config import RecommendationConfig
from .constraints import build_filter, verify_hard_constraints
from .explain import ExplanationBuilder
from .rerank import Reranker
from .scoring import Candidate, CandidateScorer

logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    id: str
    product_name: str
    brand: str | None
    score: float
    similarity: float
    reasons: list[str]
    penalties: dict[str, float] = Field(default_factory=dict)
    nutriments: dict[str, float] = Field(default_factory=dict)


class RecommendationResult(BaseModel):
    source_id: str
    source_name: str
    items: list[Recommendation]
    stats: dict[str, Any]


class Recommender:
    def __init__(self, catalog: Catalog, index: VectorIndex) -> None:
        self.catalog = catalog
        self.index = index

    @classmethod
    def build(cls, catalog: Catalog, index: VectorIndex) -> Recommender:
        index.build(catalog.vectors, catalog.payloads)
        return cls(catalog, index)

    def search_vector(
        self, vector: Matrix, k: int, config: RecommendationConfig | None = None
    ) -> list[Candidate]:
        """Plain filtered similarity search for a query vector (no source product)."""
        config = config or RecommendationConfig()
        flt = build_filter(config, None)
        ids, scores = self.index.search(np.asarray(vector, np.float32)[None, :], k, flt)
        return self._candidates(ids[0], scores[0], config.similarity.min_score)

    def _candidates(self, ids: Any, scores: Any, min_score: float) -> list[Candidate]:
        out: list[Candidate] = []
        for row, score in zip(ids, scores, strict=True):
            if row < 0 or score < min_score:
                continue
            out.append(
                Candidate(
                    id=self.catalog.products[row].id,
                    similarity=float(score),
                    payload=self.catalog.payloads[row],
                )
            )
        return out

    def recommend(
        self,
        source_id: str,
        config: RecommendationConfig | None = None,
        top_n: int = 7,
    ) -> RecommendationResult:
        config = config or RecommendationConfig()
        started = time.perf_counter()
        row = self.catalog.row_of(source_id)
        source = self.catalog.payloads[row]

        flt = build_filter(config, source, source_id)
        ids, scores = self.index.search(
            self.catalog.vectors[row : row + 1], config.candidate_search.limit, flt
        )
        raw = self._candidates(ids[0], scores[0], config.similarity.min_score)
        verified = verify_hard_constraints(raw, config, source, source_id)
        scored = CandidateScorer(config).score(verified)
        scored = Reranker(config).apply(scored, source)
        scored = ExplanationBuilder(config).build(scored)

        items = [
            Recommendation(
                id=s.id,
                product_name=s.payload["product_name"],
                brand=s.payload.get("brand"),
                score=round(s.score, 6),
                similarity=round(s.features.similarity, 6),
                reasons=s.debug.get("reasons", []),
                penalties=s.penalties,
                nutriments=s.payload.get("nutriments", {}),
            )
            for s in scored[:top_n]
        ]
        elapsed_ms = round(1000 * (time.perf_counter() - started), 3)
        stats = {
            "index": self.index.name,
            "retrieved": len(raw),
            "after_verification": len(verified),
            "returned": len(items),
            "latency_ms": elapsed_ms,
        }
        logger.info("recommendation built", extra={"source_id": source_id, **stats})
        return RecommendationResult(
            source_id=source_id, source_name=source["product_name"], items=items, stats=stats
        )
