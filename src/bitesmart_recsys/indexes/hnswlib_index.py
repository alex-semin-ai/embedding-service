"""hnswlib backend. Filtering uses hnswlib's `filter=` callback over an id mask."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .base import IdMatrix, IndexUnavailableError, Matrix, VectorIndex
from .filters import Filter


class HnswlibIndex(VectorIndex):
    name = "hnswlib"

    def __init__(self, m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        super().__init__()
        try:
            import hnswlib
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise IndexUnavailableError("hnswlib is not installed, use the 'ann' extra") from exc
        self._hnswlib = hnswlib
        self.m, self.ef_construction, self.ef_search = m, ef_construction, ef_search
        self._index: Any = None

    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        index = self._hnswlib.Index(space="ip", dim=self.dim)
        index.init_index(max_elements=self.n, ef_construction=self.ef_construction, M=self.m)
        index.add_items(vectors, np.arange(self.n))
        index.set_ef(self.ef_search)
        self._index = index

    def _query_one(self, q: Matrix, k: int, mask: Any) -> tuple[list[int], list[float]]:
        kwargs: dict[str, Any] = {}
        if mask is not None:
            kwargs["filter"] = lambda i: bool(mask[i])
        current = k
        while current >= 1:
            try:
                labels, dist = self._index.knn_query(q[None, :], k=current, **kwargs)
                return labels[0].tolist(), (1.0 - dist[0]).tolist()
            except RuntimeError:
                # hnswlib raises when the filtered graph walk finds fewer than k points
                current //= 2
        return [], []

    def search(
        self, queries: Matrix, k: int, flt: Filter | None = None
    ) -> tuple[IdMatrix, Matrix]:
        self._index.set_ef(max(self.ef_search, k))
        mask = self.allowed_mask(flt)
        ids_out, scores_out = self.empty_result(len(queries), k)
        if mask is not None and not mask.any():
            return ids_out, scores_out
        for row, q in enumerate(queries):
            labels, scores = self._query_one(np.ascontiguousarray(q), k, mask)
            ids_out[row, : len(labels)] = labels
            scores_out[row, : len(scores)] = scores
        return ids_out, scores_out

    def set_search_params(self, ef_search: int | None = None) -> None:
        if ef_search is not None:
            self.ef_search = int(ef_search)

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.name,
            "M": self.m,
            "efConstruction": self.ef_construction,
            "efSearch": self.ef_search,
        }

    def memory_bytes(self) -> int | None:
        if self._index is None:
            return None
        # vectors + level-0 links (2M ids) + upper layers, close to hnswlib's own estimate
        per_element = self.dim * 4 + self.m * 2 * 4 + 8 + self.m * 4
        return int(self.n * per_element)
