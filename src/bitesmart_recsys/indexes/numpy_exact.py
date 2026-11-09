"""Exact brute-force search in NumPy. This is the ground truth for recall."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .base import IdMatrix, Matrix, VectorIndex
from .filters import Filter


class NumpyExactIndex(VectorIndex):
    name = "numpy_exact"
    _CHUNK = 256

    def __init__(self) -> None:
        super().__init__()
        self._vectors: Matrix = np.zeros((0, 0), dtype=np.float32)

    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        self._vectors = vectors

    def search(
        self, queries: Matrix, k: int, flt: Filter | None = None
    ) -> tuple[IdMatrix, Matrix]:
        mask = self.allowed_mask(flt)
        ids_out, scores_out = self.empty_result(len(queries), k)
        kk = min(k, self.n)
        for start in range(0, len(queries), self._CHUNK):
            block = queries[start : start + self._CHUNK] @ self._vectors.T
            if mask is not None:
                block[:, ~mask] = -np.inf
            top = np.argpartition(-block, kk - 1, axis=1)[:, :kk] if kk < self.n else (
                np.tile(np.arange(self.n), (len(block), 1))
            )
            top_scores = np.take_along_axis(block, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top = np.take_along_axis(top, order, axis=1)
            top_scores = np.take_along_axis(top_scores, order, axis=1)
            valid = np.isfinite(top_scores)
            rows = slice(start, start + len(block))
            ids_out[rows, :kk] = np.where(valid, top, -1)
            scores_out[rows, :kk] = top_scores
        return ids_out, scores_out

    def memory_bytes(self) -> int | None:
        return int(self._vectors.nbytes)
