"""Common interface for all vector index backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .filters import Filter, compute_mask

Matrix = NDArray[np.float32]
IdMatrix = NDArray[np.int64]


class IndexUnavailableError(RuntimeError):
    """The optional library behind a backend is not installed."""


class VectorIndex(ABC):
    """Inner product index over L2-normalised vectors (cosine similarity).

    Points are identified by their row number 0..n-1. `search` returns ids padded with -1
    and scores padded with -inf when fewer than k points match.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._payloads: Sequence[Mapping[str, Any]] = ()
        self._mask_cache: dict[Filter, NDArray[np.bool_]] = {}
        self.n = 0
        self.dim = 0

    # ---- lifecycle
    def build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")
        self.n, self.dim = vectors.shape
        self._payloads = payloads
        self._mask_cache.clear()
        self._build(np.ascontiguousarray(vectors, dtype=np.float32), payloads)

    @abstractmethod
    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None: ...

    @abstractmethod
    def search(
        self, queries: Matrix, k: int, flt: Filter | None = None
    ) -> tuple[IdMatrix, Matrix]: ...

    def set_search_params(self, **params: Any) -> None:
        """Set query-time knobs (ef_search, nprobe, ...). Unknown names raise TypeError."""
        if params:
            raise TypeError(f"{self.name} has no search parameters {sorted(params)}")

    def describe(self) -> dict[str, Any]:
        return {"index": self.name}

    def memory_bytes(self) -> int | None:
        """Size of the index structure in bytes, or None when the backend cannot tell."""
        return None

    def close(self) -> None:  # noqa: B027 - optional hook
        return None

    # ---- helpers for subclasses
    def allowed_mask(self, flt: Filter | None) -> NDArray[np.bool_] | None:
        """Boolean mask of points that pass `flt`, cached per filter. None means no filter."""
        if flt is None or flt.is_empty:
            return None
        mask = self._mask_cache.get(flt)
        if mask is None:
            mask = compute_mask(self._payloads, flt)
            self._mask_cache[flt] = mask
        return mask

    @staticmethod
    def empty_result(nq: int, k: int) -> tuple[IdMatrix, Matrix]:
        return (
            np.full((nq, k), -1, dtype=np.int64),
            np.full((nq, k), -np.inf, dtype=np.float32),
        )
