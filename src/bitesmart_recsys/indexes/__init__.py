"""Vector index backends behind one interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import IndexUnavailableError, VectorIndex
from .faiss_indexes import FaissFlatIndex, FaissHNSWIndex, FaissIVFIndex
from .filters import AnyOf, Condition, Filter, matches
from .numpy_exact import NumpyExactIndex
from .qdrant_index import QdrantIndex, to_qdrant_filter

_BACKENDS: dict[str, Callable[..., VectorIndex]] = {
    "numpy_exact": NumpyExactIndex,
    "faiss_flat": FaissFlatIndex,
    "faiss_hnsw": FaissHNSWIndex,
    "faiss_ivf": FaissIVFIndex,
    "qdrant": QdrantIndex,
}


def _hnswlib_factory(**kwargs: Any) -> VectorIndex:
    from .hnswlib_index import HnswlibIndex

    return HnswlibIndex(**kwargs)


_BACKENDS["hnswlib"] = _hnswlib_factory


def create_index(name: str, **kwargs: Any) -> VectorIndex:
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown index '{name}', choose from {sorted(_BACKENDS)}") from None
    return factory(**kwargs)


def backend_names() -> list[str]:
    return sorted(_BACKENDS)


__all__ = [
    "AnyOf",
    "Condition",
    "Filter",
    "IndexUnavailableError",
    "NumpyExactIndex",
    "QdrantIndex",
    "VectorIndex",
    "backend_names",
    "create_index",
    "matches",
    "to_qdrant_filter",
]
