"""FAISS backends. Metadata filters are applied inside the search with an ID bitmap selector."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import faiss
import numpy as np

from .base import IdMatrix, Matrix, VectorIndex
from .filters import Filter


class _FaissIndex(VectorIndex):
    _index: Any

    def _selector(self, flt: Filter | None) -> tuple[Any, Any] | None:
        mask = self.allowed_mask(flt)
        if mask is None:
            return None
        bits = np.packbits(mask, bitorder="little")
        selector = faiss.IDSelectorBitmap(self.n, faiss.swig_ptr(bits))
        return selector, bits  # `bits` must outlive the selector

    def _params(self, selector: Any) -> Any:
        params = faiss.SearchParameters()
        params.sel = selector
        return params

    def search(self, queries: Matrix, k: int, flt: Filter | None = None) -> tuple[IdMatrix, Matrix]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        picked = self._selector(flt)
        if picked is not None and not picked[1].any():
            return self.empty_result(len(queries), k)
        params = self._params(picked[0]) if picked else None
        scores, ids = self._index.search(queries, k, params=params)
        scores = np.where(ids < 0, -np.inf, scores).astype(np.float32)
        return ids.astype(np.int64), scores

    def memory_bytes(self) -> int | None:
        return len(faiss.serialize_index(self._index))


class FaissFlatIndex(_FaissIndex):
    name = "faiss_flat"

    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors)


class FaissHNSWIndex(_FaissIndex):
    name = "faiss_hnsw"

    def __init__(self, m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        super().__init__()
        self.m, self.ef_construction, self.ef_search = m, ef_construction, ef_search

    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.add(vectors)
        self._index = index

    def _params(self, selector: Any) -> Any:
        params = faiss.SearchParametersHNSW()  # type: ignore[attr-defined]
        params.efSearch = self.ef_search
        params.sel = selector
        return params

    def search(self, queries: Matrix, k: int, flt: Filter | None = None) -> tuple[IdMatrix, Matrix]:
        # An unfiltered search must still honour efSearch, so always pass parameters.
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        picked = self._selector(flt)
        if picked is not None and not picked[1].any():
            return self.empty_result(len(queries), k)
        params = self._params(picked[0] if picked else None)
        scores, ids = self._index.search(queries, k, params=params)
        scores = np.where(ids < 0, -np.inf, scores).astype(np.float32)
        return ids.astype(np.int64), scores

    def set_search_params(self, **params: Any) -> None:
        self.ef_search = self._accept_params(params, {"ef_search"}).get("ef_search", self.ef_search)

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.name,
            "M": self.m,
            "efConstruction": self.ef_construction,
            "efSearch": self.ef_search,
        }


class FaissIVFIndex(_FaissIndex):
    name = "faiss_ivf"

    def __init__(self, nlist: int | None = None, nprobe: int = 8) -> None:
        super().__init__()
        self.nlist, self.nprobe = nlist, nprobe
        self._quantizer: Any = None

    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        nlist = self.nlist or max(1, int(4 * math.sqrt(self.n)))
        nlist = min(nlist, self.n)
        self.nlist = nlist
        self._quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(self._quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        train = vectors
        if self.n > 256 * nlist:
            rng = np.random.default_rng(0)
            train = vectors[rng.choice(self.n, 256 * nlist, replace=False)]
        index.train(train)
        index.add(vectors)
        self._index = index

    def _params(self, selector: Any) -> Any:
        params = faiss.SearchParametersIVF()
        params.nprobe = self.nprobe
        params.sel = selector
        return params

    def search(self, queries: Matrix, k: int, flt: Filter | None = None) -> tuple[IdMatrix, Matrix]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        picked = self._selector(flt)
        if picked is not None and not picked[1].any():
            return self.empty_result(len(queries), k)
        params = self._params(picked[0] if picked else None)
        scores, ids = self._index.search(queries, k, params=params)
        scores = np.where(ids < 0, -np.inf, scores).astype(np.float32)
        return ids.astype(np.int64), scores

    def set_search_params(self, **params: Any) -> None:
        self.nprobe = self._accept_params(params, {"nprobe"}).get("nprobe", self.nprobe)

    def describe(self) -> dict[str, Any]:
        return {"index": self.name, "nlist": self.nlist, "nprobe": self.nprobe}
