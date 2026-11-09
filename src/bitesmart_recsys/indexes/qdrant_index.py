"""Qdrant backend, either a real server (url) or the embedded local mode (':memory:').

The embedded mode is exact brute force, so only a server gives a fair HNSW comparison.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .base import IdMatrix, Matrix, VectorIndex
from .filters import AnyOf, Condition, Filter, Node

logger = logging.getLogger(__name__)

_RANGE = {"gte": "gte", "lte": "lte"}


def _condition(cond: Condition) -> qm.FieldCondition:
    if cond.op == "eq":
        return qm.FieldCondition(key=cond.field, match=qm.MatchValue(value=cond.value))
    if cond.op == "any":
        return qm.FieldCondition(key=cond.field, match=qm.MatchAny(any=list(cond.value)))
    return qm.FieldCondition(key=cond.field, range=qm.Range(**{_RANGE[cond.op]: cond.value}))


def _node(node: Node) -> qm.Condition:
    if isinstance(node, AnyOf):
        return qm.Filter(should=[_condition(c) for c in node.conditions])
    return _condition(node)


def to_qdrant_filter(flt: Filter | None) -> qm.Filter | None:
    """Translate the neutral filter into a Qdrant filter."""
    if flt is None or flt.is_empty:
        return None
    return qm.Filter(
        must=[_node(n) for n in flt.must] or None,
        must_not=[_node(n) for n in flt.must_not] or None,
    )


class QdrantIndex(VectorIndex):
    name = "qdrant"

    def __init__(
        self,
        url: str | None = None,
        collection: str = "bitesmart_bench",
        m: int = 16,
        ef_construct: int = 128,
        hnsw_ef: int = 64,
        payload_schema: Mapping[str, str] | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        super().__init__()
        self.url = url
        self.collection = collection
        self.m, self.ef_construct, self.hnsw_ef = m, ef_construct, hnsw_ef
        self.payload_schema = dict(payload_schema or {})
        self._client = client or (QdrantClient(url=url) if url else QdrantClient(":memory:"))

    @property
    def is_server(self) -> bool:
        return self.url is not None

    def _build(self, vectors: Matrix, payloads: Sequence[Mapping[str, Any]]) -> None:
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            hnsw_config=qm.HnswConfigDiff(m=self.m, ef_construct=self.ef_construct),
        )
        if self.is_server:
            for field, schema in self.payload_schema.items():
                self._client.create_payload_index(
                    self.collection, field_name=field, field_schema=schema
                )
        self._client.upload_collection(
            self.collection,
            vectors=vectors,
            payload=[dict(p) for p in payloads],
            ids=range(self.n),
            batch_size=256,
        )
        logger.info("qdrant collection built", extra={"collection": self.collection, "n": self.n})

    def search(
        self, queries: Matrix, k: int, flt: Filter | None = None
    ) -> tuple[IdMatrix, Matrix]:
        qfilter = to_qdrant_filter(flt)
        params = qm.SearchParams(hnsw_ef=self.hnsw_ef, exact=False) if self.is_server else None
        ids_out, scores_out = self.empty_result(len(queries), k)
        for row, q in enumerate(queries):
            response = self._client.query_points(
                self.collection,
                query=np.asarray(q, dtype=np.float32).tolist(),
                limit=k,
                query_filter=qfilter,
                search_params=params,
                with_payload=False,
            )
            for col, point in enumerate(response.points):
                ids_out[row, col] = int(point.id)
                scores_out[row, col] = point.score
        return ids_out, scores_out

    def set_search_params(self, hnsw_ef: int | None = None) -> None:
        if hnsw_ef is not None:
            self.hnsw_ef = int(hnsw_ef)

    def describe(self) -> dict[str, Any]:
        mode = "server" if self.is_server else "embedded-exact"
        return {"index": self.name, "mode": mode, "m": self.m, "hnsw_ef": self.hnsw_ef}

    def close(self) -> None:
        self._client.close()
