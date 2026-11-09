from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from bitesmart_recsys.bench.runner import recall_at_k
from bitesmart_recsys.indexes import (
    AnyOf,
    Condition,
    Filter,
    IndexUnavailableError,
    backend_names,
    create_index,
    to_qdrant_filter,
)
from bitesmart_recsys.indexes.qdrant_index import QdrantIndex

FILTERS: dict[str, Filter | None] = {
    "none": None,
    "eq": Filter(must=(Condition("group", "eq", 3),)),
    "mixed": Filter(
        must=(Condition("nutriments.sugar", "lte", 30.0),),
        must_not=(Condition("tags", "eq", "b"),),
    ),
    "anyof": Filter(must=(AnyOf((Condition("group", "eq", 1), Condition("group", "eq", 2))),)),
    "empty_result": Filter(must=(Condition("group", "eq", 999),)),
}

# (backend, build kwargs, search params, minimum recall on every filter)
BACKENDS: list[tuple[str, dict[str, Any], dict[str, Any], float]] = [
    ("faiss_flat", {}, {}, 1.0),
    # In-search filtering costs HNSW recall when the filter is selective and aligned with
    # the clusters, even at a large ef. This is the effect the benchmark quantifies.
    ("faiss_hnsw", {"m": 32, "ef_construction": 200}, {"ef_search": 512}, 0.7),
    ("faiss_ivf", {"nlist": 16}, {"nprobe": 16}, 0.99),
    ("qdrant", {}, {}, 1.0),
    ("hnswlib", {"m": 32, "ef_construction": 200}, {"ef_search": 512}, 0.85),
]


@pytest.fixture(scope="module")
def exact(random_data: Any) -> Any:
    vectors, payloads, _ = random_data
    index = create_index("numpy_exact")
    index.build(vectors, payloads)
    return index


@pytest.mark.parametrize("filter_name", list(FILTERS))
@pytest.mark.parametrize(("backend", "build_kw", "search_kw", "min_recall"), BACKENDS)
def test_backend_matches_exact_search(
    backend: str,
    build_kw: dict[str, Any],
    search_kw: dict[str, Any],
    min_recall: float,
    filter_name: str,
    random_data: Any,
    exact: Any,
) -> None:
    vectors, payloads, queries = random_data
    try:
        index = create_index(backend, **build_kw)
    except IndexUnavailableError:
        pytest.skip(f"{backend} is not installed")
    index.build(vectors, payloads)
    index.set_search_params(**search_kw)
    flt = FILTERS[filter_name]

    ids, scores = index.search(queries, 10, flt)
    truth, _ = exact.search(queries, 10, flt)

    assert ids.shape == (len(queries), 10)
    mask = exact.allowed_mask(flt)
    returned = ids[ids >= 0]
    if mask is not None:  # a filter must never leak a non matching point
        assert mask[returned].all()
    if filter_name == "empty_result":
        assert (ids == -1).all()
        assert np.isneginf(scores).all()
    else:
        assert recall_at_k(ids, truth) >= min_recall
    index.close()


def test_exact_scores_sorted_and_padded(random_data: Any) -> None:
    vectors, payloads, queries = random_data
    index = create_index("numpy_exact")
    index.build(vectors[:5], payloads[:5])
    ids, scores = index.search(queries[:2], 8)  # k larger than n
    assert (ids[:, 5:] == -1).all()
    assert (np.diff(scores[:, :5], axis=1) <= 1e-6).all()


def test_build_validates_lengths(random_data: Any) -> None:
    vectors, payloads, _ = random_data
    with pytest.raises(ValueError, match="same length"):
        create_index("numpy_exact").build(vectors, payloads[:-1])


def test_registry_errors_and_unknown_params() -> None:
    assert {"numpy_exact", "faiss_hnsw", "qdrant", "hnswlib"} <= set(backend_names())
    with pytest.raises(ValueError, match="unknown index"):
        create_index("nope")
    with pytest.raises(TypeError):
        create_index("faiss_flat").set_search_params(ef_search=1)


def test_describe_and_memory(random_data: Any) -> None:
    vectors, payloads, _ = random_data
    hnsw = create_index("faiss_hnsw", m=16)
    hnsw.build(vectors, payloads)
    flat = create_index("faiss_flat")
    flat.build(vectors, payloads)
    assert hnsw.describe()["M"] == 16
    assert hnsw.memory_bytes() > flat.memory_bytes() >= vectors.nbytes


def test_qdrant_filter_translation() -> None:
    flt = Filter(
        must=(
            Condition("a", "eq", "x"),
            Condition("b", "any", ("p", "q")),
            Condition("c", "lte", 2.5),
            AnyOf((Condition("d", "eq", 1), Condition("d", "eq", 2))),
        ),
        must_not=(Condition("e", "gte", 1.0),),
    )
    q = to_qdrant_filter(flt)
    assert q is not None
    assert q.must is not None
    assert len(q.must) == 4
    assert q.must_not is not None
    assert len(q.must_not) == 1
    assert to_qdrant_filter(None) is None
    assert to_qdrant_filter(Filter()) is None


def test_qdrant_server_mode_creates_payload_indexes(random_data: Any) -> None:
    """The real server is mocked, we check the calls made against its client API."""
    vectors, payloads, queries = random_data
    client = MagicMock()
    client.collection_exists.return_value = True
    response = MagicMock()
    point = MagicMock(id=7, score=0.5)
    response.points = [point]
    client.query_points.return_value = response

    index = QdrantIndex(
        url="http://qdrant:6333", payload_schema={"group": "integer"}, client=client, m=24
    )
    index.build(vectors[:10], payloads[:10])
    client.delete_collection.assert_called_once()
    client.create_payload_index.assert_called_once_with(
        "bitesmart_bench", field_name="group", field_schema="integer"
    )
    ids, scores = index.search(queries[:1], 3, Filter(must=(Condition("group", "eq", 1),)))
    assert ids[0].tolist() == [7, -1, -1]
    assert scores[0, 0] == pytest.approx(0.5)
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["search_params"].hnsw_ef == 64
    assert index.describe()["mode"] == "server"
