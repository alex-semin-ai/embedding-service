"""Index benchmark. recall@k against the exact filtered result, latency and memory."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import psutil
from numpy.typing import NDArray

from bitesmart_recsys.indexes import (
    Filter,
    IndexUnavailableError,
    VectorIndex,
    create_index,
)
from bitesmart_recsys.recsys.models import PAYLOAD_SCHEMA

from .dataset import BenchData, benchmark_filters

logger = logging.getLogger(__name__)


@dataclass
class IndexSpec:
    name: str
    build: dict[str, Any] = field(default_factory=dict)
    sweep: list[dict[str, Any]] = field(default_factory=lambda: [{}])


@dataclass
class Result:
    index: str
    build_params: str
    search_params: str
    filter: str
    selectivity: float
    n: int
    build_s: float
    recall_at_k: float
    p50_ms: float
    p95_ms: float
    qps: float
    index_mb: float | None
    build_rss_mb: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_specs(qdrant_url: str | None = None, qdrant_local: bool = False) -> list[IndexSpec]:
    specs = [
        IndexSpec("numpy_exact"),
        IndexSpec("faiss_flat"),
        IndexSpec(
            "faiss_hnsw",
            {"m": 32, "ef_construction": 200},
            [{"ef_search": e} for e in (16, 64, 256)],
        ),
        IndexSpec("faiss_ivf", {}, [{"nprobe": p} for p in (1, 8, 32)]),
        IndexSpec(
            "hnswlib",
            {"m": 32, "ef_construction": 200},
            [{"ef_search": e} for e in (16, 64, 256)],
        ),
    ]
    if qdrant_url:
        specs.append(
            IndexSpec(
                "qdrant",
                {"url": qdrant_url, "m": 32, "ef_construct": 200, "payload_schema": PAYLOAD_SCHEMA},
                [{"hnsw_ef": e} for e in (16, 64, 256)],
            )
        )
    elif qdrant_local:
        specs.append(IndexSpec("qdrant"))
    return specs


def recall_at_k(found: NDArray[np.int64], truth: NDArray[np.int64]) -> float:
    """Mean over queries of |found ∩ truth| / |truth|. Queries with empty truth are skipped."""
    scores = []
    for f, t in zip(found, truth, strict=True):
        wanted = {int(x) for x in t if x >= 0}
        if wanted:
            scores.append(len(wanted & {int(x) for x in f if x >= 0}) / len(wanted))
    return float(np.mean(scores)) if scores else float("nan")


def _rss_mb() -> float:
    """Process memory in MB. Windows trims the working set under pressure, so use the
    private (committed) bytes there. Elsewhere RSS is the usual measure."""
    info = psutil.Process().memory_info()
    return float(getattr(info, "private", info.rss)) / 2**20


def _isolated_build_worker(spec: IndexSpec, data_dir: str) -> float:
    """Runs in a fresh process, so the RSS growth belongs to this index alone.

    The data is read from files, not received as an argument, so no transient pickle
    buffers are alive when the baseline is taken. A tiny warm-up index loads the native
    libraries first, so their one-off cost is not charged to the index.
    """
    import pickle
    from pathlib import Path

    warm_up = create_index(spec.name, **spec.build)
    warm_up.close()
    base = Path(data_dir)
    vectors = np.load(base / "vectors.npy")
    with (base / "payloads.pkl").open("rb") as fh:
        payloads = pickle.load(fh)
    gc.collect()
    before = _rss_mb()
    index = create_index(spec.name, **spec.build)
    index.build(vectors, payloads)
    gc.collect()
    delta = _rss_mb() - before
    index.close()
    return delta


def _qdrant_resident_mb(url: str) -> float | None:
    """Resident memory of a Qdrant server, from its Prometheus endpoint."""
    import httpx

    try:
        text = httpx.get(f"{url.rstrip('/')}/metrics", timeout=10).text
    except httpx.HTTPError:
        return None
    for line in text.splitlines():
        if line.startswith("memory_resident_bytes"):
            return float(line.split()[-1]) / 2**20
    return None


def measure_build_memory(spec: IndexSpec, data: BenchData) -> float | None:
    """RSS growth in MB caused by building the index. None when it cannot be measured."""
    if spec.name == "qdrant" and spec.build.get("url"):
        return None  # measured around the real build in run_benchmark
    import multiprocessing as mp
    import pickle
    import tempfile
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    try:
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "vectors.npy", data.vectors)
            with (Path(tmp) / "payloads.pkl").open("wb") as fh:
                pickle.dump(data.payloads, fh)
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                return pool.submit(_isolated_build_worker, spec, tmp).result()
    except Exception:  # unavailable optional backend or sandboxed environment
        logger.warning(
            "isolated memory measurement failed", extra={"index": spec.name}, exc_info=True
        )
        return None


def _timed_queries(
    index: VectorIndex, queries: NDArray[np.float32], k: int, flt: Filter | None
) -> tuple[NDArray[np.int64], list[float], float]:
    for q in queries[:3]:  # warm up caches and the filter mask
        index.search(q[None, :], k, flt)
    ids = np.full((len(queries), k), -1, dtype=np.int64)
    latencies: list[float] = []
    start = time.perf_counter()
    for row, q in enumerate(queries):
        t0 = time.perf_counter()
        found, _ = index.search(q[None, :], k, flt)
        latencies.append(1000 * (time.perf_counter() - t0))
        ids[row] = found[0]
    return ids, latencies, time.perf_counter() - start


def run_benchmark(
    data: BenchData,
    specs: list[IndexSpec],
    filters: dict[str, Filter | None] | None = None,
    k: int = 10,
    progress: Callable[[str], None] | None = None,
    measure_memory: bool = True,
) -> list[Result]:
    filters = filters or benchmark_filters()
    exact = create_index("numpy_exact")
    exact.build(data.vectors, data.payloads)
    truth = {name: exact.search(data.queries, k, flt)[0] for name, flt in filters.items()}
    selectivity = {
        name: (1.0 if flt is None else float(exact.allowed_mask(flt).mean()))  # type: ignore[union-attr]
        for name, flt in filters.items()
    }

    results: list[Result] = []
    for spec in specs:
        try:
            index = create_index(spec.name, **spec.build)
        except IndexUnavailableError as exc:
            logger.warning("backend skipped", extra={"index": spec.name, "reason": str(exc)})
            continue
        rss_delta = measure_build_memory(spec, data) if measure_memory else None
        qdrant_url = spec.build.get("url") if spec.name == "qdrant" else None
        server_before = _qdrant_resident_mb(qdrant_url) if qdrant_url else None
        t0 = time.perf_counter()
        index.build(data.vectors, data.payloads)
        build_s = time.perf_counter() - t0
        if qdrant_url and server_before is not None:
            server_after = _qdrant_resident_mb(qdrant_url)
            rss_delta = None if server_after is None else server_after - server_before
        mem = index.memory_bytes()

        for params in spec.sweep:
            index.set_search_params(**params)
            for fname, flt in filters.items():
                found, lat, total = _timed_queries(index, data.queries, k, flt)
                results.append(
                    Result(
                        index=spec.name,
                        build_params=str(
                            {k_: v for k_, v in spec.build.items() if k_ != "payload_schema"}
                        ),
                        search_params=str(params),
                        filter=fname,
                        selectivity=round(selectivity[fname], 4),
                        n=data.n,
                        build_s=round(build_s, 3),
                        recall_at_k=round(recall_at_k(found, truth[fname]), 4),
                        p50_ms=round(float(np.percentile(lat, 50)), 3),
                        p95_ms=round(float(np.percentile(lat, 95)), 3),
                        qps=round(len(data.queries) / total, 1),
                        index_mb=None if mem is None else round(mem / 2**20, 2),
                        build_rss_mb=None if rss_delta is None else round(rss_delta, 1),
                    )
                )
                if progress:
                    progress(f"{spec.name} {params} {fname} recall={results[-1].recall_at_k}")
        index.close()
        del index
    return results
