"""Throughput of the embedding path with and without dynamic batching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bitesmart_recsys.embedding import DynamicBatcher, Embedder


@dataclass
class LoadResult:
    max_batch_size: int
    max_wait_ms: float
    concurrency: int
    requests: int
    seconds: float
    requests_per_s: float
    p50_ms: float
    p95_ms: float
    mean_batch: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_load(
    embedder: Embedder,
    texts: list[str],
    concurrency: int,
    requests: int,
    max_batch_size: int,
    max_wait_ms: float,
) -> LoadResult:
    batcher = DynamicBatcher(embedder, max_batch_size, max_wait_ms, max_queue_size=100_000)
    await batcher.start()
    latencies: list[float] = []
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        async with sem:
            t0 = time.perf_counter()
            await batcher.submit([texts[i % len(texts)]])
            latencies.append(1000 * (time.perf_counter() - t0))

    await batcher.submit([texts[0]])  # warm up the model
    batcher.stats.items = batcher.stats.batches = 0
    start = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(requests)))
    seconds = time.perf_counter() - start
    await batcher.stop()
    return LoadResult(
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        concurrency=concurrency,
        requests=requests,
        seconds=round(seconds, 3),
        requests_per_s=round(requests / seconds, 1),
        p50_ms=round(float(np.percentile(latencies, 50)), 2),
        p95_ms=round(float(np.percentile(latencies, 95)), 2),
        mean_batch=round(batcher.stats.mean_batch, 2),
    )


async def compare_batching(
    embedder: Embedder, texts: list[str], concurrency: int = 32, requests: int = 256
) -> list[LoadResult]:
    """No batching (batch size 1) against growing batch sizes, same offered load."""
    return [
        await run_load(embedder, texts, concurrency, requests, size, wait)
        for size, wait in ((1, 0.0), (8, 5.0), (32, 10.0))
    ]
