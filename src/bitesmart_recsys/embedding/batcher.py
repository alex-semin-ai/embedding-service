"""Dynamic batching. Concurrent requests are merged into one model call.

A worker waits for the first pending text, then keeps collecting more until either
`max_batch_size` texts are queued or `max_wait_ms` has passed since the first one.
This trades a few milliseconds of latency for much higher throughput under load.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .base import Embedder, Matrix

logger = logging.getLogger(__name__)


@dataclass
class BatcherStats:
    items: int = 0
    batches: int = 0
    max_batch: int = 0
    total_queue_wait_s: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def mean_batch(self) -> float:
        return self.items / self.batches if self.batches else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "items": self.items,
            "batches": self.batches,
            "max_batch": self.max_batch,
            "mean_batch": round(self.mean_batch, 3),
            "mean_queue_wait_ms": round(1000 * self.total_queue_wait_s / max(self.items, 1), 3),
        }


@dataclass
class _Pending:
    text: str
    future: asyncio.Future[np.ndarray]
    enqueued: float


class QueueFullError(RuntimeError):
    """Raised when the pending queue is full, so the API can answer 429."""


class DynamicBatcher:
    def __init__(
        self,
        embedder: Embedder,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
        max_queue_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.embedder = embedder
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.max_queue_size = max_queue_size
        self.stats = BatcherStats()
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="dynamic-batcher")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        self._executor.shutdown(wait=False)

    async def submit(self, texts: list[str]) -> Matrix:
        """Embed `texts`, returning rows in input order."""
        if not texts:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        if self._worker is None:
            raise RuntimeError("batcher is not started")
        if self._queue.qsize() + len(texts) > self.max_queue_size:
            raise QueueFullError("embedding queue is full")
        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        pending = [_Pending(t, loop.create_future(), now) for t in texts]
        for item in pending:
            self._queue.put_nowait(item)
        rows = await asyncio.gather(*(p.future for p in pending))
        return np.stack(rows).astype(np.float32)

    async def _collect(self) -> list[_Pending]:
        batch = [await self._queue.get()]
        deadline = time.perf_counter() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            live = [p for p in batch if not p.future.cancelled()]
            if not live:
                continue
            started = time.perf_counter()
            try:
                vectors = await loop.run_in_executor(
                    self._executor, self.embedder.embed, [p.text for p in live]
                )
            except Exception as exc:
                logger.exception("embedding batch failed", extra={"batch_size": len(live)})
                for p in live:
                    if not p.future.done():
                        p.future.set_exception(exc)
                continue
            for p, vec in zip(live, vectors, strict=True):
                if not p.future.done():
                    p.future.set_result(vec)
            self._record(live, started)

    def _record(self, live: list[_Pending], started: float) -> None:
        s = self.stats
        s.items += len(live)
        s.batches += 1
        s.max_batch = max(s.max_batch, len(live))
        s.batch_sizes.append(len(live))
        s.total_queue_wait_s += sum(started - p.enqueued for p in live)
        logger.debug(
            "batch processed",
            extra={
                "batch_size": len(live),
                "model_ms": round(1000 * (time.perf_counter() - started), 2),
            },
        )
