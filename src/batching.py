"""
batching.py - Dynamic Request Batcher
MLOps Milestone 5

Strategy: Hybrid batching — dispatch when EITHER condition is met:
  1. Accumulated batch reaches max_batch_size, OR
  2. batch_timeout_ms milliseconds have elapsed since the first queued request

This balances throughput (larger batches) against latency (not waiting too long).

Concurrency: asyncio.Lock protects the shared queue; asyncio.Event signals the
worker to process immediately when the batch is full.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single inference request waiting to be processed."""

    prompt: str
    model_params: dict
    # Resolved by the batch worker when inference completes
    future: asyncio.Future = field(default_factory=asyncio.get_event_loop().create_future)


class DynamicBatcher:
    """
    Collects incoming requests into batches and dispatches them to a
    user-supplied async processing function.

    Usage:
        async def my_processor(batch: List[BatchRequest]) -> None:
            results = await model.generate([r.prompt for r in batch])
            for req, res in zip(batch, results):
                req.future.set_result(res)

        batcher = DynamicBatcher(8, 50, my_processor)
        await batcher.start()
        result = await batcher.submit("Hello world", {})
        await batcher.stop()
    """

    def __init__(
        self,
        max_batch_size: int,
        batch_timeout_ms: float,
        process_batch_fn: Callable[[List[BatchRequest]], Awaitable[None]],
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.process_batch_fn = process_batch_fn

        self._queue: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        # Signalled when the queue reaches max_batch_size (bypass timeout)
        self._full_event = asyncio.Event()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Observability counters
        self._total_batches = 0
        self._total_requests = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch the background batch-worker coroutine."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(
            "DynamicBatcher started  max_size=%d  timeout=%.0fms",
            self.max_batch_size,
            self.batch_timeout_ms,
        )

    async def stop(self) -> None:
        """Gracefully shut down: drain remaining queue then cancel worker."""
        self._running = False
        self._full_event.set()  # wake worker for final drain
        if self._worker_task:
            await self._worker_task

    # ── Public submit API ─────────────────────────────────────────────────────

    async def submit(self, prompt: str, model_params: dict) -> str:
        """
        Enqueue a request and await its result.
        Returns the model output string.
        """
        loop = asyncio.get_running_loop()
        req = BatchRequest(
            prompt=prompt,
            model_params=model_params,
            future=loop.create_future(),
        )

        async with self._lock:
            self._queue.append(req)
            self._total_requests += 1
            queue_len = len(self._queue)

        logger.debug("Enqueued request  queue_depth=%d", queue_len)

        if queue_len >= self.max_batch_size:
            self._full_event.set()

        return await req.future

    # ── Worker loop ───────────────────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        timeout_secs = self.batch_timeout_ms / 1000.0
        while self._running:
            try:
                await asyncio.wait_for(self._full_event.wait(), timeout=timeout_secs)
            except asyncio.TimeoutError:
                pass  # Timeout reached — process whatever is in the queue
            finally:
                self._full_event.clear()

            async with self._lock:
                if not self._queue:
                    continue
                batch = self._queue[: self.max_batch_size]
                self._queue = self._queue[self.max_batch_size :]

            self._total_batches += 1
            batch_num = self._total_batches
            batch_size = len(batch)
            logger.debug(
                "Dispatching batch #%d  size=%d", batch_num, batch_size
            )

            t0 = time.perf_counter()
            try:
                await self.process_batch_fn(batch)
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch #%d failed: %s", batch_num, exc)
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "Batch #%d completed in %.1fms  avg=%.1fms/req",
                batch_num,
                elapsed_ms,
                elapsed_ms / batch_size,
            )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        avg = (
            self._total_requests / self._total_batches
            if self._total_batches > 0
            else 0.0
        )
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": round(avg, 2),
            "queue_depth": len(self._queue),
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
        }
