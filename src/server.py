"""
server.py - LLM Inference Server
MLOps Milestone 5

Provides:
  POST /generate   — run inference (with batching + caching)
  GET  /health     — liveness check
  GET  /stats      — real-time metrics

Architecture:
  Request → Cache lookup (hit: return immediately)
           → DynamicBatcher (groups with other concurrent requests)
           → LLM.generate_batch()
           → Cache store → Response

Concurrency: fully async via FastAPI + asyncio; no threading needed.
Race-condition safety: asyncio.Lock used in both caching.py and batching.py.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import config
from src.caching import InferenceCache
from src.batching import DynamicBatcher, BatchRequest

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model backend ─────────────────────────────────────────────────────────────

class MockLLM:
    """
    Simulates a transformer model with realistic latency scaling.
    Replace generate_batch() with a real HuggingFace pipeline for production.

    Latency model:
      - Serial single request: ~100 ms
      - Batched N requests:    ~100 + 20*(N-1) ms  (amortised compute)
    This mirrors real GPU behaviour where batching reduces per-request cost.
    """

    BASE_MS = 100.0   # base latency per batch call
    PER_REQ_MS = 20.0  # incremental cost per extra request in a batch

    async def generate_batch(self, prompts: List[str], params: dict) -> List[str]:
        n = len(prompts)
        simulated_secs = (self.BASE_MS + self.PER_REQ_MS * (n - 1)) / 1000.0
        await asyncio.sleep(simulated_secs)
        return [
            f"[Model response to: '{p[:60]}{'...' if len(p) > 60 else ''}']"
            for p in prompts
        ]


class HuggingFaceLLM:
    """
    Real HuggingFace model backend.
    Activated when MODEL_NAME env var is set to a model ID (e.g. "gpt2").
    """

    def __init__(self, model_name: str):
        from transformers import pipeline
        logger.info("Loading model: %s", model_name)
        self._pipe = pipeline("text-generation", model=model_name)
        logger.info("Model loaded.")

    async def generate_batch(self, prompts: List[str], params: dict) -> List[str]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._pipe(
                prompts,
                max_new_tokens=params.get("max_new_tokens", 50),
                do_sample=False,
            ),
        )
        return [r[0]["generated_text"] for r in results]


# ── Global singletons ─────────────────────────────────────────────────────────

_llm: Optional[object] = None
_cache: Optional[InferenceCache] = None
_batcher: Optional[DynamicBatcher] = None


# ── Batch processor (wired into DynamicBatcher) ───────────────────────────────

async def _process_batch(batch: List[BatchRequest]) -> None:
    """Called by DynamicBatcher for each dispatched batch."""
    prompts = [r.prompt for r in batch]
    params = batch[0].model_params  # uniform params per batch
    try:
        results = await _llm.generate_batch(prompts, params)
        for req, result in zip(batch, results):
            if not req.future.done():
                req.future.set_result(result)
    except Exception as exc:
        for req in batch:
            if not req.future.done():
                req.future.set_exception(exc)
        raise


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _cache, _batcher
    logger.info("Initialising inference server...")

    # Initialise model backend
    if config.model_name == "mock":
        _llm = MockLLM()
        logger.info("Using MockLLM (no GPU required)")
    else:
        _llm = HuggingFaceLLM(config.model_name)

    # Initialise cache
    _cache = InferenceCache(
        max_entries=config.cache_max_entries,
        ttl_seconds=config.cache_ttl_seconds,
    )

    # Initialise batcher
    _batcher = DynamicBatcher(
        max_batch_size=config.batch_size,
        batch_timeout_ms=config.batch_timeout_ms,
        process_batch_fn=_process_batch,
    )
    await _batcher.start()

    logger.info("Server ready on %s:%d", config.host, config.port)
    yield

    logger.info("Shutting down batcher...")
    await _batcher.stop()
    logger.info("Server stopped.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Inference Server",
    description="Dynamic batching + caching for production LLM serving",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input text for the model")
    max_new_tokens: int = Field(config.max_new_tokens, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    use_cache: bool = Field(True, description="Whether to use the response cache")


class GenerateResponse(BaseModel):
    text: str
    cached: bool
    latency_ms: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Run LLM inference with dynamic batching and caching.

    - If use_cache=True and a matching entry exists: returns immediately (sub-ms)
    - Otherwise, request is grouped with concurrent calls and processed together
    """
    t0 = time.perf_counter()
    model_params = {
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
    }

    # ── Cache lookup ──────────────────────────────────────────────────────────
    if req.use_cache:
        cached_text = await _cache.get(req.prompt, model_params)
        if cached_text is not None:
            latency = (time.perf_counter() - t0) * 1000
            return GenerateResponse(
                text=cached_text,
                cached=True,
                latency_ms=round(latency, 2),
            )

    # ── Submit to batcher ─────────────────────────────────────────────────────
    try:
        result = await _batcher.submit(req.prompt, model_params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Populate cache ────────────────────────────────────────────────────────
    if req.use_cache:
        await _cache.set(req.prompt, model_params, result)

    latency = (time.perf_counter() - t0) * 1000
    return GenerateResponse(
        text=result,
        cached=False,
        latency_ms=round(latency, 2),
    )


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    """Real-time operational metrics."""
    return {
        "cache": _cache.stats(),
        "batcher": _batcher.stats(),
        "config": {
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "batch_timeout_ms": config.batch_timeout_ms,
            "cache_ttl_seconds": config.cache_ttl_seconds,
            "cache_max_entries": config.cache_max_entries,
        },
    }


@app.post("/cache/clear")
async def clear_cache():
    """Flush the response cache (useful for benchmarking cold-cache scenarios)."""
    await _cache.clear()
    return {"message": "Cache cleared"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level=config.log_level.lower(),
    )
