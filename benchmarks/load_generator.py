"""
load_generator.py - Synthetic Load Generator
MLOps Milestone 5

Sends concurrent HTTP requests to the inference server at a configurable rate.
Supports a repeat_ratio to simulate cache-friendly workloads.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp

# Representative prompts covering a range of lengths and topics
SAMPLE_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "What is the capital of France?",
    "Write a short poem about machine learning.",
    "How does photosynthesis work?",
    "What are the main causes of climate change?",
    "Describe the process of making sourdough bread.",
    "What is gradient descent in neural networks?",
    "Explain what a transformer model is.",
    "How do mRNA vaccines work?",
    "What is the difference between RAM and ROM?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the Pythagorean theorem?",
    "How does TCP/IP work?",
    "What are the benefits of containerisation with Docker?",
    "Describe the water cycle.",
]


@dataclass
class RequestResult:
    prompt: str
    latency_ms: float
    cached: bool
    success: bool
    error: str = ""


async def _single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    use_cache: bool = True,
    timeout_secs: float = 30.0,
) -> RequestResult:
    payload = {
        "prompt": prompt,
        "use_cache": use_cache,
        "max_new_tokens": 50,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{base_url}/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_secs),
        ) as resp:
            data = await resp.json()
            latency = (time.perf_counter() - t0) * 1000
            return RequestResult(
                prompt=prompt,
                latency_ms=data.get("latency_ms", round(latency, 2)),
                cached=data.get("cached", False),
                success=resp.status == 200,
                error="" if resp.status == 200 else str(data),
            )
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return RequestResult(
            prompt=prompt,
            latency_ms=round(latency, 2),
            cached=False,
            success=False,
            error=str(exc),
        )


async def run_load(
    base_url: str,
    requests_per_second: float,
    duration_seconds: float,
    repeat_ratio: float = 0.3,
    use_cache: bool = True,
) -> List[RequestResult]:
    """
    Send synthetic traffic to the inference server.

    Args:
        base_url:             Server base URL (e.g. "http://localhost:8000")
        requests_per_second:  Target arrival rate
        duration_seconds:     How long to run the load
        repeat_ratio:         Fraction of requests using a repeated prompt
                              (increases cache hit rate; 0 = all unique)
        use_cache:            Whether to enable caching on requests

    Returns:
        List of RequestResult objects for analysis.
    """
    total_requests = max(1, int(requests_per_second * duration_seconds))
    interval_secs = 1.0 / requests_per_second
    repeatable = SAMPLE_PROMPTS[:5]  # subset used for repeated prompts

    connector = aiohttp.TCPConnector(limit=0)  # no connection cap
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _ in range(total_requests):
            if random.random() < repeat_ratio and repeatable:
                prompt = random.choice(repeatable)
            else:
                prompt = random.choice(SAMPLE_PROMPTS)

            task = asyncio.create_task(
                _single_request(session, base_url, prompt, use_cache)
            )
            tasks.append(task)
            await asyncio.sleep(interval_secs)

        results = await asyncio.gather(*tasks)

    return list(results)


# Allow running standalone for quick smoke-tests
if __name__ == "__main__":
    import json

    async def _smoke_test():
        print("Sending 10 requests at 2 req/s ...")
        results = await run_load("http://localhost:8000", 2, 5)
        success = sum(1 for r in results if r.success)
        cached = sum(1 for r in results if r.cached)
        avg_lat = sum(r.latency_ms for r in results if r.success) / max(success, 1)
        print(f"Success: {success}/{len(results)}  Cached: {cached}  Avg latency: {avg_lat:.1f}ms")

    asyncio.run(_smoke_test())
