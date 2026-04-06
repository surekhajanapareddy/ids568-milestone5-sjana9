#!/usr/bin/env python3
"""
run_benchmarks.py - Benchmark Orchestration
MLOps Milestone 5

Runs four benchmark scenarios and saves results + charts:
  1. Cold-cache vs warm-cache latency
  2. Load levels: low / medium / high throughput
  3. Single (serial) vs concurrent (batched) request latency
  4. Cache hit-rate over time

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --base-url http://localhost:8000
    python benchmarks/run_benchmarks.py --help
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List

import aiohttp

# Make project root importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.load_generator import run_load, RequestResult, _single_request

RESULTS_DIR = Path(__file__).parent / "results"
VIZ_DIR = PROJECT_ROOT / "analysis" / "visualizations"
BASE_URL_DEFAULT = "http://localhost:8000"


# ── Statistics helpers ────────────────────────────────────────────────────────

def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = max(0, int(len(sorted_data) * p / 100) - 1)
    return sorted_data[idx]


def summarize(results: List[RequestResult]) -> dict:
    successful = [r for r in results if r.success]
    latencies = [r.latency_ms for r in successful]
    cached = [r for r in successful if r.cached]

    if not latencies:
        return {"error": "No successful requests", "total_requests": len(results)}

    total_wall_secs = max(latencies) / 1000.0 + 0.001

    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "cached_count": len(cached),
        "cache_hit_rate": round(len(cached) / len(successful), 4),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_median_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "latency_p99_ms": round(percentile(latencies, 99), 2),
        "throughput_rps": round(len(successful) / total_wall_secs, 2),
    }


# ── Benchmark scenarios ───────────────────────────────────────────────────────

async def bench_cold_vs_warm(base_url: str) -> dict:
    """Compare first-request (cold) vs repeated-request (warm) cache performance."""
    print("  [1/4] Cold cache vs warm cache...")

    # Flush cache to ensure cold start
    async with aiohttp.ClientSession() as s:
        await s.post(f"{base_url}/cache/clear")

    cold = await run_load(base_url, requests_per_second=5, duration_seconds=8,
                          repeat_ratio=0.0, use_cache=True)

    # Warm: same prompts repeated, cache should hit
    warm = await run_load(base_url, requests_per_second=5, duration_seconds=8,
                          repeat_ratio=1.0, use_cache=True)

    return {"cold_cache": summarize(cold), "warm_cache": summarize(warm)}


async def bench_load_levels(base_url: str) -> dict:
    """Measure throughput and latency under low / medium / high load."""
    print("  [2/4] Load level sweep (low → medium → high)...")
    levels = [
        ("low",    10,  8),
        ("medium", 50,  8),
        ("high",  100,  8),
    ]
    out = {}
    for name, rps, dur in levels:
        print(f"         {name}: {rps} req/s for {dur}s ...")
        r = await run_load(base_url, rps, dur, repeat_ratio=0.3)
        out[name] = summarize(r)
    return out


async def bench_single_vs_batched(base_url: str) -> dict:
    """Serial single requests vs concurrent requests (batched by server)."""
    print("  [3/4] Single serial vs concurrent batched latency...")

    async with aiohttp.ClientSession() as session:
        # Flush cache so results reflect raw inference
        await session.post(f"{base_url}/cache/clear")

        # Serial single requests
        single_latencies = []
        for i in range(20):
            r = await _single_request(session, base_url, f"Serial prompt #{i}", use_cache=False)
            if r.success:
                single_latencies.append(r.latency_ms)

        # Concurrent requests — server groups these into batches
        await session.post(f"{base_url}/cache/clear")
        tasks = [
            asyncio.create_task(
                _single_request(session, base_url, f"Concurrent prompt #{i}", use_cache=False)
            )
            for i in range(20)
        ]
        concurrent_results = await asyncio.gather(*tasks)
        conc_latencies = [r.latency_ms for r in concurrent_results if r.success]

    return {
        "single_serial": {
            "n": len(single_latencies),
            "mean_ms": round(statistics.mean(single_latencies), 2) if single_latencies else 0,
            "p95_ms": round(percentile(single_latencies, 95), 2) if single_latencies else 0,
        },
        "concurrent_batched": {
            "n": len(conc_latencies),
            "mean_ms": round(statistics.mean(conc_latencies), 2) if conc_latencies else 0,
            "p95_ms": round(percentile(conc_latencies, 95), 2) if conc_latencies else 0,
        },
    }


async def bench_cache_hit_rate_over_time(base_url: str) -> dict:
    """Track cache hit rate as the cache warms up over successive requests."""
    print("  [4/4] Cache hit-rate over time (cache warm-up)...")

    async with aiohttp.ClientSession() as session:
        await session.post(f"{base_url}/cache/clear")

    snapshots = []
    for window in range(10):
        r = await run_load(base_url, requests_per_second=5, duration_seconds=3,
                           repeat_ratio=0.5, use_cache=True)
        s = summarize(r)
        snapshots.append({
            "window": window + 1,
            "hit_rate": s.get("cache_hit_rate", 0),
            "mean_latency_ms": s.get("latency_mean_ms", 0),
        })

    return {"snapshots": snapshots}


# ── Visualisations ────────────────────────────────────────────────────────────

def generate_charts(results: dict) -> None:
    """Generate matplotlib charts and save to analysis/visualizations/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [charts] matplotlib not available — skipping chart generation")
        return

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chart 1: Cold vs Warm latency ─────────────────────────────────────────
    cvw = results.get("cold_vs_warm", {})
    if cvw:
        labels = ["Cold Cache", "Warm Cache"]
        means  = [cvw["cold_cache"].get("latency_mean_ms", 0),
                  cvw["warm_cache"].get("latency_mean_ms", 0)]
        p95s   = [cvw["cold_cache"].get("latency_p95_ms", 0),
                  cvw["warm_cache"].get("latency_p95_ms", 0)]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([i - 0.2 for i in x], means, 0.35, label="Mean", color="#4C72B0")
        ax.bar([i + 0.2 for i in x], p95s,  0.35, label="P95",  color="#DD8452")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylabel("Latency (ms)"); ax.set_title("Cold vs Warm Cache Latency")
        ax.legend(); fig.tight_layout()
        fig.savefig(VIZ_DIR / "cold_vs_warm_latency.png", dpi=150)
        plt.close(fig)

    # ── Chart 2: Throughput by load level ────────────────────────────────────
    ll = results.get("load_levels", {})
    if ll:
        names  = list(ll.keys())
        thrput = [ll[n].get("throughput_rps", 0) for n in names]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, thrput, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_ylabel("Throughput (req/s)"); ax.set_title("Throughput by Load Level")
        fig.tight_layout()
        fig.savefig(VIZ_DIR / "throughput_by_load.png", dpi=150)
        plt.close(fig)

    # ── Chart 3: Single vs Batched ───────────────────────────────────────────
    svb = results.get("single_vs_batched", {})
    if svb:
        labels = ["Serial Single", "Concurrent Batched"]
        means  = [svb["single_serial"].get("mean_ms", 0),
                  svb["concurrent_batched"].get("mean_ms", 0)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, color=["#4C72B0", "#55A868"])
        ax.set_ylabel("Mean Latency (ms)"); ax.set_title("Serial vs Batched Request Latency")
        fig.tight_layout()
        fig.savefig(VIZ_DIR / "single_vs_batched.png", dpi=150)
        plt.close(fig)

    # ── Chart 4: Cache hit-rate over time ────────────────────────────────────
    chr_ = results.get("cache_hit_rate_over_time", {})
    if chr_ and chr_.get("snapshots"):
        snaps = chr_["snapshots"]
        windows   = [s["window"] for s in snaps]
        hit_rates = [s["hit_rate"] * 100 for s in snaps]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(windows, hit_rates, marker="o", color="#4C72B0")
        ax.set_xlabel("Time Window"); ax.set_ylabel("Cache Hit Rate (%)")
        ax.set_title("Cache Hit Rate Over Time (Warm-Up)")
        ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(VIZ_DIR / "cache_hit_rate_over_time.png", dpi=150)
        plt.close(fig)

    print(f"  [charts] Saved to {VIZ_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(base_url: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Quick connectivity check
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                assert r.status == 200, f"Health check failed: {r.status}"
    except Exception as exc:
        print(f"ERROR: Cannot reach server at {base_url}: {exc}")
        print("Start the server first:  python -m src.server")
        sys.exit(1)

    print(f"Running benchmarks against {base_url}\n")
    results = {}

    results["cold_vs_warm"]          = await bench_cold_vs_warm(base_url)
    results["load_levels"]           = await bench_load_levels(base_url)
    results["single_vs_batched"]     = await bench_single_vs_batched(base_url)
    results["cache_hit_rate_over_time"] = await bench_cache_hit_rate_over_time(base_url)
    results["metadata"] = {
        "base_url": base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save raw results
    out_file = RESULTS_DIR / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {out_file}")

    # Generate charts
    generate_charts(results)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 5 Benchmark Suite")
    parser.add_argument(
        "--base-url",
        default=BASE_URL_DEFAULT,
        help=f"Inference server URL (default: {BASE_URL_DEFAULT})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.base_url))
