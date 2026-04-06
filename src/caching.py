"""
caching.py - Privacy-Preserving Inference Cache
MLOps Milestone 5

Design decisions:
  - Keys are SHA-256 hashes of (prompt + model_params) — no plaintext user IDs
  - LRU eviction when max_entries is reached
  - TTL expiration checked on every read
  - asyncio.Lock guards all mutations to prevent race conditions
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class CacheEntry:
    """Wraps a cached value with an expiry timestamp."""

    def __init__(self, value: str, ttl: float):
        self.value = value
        self.expires_at: float = time.monotonic() + ttl

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class InferenceCache:
    """
    Async-safe, in-process LRU cache with configurable TTL and capacity.

    Privacy guarantee: cache keys are derived only from the hash of the
    prompt and model parameters. No user identifiers are ever stored.
    """

    def __init__(self, max_entries: int, ttl_seconds: float):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # Stats counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_key(self, prompt: str, model_params: dict) -> str:
        """
        Derive a privacy-preserving cache key.
        Uses SHA-256(prompt + sorted model_params JSON).
        Never includes user IDs, emails, or other PII.
        """
        payload = json.dumps(
            {"prompt": prompt, "params": model_params}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(self, prompt: str, model_params: dict) -> Optional[str]:
        """Return cached value, or None on miss / expiry."""
        key = self._make_key(prompt, model_params)
        async with self._lock:
            if key not in self._store:
                self._misses += 1
                logger.debug("Cache MISS  key=%.8s", key)
                return None

            entry = self._store[key]
            if entry.is_expired():
                del self._store[key]
                self._misses += 1
                logger.debug("Cache EXPIRED key=%.8s", key)
                return None

            # Promote to most-recently-used position
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Cache HIT   key=%.8s", key)
            return entry.value

    async def set(self, prompt: str, model_params: dict, value: str) -> None:
        """Store a value in the cache, evicting LRU entries if at capacity."""
        key = self._make_key(prompt, model_params)
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(value, self.ttl_seconds)

            # Evict oldest (least-recently-used) entries
            while len(self._store) > self.max_entries:
                oldest_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("Cache EVICT key=%.8s", oldest_key)

    async def invalidate(self, prompt: str, model_params: dict) -> None:
        """Explicitly remove a single entry."""
        key = self._make_key(prompt, model_params)
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        """Flush all entries and reset stats."""
        async with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict:
        """Return a snapshot of cache health metrics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._store),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(hit_rate, 4),
        }
