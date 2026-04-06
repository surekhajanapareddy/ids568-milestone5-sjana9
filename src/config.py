"""
config.py - Configuration Management
MLOps Milestone 5 - LLM Inference Server

All settings can be overridden via environment variables.
Example: BATCH_SIZE=16 python -m src.server
"""

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    # ── Batching ──────────────────────────────────────────────────────────────
    # Max requests to group into one inference call
    batch_size: int = int(os.getenv("BATCH_SIZE", "8"))
    # Max milliseconds to wait before processing an incomplete batch
    batch_timeout_ms: float = float(os.getenv("BATCH_TIMEOUT_MS", "50"))

    # ── Caching ───────────────────────────────────────────────────────────────
    # Seconds before a cache entry expires (TTL = time-to-live)
    cache_ttl_seconds: float = float(os.getenv("CACHE_TTL_SECONDS", "300"))
    # Maximum number of entries held in memory
    cache_max_entries: int = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))

    # ── Model ─────────────────────────────────────────────────────────────────
    # Set to a HuggingFace model ID (e.g. "gpt2") to use a real model
    model_name: str = os.getenv("MODEL_NAME", "mock")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "50"))

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton instance imported by other modules
config = ServerConfig()
