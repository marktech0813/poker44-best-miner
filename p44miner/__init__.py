"""Poker44 best-miner package.

A fast, release-robust, chunk-level behavioral bot detector for Poker44 subnet 126.

The package is intentionally split so the feature/inference code has NO bittensor
dependency and can be imported from training scripts, tests, and the miner alike:

- ``features``         chunk -> fixed-length feature vector (shared train + infer)
- ``reward``           exact reproduction of the on-chain Poker44 reward
- ``benchmark_client`` download + cache public benchmark releases
- ``dataset``          build (X, y, release-group) arrays from cached releases
- ``model``            regularized ensemble definition + rank-average blend
- ``train``            release-split CV training that optimizes the true reward
- ``infer``            artifact loading + safe batch inference (heuristic fallback)
"""

from __future__ import annotations

__version__ = "1.0.0"

from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_chunk_features,
    transform_chunks,
    vectorize,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "extract_chunk_features",
    "transform_chunks",
    "vectorize",
    "__version__",
]
