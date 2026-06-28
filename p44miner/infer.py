"""Artifact loading + safe batch inference for the live miner.

Guarantees the miner contract no matter what:
* returns exactly one score per chunk, each clamped to [0, 1], in input order;
* never raises out of ``score_chunks`` (errors degrade to a heuristic fallback);
* loads the model once at construction (no disk/network in the hot path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_chunk_features,
    transform_chunks,
)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _heuristic_risk(chunk: Sequence[Dict[str, Any]]) -> float:
    """Untrained, deterministic fallback ranking in [0, 1].

    This exists only so a freshly-deployed miner (no artifact yet) still returns a
    sane, ordered response. It is NOT competitive; train a model to replace it.
    """
    feats = extract_chunk_features(chunk)
    # Combine a few magnitude signals; direction is unknown so this is intentionally
    # mild and centered near 0.5.
    raw = (
        0.5
        + 0.15 * (feats.get("action_entropy", 0.0) - 1.0)
        + 0.10 * (feats.get("aggression_rate", 0.0) - 0.3)
        - 0.10 * (feats.get("zero_amount_share", 0.0) - 0.3)
    )
    return float(max(0.0, min(1.0, raw)))


class Predictor:
    def __init__(self, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.ready = False
        self.reason = ""
        self._load()

    def _load(self) -> None:
        model_path = self.artifact_dir / "model.joblib"
        if not model_path.exists():
            self.reason = f"no artifact at {model_path}; using heuristic fallback"
            return
        try:
            import joblib

            model = joblib.load(model_path)
            schema_ok = (
                getattr(model, "feature_schema_version", None) == FEATURE_SCHEMA_VERSION
                and list(getattr(model, "feature_names", [])) == list(FEATURE_NAMES)
            )
            if not schema_ok:
                self.reason = "feature schema mismatch; using heuristic fallback"
                return
            self.model = model
            self.ready = True
            self.reason = "model loaded"
        except Exception as exc:  # noqa: BLE001 - inference must never crash
            self.reason = f"failed to load artifact ({exc}); using heuristic fallback"

    def score_chunks(self, chunks: Optional[Sequence[Sequence[Dict[str, Any]]]]) -> List[float]:
        chunks = list(chunks or [])
        if not chunks:
            return []
        if self.ready and self.model is not None:
            try:
                X = transform_chunks(chunks)
                risk = self.model.predict_risk(X)
                return [float(max(0.0, min(1.0, s))) for s in risk]
            except Exception:  # noqa: BLE001 - fall back rather than fail the request
                pass
        return [_heuristic_risk(chunk) for chunk in chunks]
