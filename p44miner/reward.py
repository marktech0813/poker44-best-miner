"""Exact reproduction of the on-chain Poker44 reward (poker44/score/scoring.py).

Keeping a verbatim copy here lets the trainer optimize/measure the *true* objective
instead of a proxy. If the subnet changes its scoring, update this file to match.

IMPORTANT properties (these drive the whole modeling strategy):
* ``average_precision_score`` and ``recall@(FPR<=0.05)`` are both RANK-based. They
  depend only on the ordering of scores vs labels, NOT on absolute calibration or a
  0.5 threshold. => optimize ranking (AP), not probability calibration.
* ``human_safety_penalty`` is a constant 1.0 in the current code; there is no
  ``(1-FPR)^2`` multiplier and no "FPR>=10% zeroes the score" rule.
* Weighting is 0.75 * AP + 0.25 * recall.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score


def _recall_at_fpr(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> Tuple[float, float]:
    """Best bot recall reachable while keeping human false-positive rate bounded."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count <= 0 or negative_count <= 0 or scores.size == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(positive_count, 1)
    fpr = fp / max(negative_count, 1)

    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0

    allowed_indices = np.flatnonzero(allowed)
    best_local = int(allowed_indices[np.argmax(recall[allowed])])
    return float(recall[best_local]), float(fpr[best_local])


def reward(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Compute the Poker44 reward and its component metrics."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true)
    if y_pred.size and np.any(y_true == 1):
        ap_score = float(average_precision_score(y_true, y_pred))
    else:
        ap_score = 0.0

    bot_recall, fpr = _recall_at_fpr(y_pred, y_true, max_fpr=0.05)
    human_safety_penalty = 1.0

    base_score = 0.75 * ap_score + 0.25 * bot_recall
    rew = base_score * human_safety_penalty

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": rew,
    }
    return rew, res
