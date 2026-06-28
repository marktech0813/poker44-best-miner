"""Build model-ready arrays from cached Poker44 benchmark releases.

One training example == one chunk group (~30-40 hands of a single entity) with one
label (1 = bot, 0 = human). We carry the ``sourceDate`` as a group key so we can do
release-separated cross-validation (the discriminative signal direction flips between
releases, so leaking a release across folds badly overstates performance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .benchmark_client import BenchmarkClient
from .features import FEATURE_NAMES, transform_chunks


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    release: np.ndarray              # source date per row (GroupKFold key)
    split: np.ndarray                # 'train' / 'validation' / '' per row
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def summary(self) -> str:
        n = len(self)
        pos = int(self.y.sum()) if n else 0
        releases = sorted(set(self.release.tolist()))
        return (
            f"Dataset: {n} groups | {pos} bot / {n - pos} human | "
            f"{len(releases)} releases | {self.X.shape[1]} features"
        )


def build_dataset(
    client: Optional[BenchmarkClient] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Read every cached chunk-response object and vectorize all groups."""
    if client is None:
        client = BenchmarkClient(cache_dir=cache_dir) if cache_dir else BenchmarkClient()

    rows: List[np.ndarray] = []
    labels: List[int] = []
    releases: List[str] = []
    splits: List[str] = []

    for obj in client.iter_cached():
        groups = obj.get("chunks") or []
        gt = obj.get("groundTruth")
        if gt is None:
            # fall back to string labels if numeric is absent
            gt_labels = obj.get("groundTruthLabels") or []
            gt = [1 if str(v).strip().lower() in ("bot", "ai", "1", "true") else 0 for v in gt_labels]
        if len(groups) != len(gt):
            # length mismatch -> skip to avoid mislabeling
            continue
        source_date = str(obj.get("sourceDate") or "")
        split = str(obj.get("split") or "")
        feats = transform_chunks(groups)
        for i in range(feats.shape[0]):
            rows.append(feats[i])
            labels.append(int(gt[i]))
            releases.append(source_date)
            splits.append(split)

    if not rows:
        return Dataset(
            X=np.zeros((0, len(FEATURE_NAMES)), dtype=float),
            y=np.zeros((0,), dtype=int),
            release=np.zeros((0,), dtype=object),
            split=np.zeros((0,), dtype=object),
        )

    return Dataset(
        X=np.vstack(rows).astype(float),
        y=np.asarray(labels, dtype=int),
        release=np.asarray(releases, dtype=object),
        split=np.asarray(splits, dtype=object),
    )
