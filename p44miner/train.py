"""Release-split training for the Poker44 chunk-level bot detector.

Pipeline:
    1. (optional) download + cache all public benchmark releases
    2. build (X, y, release) arrays
    3. release-separated CV (GroupKFold by sourceDate) -> out-of-fold predictions
    4. pick blend weights that maximize the TRUE per-release reward on OOF preds
    5. report per-release + overall metrics
    6. refit base learners on all data and export the artifact

Artifacts written to ``artifacts/``:
    model.joblib          - the BlendModel (estimators + weights + schema)
    feature_schema.json    - ordered feature names + schema version
    metrics.json           - CV metrics for regression tracking
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .benchmark_client import BenchmarkClient
from .dataset import Dataset, build_dataset
from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .model import BlendModel, _proba1, build_base_models
from .reward import reward

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _mean_per_release_reward(
    scores: np.ndarray, y: np.ndarray, release: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """Average the true reward computed independently per release (mirrors live windows)."""
    per: Dict[str, float] = {}
    for r in sorted(set(release.tolist())):
        mask = release == r
        if mask.sum() < 2 or len(set(y[mask].tolist())) < 2:
            continue
        rew, _ = reward(scores[mask], y[mask])
        per[str(r)] = rew
    overall = float(np.mean(list(per.values()))) if per else 0.0
    return overall, per


def _oof_predictions(ds: Dataset, n_splits: int, seed: int) -> Dict[str, np.ndarray]:
    """Out-of-fold positive-class probabilities for each base model."""
    base = build_base_models(random_state=seed)
    oof: Dict[str, np.ndarray] = {name: np.zeros(len(ds), dtype=float) for name in base}

    unique_releases = sorted(set(ds.release.tolist()))
    n_splits = max(2, min(n_splits, len(unique_releases)))
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr, te) in enumerate(gkf.split(ds.X, ds.y, groups=ds.release), start=1):
        if len(set(ds.y[tr].tolist())) < 2:
            continue
        for name, est in build_base_models(random_state=seed).items():
            model = clone(est)
            model.fit(ds.X[tr], ds.y[tr])
            oof[name][te] = _proba1(model, ds.X[te])
        print(f"  [cv] fold {fold}/{n_splits} done ({len(tr)} train / {len(te)} test rows)")
    return oof


def _search_blend_weights(
    oof: Dict[str, np.ndarray], ds: Dataset
) -> Tuple[Dict[str, float], float]:
    """Grid-search simplex weights to maximize mean per-release OOF reward."""
    names = list(oof.keys())
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_w: Dict[str, float] = {n: 1.0 / len(names) for n in names}
    best_score = -1.0

    for combo in itertools.product(grid, repeat=len(names)):
        if sum(combo) <= 0:
            continue
        total = sum(combo)
        weights = {n: combo[i] / total for i, n in enumerate(names)}
        blended = np.zeros(len(ds), dtype=float)
        for n in names:
            blended += weights[n] * oof[n]
        score, _ = _mean_per_release_reward(blended, ds.y, ds.release)
        if score > best_score:
            best_score = score
            best_w = weights
    return best_w, best_score


def train(
    cache_dir: str | None = None,
    download: bool = False,
    max_releases: int = 400,
    n_splits: int = 5,
    seed: int = 17,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Dict[str, object]:
    client = BenchmarkClient(cache_dir=cache_dir) if cache_dir else BenchmarkClient()
    if download:
        client.download_all(max_releases=max_releases)

    ds = build_dataset(client=client)
    print(ds.summary())
    if len(ds) < 20:
        raise SystemExit(
            "Not enough cached data to train. Run with --download first, e.g.\n"
            "  python -m p44miner.train --download"
        )

    n_releases = len(set(ds.release.tolist()))
    base_cv: Dict[str, float] = {}
    if n_releases < 2:
        print(
            f"[train] WARNING: only {n_releases} release(s) cached; cannot do "
            "release-separated CV. Using equal blend weights. Run --download to fetch more."
        )
        weights = {n: 1.0 for n in build_base_models(random_state=seed)}
        blend_cv = 0.0
        oof = {n: np.zeros(len(ds), dtype=float) for n in weights}
    else:
        # ---- per-base-model OOF + CV reward ----
        print("[train] running release-separated cross-validation...")
        oof = _oof_predictions(ds, n_splits=n_splits, seed=seed)
        for name, preds in oof.items():
            score, _ = _mean_per_release_reward(preds, ds.y, ds.release)
            base_cv[name] = score
            print(f"  [cv] {name:8s} mean per-release reward = {score:.4f}")

        # ---- blend weight selection ----
        weights, blend_cv = _search_blend_weights(oof, ds)
    print(f"[train] best blend weights: { {k: round(v, 3) for k, v in weights.items()} }")
    print(f"[train] blended mean per-release reward (OOF) = {blend_cv:.4f}")

    blended_oof = np.zeros(len(ds), dtype=float)
    for n in oof:
        blended_oof += weights[n] * oof[n]
    _, per_release = _mean_per_release_reward(blended_oof, ds.y, ds.release)
    _, global_metrics = reward(blended_oof, ds.y)

    # ---- final fit on all data ----
    print("[train] fitting final base models on all data...")
    final_estimators: Dict[str, object] = {}
    for name, est in build_base_models(random_state=seed).items():
        model = clone(est)
        model.fit(ds.X, ds.y)
        final_estimators[name] = model

    blend = BlendModel(
        estimators=final_estimators,
        weights=weights,
        feature_names=list(FEATURE_NAMES),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )

    # ---- export ----
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(blend, artifact_dir / "model.joblib")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {"schema_version": FEATURE_SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES)},
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = {
        "n_groups": len(ds),
        "n_releases": len(set(ds.release.tolist())),
        "n_features": len(FEATURE_NAMES),
        "base_cv_reward": base_cv,
        "blend_weights": weights,
        "blend_cv_mean_per_release_reward": blend_cv,
        "global_oof": global_metrics,
        "per_release_reward": per_release,
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[train] global OOF metrics: {json.dumps(global_metrics)}")
    print(f"[train] artifact written to {artifact_dir / 'model.joblib'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Poker44 chunk bot detector.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--download", action="store_true", help="download releases first")
    parser.add_argument("--max-releases", type=int, default=400)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train(
        cache_dir=args.cache_dir,
        download=args.download,
        max_releases=args.max_releases,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
