"""Ensemble model for Poker44 chunk-level bot detection.

Why this design:
* The reward is rank-based (AP + recall@FPR), so we optimize ranking quality and blend
  diverse rankers rather than chasing probability calibration.
* Data is small (~hundreds of groups) and the signal direction flips across releases, so
  every base learner is deliberately regularized to avoid memorizing one release.
* Dependency-light: only scikit-learn (already in the subnet requirements). LightGBM is
  supported opportunistically if installed, but never required.

``BlendModel`` is the serialized artifact: fitted base estimators + per-model blend
weights. Inference = weighted average of base ``predict_proba`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_base_models(random_state: int = 17) -> Dict[str, object]:
    """Return the regularized base learners keyed by name."""
    models: Dict[str, object] = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            max_iter=300,
            early_stopping=False,
            random_state=random_state,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=8,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "logreg": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }
    # Optional LightGBM if available (kept off the critical path).
    try:  # pragma: no cover - optional dependency
        from lightgbm import LGBMClassifier

        models["lgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=-1,
        )
    except Exception:  # noqa: BLE001
        pass
    return models


def _proba1(estimator, X: np.ndarray) -> np.ndarray:
    """Probability of the positive (bot) class, robust to single-class fits."""
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        return np.asarray(proba)[:, classes.index(1)]
    # estimator only saw one class -> emit that class as a constant ranking
    return np.full(X.shape[0], float(classes[0] == 1))


@dataclass
class BlendModel:
    """Serializable ensemble: fitted base estimators + blend weights."""

    estimators: Dict[str, object]
    weights: Dict[str, float]
    feature_names: List[str]
    feature_schema_version: str

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        total_w = sum(max(0.0, w) for w in self.weights.values()) or 1.0
        blended = np.zeros(X.shape[0], dtype=float)
        for name, est in self.estimators.items():
            w = max(0.0, float(self.weights.get(name, 0.0)))
            if w <= 0:
                continue
            blended += (w / total_w) * _proba1(est, X)
        blended = np.clip(blended, 0.0, 1.0)
        return np.column_stack([1.0 - blended, blended])

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return the bot-risk score (positive-class probability) per row."""
        return self.predict_proba(X)[:, 1]
