"""Miner output-contract tests (no bittensor required)."""

from __future__ import annotations

import numpy as np

from p44miner.infer import Predictor
from p44miner.reward import reward


def test_predictor_fallback_contract(mixed_chunks):
    chunks, _ = mixed_chunks
    pred = Predictor(artifact_dir="/nonexistent-artifact-dir")
    assert not pred.ready  # no artifact -> heuristic fallback
    scores = pred.score_chunks(chunks)
    assert len(scores) == len(chunks)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_empty_input_returns_empty():
    pred = Predictor(artifact_dir="/nonexistent-artifact-dir")
    assert pred.score_chunks([]) == []
    assert pred.score_chunks(None) == []


def test_order_is_stable(mixed_chunks):
    chunks, _ = mixed_chunks
    pred = Predictor(artifact_dir="/nonexistent-artifact-dir")
    a = pred.score_chunks(chunks)
    b = pred.score_chunks(chunks)
    assert a == b


def test_trained_model_beats_random_on_synthetic(mixed_chunks):
    """Sanity: a model trained on synthetic data should rank held-out synthetic
    chunks better than chance (separable by construction)."""
    from sklearn.model_selection import train_test_split

    from p44miner.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, transform_chunks
    from p44miner.model import BlendModel, build_base_models

    chunks, labels = mixed_chunks
    X = transform_chunks(chunks)
    y = np.asarray(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

    estimators = {}
    for name, est in build_base_models(random_state=0).items():
        est.fit(Xtr, ytr)
        estimators[name] = est
    blend = BlendModel(
        estimators=estimators,
        weights={n: 1.0 for n in estimators},
        feature_names=list(FEATURE_NAMES),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    risk = blend.predict_risk(Xte)
    rew, res = reward(risk, yte)
    assert res["ap_score"] > 0.6
