"""Tests pinning the reward to the on-chain formula."""

from __future__ import annotations

import numpy as np

from p44miner.reward import _recall_at_fpr, reward


def test_perfect_ranking_scores_high():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    rew, res = reward(scores, y)
    assert res["ap_score"] > 0.99
    assert res["human_safety_penalty"] == 1.0
    # 0.75*AP + 0.25*recall, both near 1.0 for perfect separation
    assert rew > 0.9


def test_weighting_is_075_025():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    rew, res = reward(scores, y)
    expected = 0.75 * res["ap_score"] + 0.25 * res["bot_recall"]
    assert abs(rew - expected) < 1e-9


def test_random_scores_are_mediocre():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    scores = rng.random(100)
    rew, res = reward(scores, y)
    assert 0.0 <= rew <= 1.0
    assert res["ap_score"] < 0.75  # below the audited separability ceiling


def test_recall_at_fpr_respects_budget():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # one human ranked above all bots -> at FPR<=0.05 (0 allowed FPs here) recall limited
    scores = np.array([0.99] + [0.1] * 9 + [0.8, 0.8, 0.8, 0.8, 0.8])
    recall, fpr = _recall_at_fpr(scores, y, max_fpr=0.05)
    assert fpr <= 0.05
    assert 0.0 <= recall <= 1.0
