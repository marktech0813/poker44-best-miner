# Model Card — `p44-chunk-behavior-ensemble`

## Overview
- **Task:** Chunk-level (player-level) bot detection for Poker44 subnet 126.
- **Input:** One chunk = a list of ~30–40 miner-visible, canonicalized poker-hand dicts.
- **Output:** One bot-risk score in `[0,1]` per chunk (higher = more bot-like). Only the
  *ranking* of scores is rewarded (`0.75*AP + 0.25*recall@FPR≤0.05`).
- **Framework:** scikit-learn (optional LightGBM). CPU-only, no network in the hot path.

## Architecture
- Ensemble blend of three regularized rankers:
  - `HistGradientBoostingClassifier` (depth 3, l2=1.0)
  - `ExtraTreesClassifier` (depth 6, balanced, sqrt features)
  - `StandardScaler → LogisticRegression` (C=0.5, balanced)
- Blend weights selected by grid search to **maximize the true mean per-release reward** on
  release-separated out-of-fold predictions.
- ~60 features: audited core rates (aggression/check/fold/zero-amount/entropy), per-hand
  dispersion stats (action count, street depth, player count, hero seat, starting-stack BB,
  pot growth, bet sizing), and scripted-behavior signals (sequence repetition, bet-size
  rigidity). No `outcome`, `board_cards`, `sb/bb/ante`, IDs, dates, or hashes are used.

## Training data
- **Source:** Public Poker44 benchmark API only
  (`https://api.poker44.net/api/v1/benchmark`), cached by `chunkHash`.
- **Splits:** GroupKFold by `sourceDate`; newest release(s) held out locally.
- **Private-data attestation:** No validator-only, private, leaked, or live hidden-label data
  was used. Features derive solely from miner-visible canonicalized payloads.

## Evaluation
- Primary metric: mean per-release reward (exact on-chain formula), reported in
  `artifacts/metrics.json` along with per-base-model CV reward and global OOF AP/recall/FPR.

## Intended use & limitations
- Intended only as a Poker44 subnet-126 miner.
- Discriminative signal is capped (~0.76 AP ceiling) and its **direction flips between
  releases**; performance varies day to day. Retrain as new releases publish.
- The model ranks; it is not a calibrated probability of "bot".

## Reproducibility
- `python -m p44miner.benchmark_client` then `python -m p44miner.train --download`.
- Artifact: `artifacts/model.joblib` (+ `feature_schema.json`, `metrics.json`).
- `implementation_sha256` in the live manifest is computed over `neurons/miner.py`,
  `p44miner/features.py`, `p44miner/model.py`, `p44miner/infer.py`.
