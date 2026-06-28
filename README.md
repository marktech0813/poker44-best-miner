# Poker44 Best Miner — `p44-chunk-behavior-ensemble`

A fast, release-robust, **chunk-level behavioral bot detector** for **Poker44 subnet 126**.
It ingests `DetectionSynapse(chunks=...)`, returns **one bot-risk score per chunk**, and
ships a compliant open-source model manifest.

This README doubles as the technical review: it documents what was **verified against the
real subnet code and the live benchmark API**, including several places where common
"direction docs" are wrong.

---

## 1. What the competition actually is (verified)

* **Task:** behavioral *bot detection*, not poker playing. A *chunk* is ~30–40 hands of a
  **single entity** with **one label** (`1=bot`, `0=human`). See
  `poker44/validator/forward.py` (`LabeledHandBatch.is_human` → one label per batch) and the
  benchmark `groundTruth` array (one label per group).
* **Output contract:** `risk_scores` length **must equal** the number of received chunks,
  each in `[0,1]`, in **input order**. `predictions`/`model_manifest` are optional.

### The reward — corrected

The reference doc floating around says `0.65*AP + 0.35*recall` times `(1-FPR)^2`, with the
human-safety penalty zeroed at `FPR≥10%`. **That is not the code.** The real
`poker44/score/scoring.py` is:

```
base = 0.75 * average_precision + 0.25 * recall_at(FPR ≤ 0.05)
reward = base * human_safety_penalty      # human_safety_penalty is a CONSTANT 1.0
```

Consequences that shape this miner:

* **AP and recall@FPR are both rank-based.** They depend only on the *ordering* of your
  scores vs the hidden labels — **not** on calibration or a 0.5 threshold. So Platt/isotonic
  calibration, "keep humans below 0.5", and threshold tuning are **irrelevant to score**.
  We optimize **ranking (AP)**, full stop.
* The 5% FPR budget is **baked into the recall term**, evaluated by a threshold sweep over
  your scores. You want the very top of your ranking to be **clean bots**.
* There is **no `(1-FPR)^2` multiplier and no "10% FPR zeroes you" rule** in the code.

### Payout & cadence (verified)

* `poker44/validator/constants.py`: `WINNER_TAKE_ALL=False`, podium split **50/30/20** to the
  top-3; `BURN_FRACTION=0.00`. So top-3 is paid, #1 takes half.
* Query **timeout is 180s** (`forward.py`), not 1s — latency is not a real constraint, but we
  still load the model once and keep inference fast.
* Reward is **0 until the validator's rolling buffer fills** — expect a cold-start ramp.

### What the canonicalizer destroys (so we ignore it)

`poker44/validator/payload_view.py` + live payloads confirm these carry **no signal**:

* `outcome` (winners/payouts/total_pot/**showdown** all zeroed),
* `streets[].board_cards` (always `[]`) and the **top-level `streets` array is empty** in real
  data → street depth must be derived from `action.street`,
* `metadata.sb/bb/ante` (constant `0.01/0.02/0.0`), hole cards (null).

The shipped **reference miner scores on `showdown` and `len(streets)` — both are dead.** This
miner avoids that trap.

### The audit leak (free feature intelligence)

`/benchmark/releases` exposes an `audit` block with the **exact discriminative features** and
their per-feature AP: `mean_starting_stack`, `aggression_rate`, `check_rate`, `fold_rate`,
`mean_action_count`, `mean_pot_growth`, `zero_amount_share`, `mean_player_count`,
`action_entropy`, `mean_street_count`, `hero_seat_mean`. Two critical facts:

1. **Separability is capped** (`maxSingleFeatureAp≤0.68`, `maxComboAp≤0.76`). Realistic AP lives
   in ~0.55–0.70; the live leader ≈0.635 composite is near the ceiling, not a soft target.
2. **Feature directions flip between releases** (e.g. `mean_starting_stack` is "direct" some
   days, "inverse" others). A model that hardcodes a direction breaks on flip days. → we
   **train across all releases**, expose magnitudes, keep features **compact + regularized**.

---

## 2. Architecture

```
p44miner/
  features.py        # chunk -> fixed-length feature row (shared by train + infer)
  reward.py          # verbatim copy of the on-chain reward
  benchmark_client.py# download + cache all releases by chunkHash
  dataset.py         # cached releases -> (X, y, release-group)
  model.py           # HGB + ExtraTrees + scaled LogReg, blended (+ optional LightGBM)
  train.py           # release-split CV, blend-weight search vs true reward, export
  infer.py           # artifact load + safe batch inference (heuristic fallback)
neurons/miner.py     # drop-in Bittensor miner + compliant manifest
artifacts/           # model.joblib, feature_schema.json, metrics.json (generated)
tests/               # feature / reward / contract tests
```

**Modeling choices, justified by the facts above:**

* **Ensemble of regularized rankers** (`HistGradientBoosting` + `ExtraTrees` + scaled
  `LogisticRegression`), blended by weights chosen to maximize the **true per-release reward**
  on out-of-fold predictions. Diverse rankers + blending is the highest-AP-per-risk move on
  small tabular data.
* **scikit-learn only** (already in the subnet requirements) — no native build deps, trivial
  deploy. LightGBM is used *only if present*.
* **~60 compact features** (not 200–600). With ~hundreds of labeled groups, a huge feature set
  would overfit; we center on the audited features plus robust dispersion (std/min/max/quantiles)
  and scripted-behavior signals (sequence repetition, bet-size rigidity).
* **GroupKFold by `sourceDate`** so a release never leaks across folds — this is the only honest
  estimate of how you'll do on tomorrow's unseen window.

---

## 3. Setup, train, deploy

> Requires **Python 3.10+**. (This was authored on a Windows box without Python; do training
> and serving on your Linux deploy host.)

### a) Get the subnet + this miner

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pip install bittensor-cli

# add this miner
cp -r /path/to/best-miner/p44miner .
cp -r /path/to/best-miner/artifacts .          # created by training (step b)
cp /path/to/best-miner/neurons/miner.py neurons/miner.py   # replaces the heuristic
pip install -r /path/to/best-miner/requirements-miner.txt
```

### b) Train (downloads the public benchmark, then fits + exports)

```bash
python -m p44miner.benchmark_client            # cache all releases (idempotent)
python -m p44miner.train --download            # CV + blend + export to artifacts/
pytest /path/to/best-miner/tests -q            # contract / feature / reward tests
```

`train.py` prints per-base-model and blended **mean per-release reward** (your honest CV
estimate) and writes `artifacts/metrics.json` for regression tracking. Retrain when new
releases land — never fold validation/test back in without creating a newer holdout.

### c) Register + run

```bash
btcli subnet register --wallet.name my_cold --wallet.hotkey my_hot \
  --netuid 126 --subtensor.network finney      # check burn cost first

WALLET_NAME=my_cold HOTKEY=my_hot AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="<vali_hotkey_1> <vali_hotkey_2>" \
POKER44_MODEL_REPO_URL="https://github.com/<you>/poker44-best-miner" \
bash run_miner.sh
```

Run under PM2 for resilience: `pm2 start run_miner.sh --name poker44_miner`.

---

## 4. Compliance (don't get zeroed)

The validator records your manifest and reviews high scorers. `evaluate_manifest_compliance`
requires: `open_source=true`, real `repo_url` (**not** the reference repo), real
`repo_commit` matching `^[0-9a-f]{7,40}$`, `model_name` (**not** the reference name),
`model_version`, `training_data_statement`, `private_data_attestation`,
`implementation_files`, `implementation_sha256`.

`run_miner.sh` wires the `POKER44_MODEL_*` env vars. **Publish the real repo and the actual
commit you serve**, keep the served code identical to what's published, and keep the
`training_data_statement`/`private_data_attestation` truthful. See `MODEL_CARD.md`.

---

## 5. Honest expectations

This is a near-winner-take-all (top-3 podium) competition against a **deliberately capped**
signal (~0.76 AP ceiling) whose **discriminative direction changes daily**. No static model
dominates every window. The realistic goal: a robust ranker that sits at/above the field's
per-release reward on average and degrades gracefully on flip days — then iterate daily as
new releases land. The biggest lever is **ranking quality (AP)**, not calibration or FPR
threshold tricks.
