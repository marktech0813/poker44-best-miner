# Poker44 Miner — Deployment Guide

End-to-end runbook for deploying `p44-chunk-behavior-ensemble` on **subnet 126
(finney)**, then keeping it fresh with a safe daily retrain.

> **Where to run what.** Bittensor + the validator/miner stack are Linux-first. Do
> registration and serving on a **Linux host** (Ubuntu 22.04 LTS recommended). You can
> train on any machine with Python 3.10+ (including the Windows box you developed on),
> but the long-running miner should live on the server.

---

## 0. Topology at a glance

```
                 ┌─────────────────────────────────────────────┐
                 │  Linux server (always-on)                    │
   validators ──▶│  axon :8091  ─▶ neurons/miner.py ─▶ Predictor │
                 │                         │                     │
                 │                         └─ artifacts/model.joblib
                 │  cron 1x/day ─▶ daily_retrain ─▶ (promote) ──┘
                 └─────────────────────────────────────────────┘
```

- **Serving**: one process (`neurons/miner.py`) under PM2, loads `artifacts/model.joblib`
  once at startup, answers `DetectionSynapse` within the 180 s timeout.
- **Retraining**: a cron job downloads new benchmark releases, retrains into a staging
  dir, and **promotes only if the model doesn't regress**, then restarts the miner.

---

## 1. Prerequisites

| Item | Recommended |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10–3.12 (bittensor support); 3.13 ok for *training only* |
| CPU / RAM | 2 vCPU / 4 GB is plenty (CPU-only inference) |
| Disk | 5 GB (benchmark cache + backups are small) |
| Network | A **public, reachable** TCP port for the axon (default 8091) |
| Funds | A little TAO for the registration burn (fluctuates) |

Install base tooling:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git curl
# PM2 for process management (needs Node):
sudo apt install -y nodejs npm && sudo npm install -g pm2
```

---

## 2. Install the subnet + this miner

```bash
# 1. Subnet
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pip install bittensor-cli

# 2. Drop in this miner (copy from your best-miner/ delivery)
cp -r /path/to/best-miner/p44miner .
cp -r /path/to/best-miner/tests .
cp -r /path/to/best-miner/artifacts .            # OK if empty; step 3 fills it
cp /path/to/best-miner/neurons/miner.py neurons/miner.py   # replaces the heuristic
cp /path/to/best-miner/run_miner.sh .
cp /path/to/best-miner/daily_retrain.sh .
pip install -r /path/to/best-miner/requirements-miner.txt
chmod +x run_miner.sh daily_retrain.sh
```

The miner file auto-discovers the sibling `p44miner/` package, so this layout
(`Poker44-subnet/p44miner`, `Poker44-subnet/artifacts`, `Poker44-subnet/neurons/miner.py`)
"just works".

---

## 3. Train the initial artifact

```bash
python -m p44miner.benchmark_client      # cache ALL public releases (idempotent)
python -m p44miner.train --download      # CV + blend-weight search + export
pytest tests -q                          # 14 tests: contract / features / reward
```

This writes:

```
artifacts/model.joblib          # the blended ensemble (loaded by the miner)
artifacts/feature_schema.json   # ordered feature names + schema version
artifacts/metrics.json          # CV metrics (used by the retrain no-regression gate)
```

Sanity-check the numbers (`metrics.json` → `blend_cv_mean_per_release_reward`). On the
current benchmark this lands around **0.85** (leave-one-release-out ≈0.85, walk-forward
≈0.80) — see `README.md` §5 for the honest live caveat.

Optional, recommended before you commit money: run the full comparison report.

```bash
python -m p44miner.evaluate              # writes artifacts/evaluation.csv + .json
```

---

## 4. Create wallet + register

```bash
# coldkey (holds funds) + hotkey (signs miner traffic)
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_hot

# check the current burn cost BEFORE registering
btcli subnet list --subtensor.network finney        # find netuid 126 + its cost
# register the hotkey on subnet 126
btcli subnet register \
  --wallet.name my_cold --wallet.hotkey my_hot \
  --netuid 126 --subtensor.network finney
```

The registration burn is **sunk once paid**. Confirm you got a UID:

```bash
btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

---

## 5. Compliance — publish your model identity (do NOT skip)

The validator records your `model_manifest` and **reviews high scorers**. A non-transparent
or copycat manifest can be penalized or zeroed. You must publish **your own** public repo
and serve the **exact commit** you publish.

1. Push this miner (the `p44miner/` package, `neurons/miner.py`, `MODEL_CARD.md`,
   training code — **not** any private data) to a public repo of *yours*.
2. Note the commit you deploy: `git rev-parse HEAD`.
3. Set these env vars (the launcher reads them). `repo_url` must be **your** repo and
   `model_name` must **not** be the reference miner's name:

```bash
export POKER44_MODEL_OPEN_SOURCE=true
export POKER44_MODEL_REPO_URL="https://github.com/<you>/poker44-best-miner"
export POKER44_MODEL_REPO_COMMIT="$(git rev-parse HEAD)"
export POKER44_MODEL_NAME="p44-chunk-behavior-ensemble"
export POKER44_MODEL_VERSION="1.0.0"
export POKER44_MODEL_FRAMEWORK="scikit-learn"
export POKER44_MODEL_LICENSE="MIT"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="Trained only on the public Poker44 benchmark releases."
export POKER44_MODEL_TRAINING_DATA_SOURCES="Poker44 public benchmark API"
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No validator-only, private, leaked, or live hidden-label data used."
```

At startup the miner logs `Manifest status=transparent` when this is satisfied; anything
else logs a warning telling you what's missing.

---

## 6. Run the miner (PM2)

The simplest path uses `run_miner.sh`, which wires the axon args, the validator allowlist,
and the compliance vars.

```bash
WALLET_NAME=my_cold HOTKEY=my_hot AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="<vali_hotkey_1> <vali_hotkey_2>" \
POKER44_MODEL_REPO_URL="https://github.com/<you>/poker44-best-miner" \
pm2 start ./run_miner.sh --name poker44_miner

pm2 logs poker44_miner          # watch startup + per-request scoring lines
pm2 save                        # persist across reboots
pm2 startup                     # follow the printed command to enable boot start
```

- If you don't know the validator hotkeys yet, omit `ALLOWED_VALIDATOR_HOTKEYS`; the
  launcher falls back to `--blacklist.force_validator_permit`.
- **Open the port**: `sudo ufw allow 8091/tcp` (and your cloud security group).
- **Cold start is normal**: reward is 0 until the validator's rolling buffer fills.

---

## 7. Daily retrain automation

`p44miner.daily_retrain` is the safe iterate-daily loop:

1. downloads only **new** releases,
2. if nothing new (and not `--force`) → exits without touching the live model,
3. trains a candidate into `artifacts/_staging/`,
4. **promotes only if** candidate CV reward ≥ live − `--tolerance`,
5. backs up the previous artifact under `artifacts/backups/<utc>/` (keeps last N),
6. optionally runs `--restart-cmd` so the miner reloads.

### Manual run

```bash
python -m p44miner.daily_retrain --restart-cmd "pm2 restart poker44_miner"
# force a rebuild even with no new data (e.g. after a code change):
python -m p44miner.daily_retrain --force --restart-cmd "pm2 restart poker44_miner"
```

Key flags: `--tolerance` (allowed regression, default 0.0 — `daily_retrain.sh` uses 0.01),
`--min-new-releases` (default 1), `--keep-backups` (default 7), `--cache-dir`,
`--artifact-dir`.

### Cron (Linux) — once a day at 06:30 UTC

The wrapper `daily_retrain.sh` activates `.venv`, logs to `logs/daily_retrain.log`, and
restarts PM2 on promotion. Edit it if your env activation differs, then:

```bash
crontab -e
# add (use the ABSOLUTE path to your subnet checkout):
30 6 * * *  RETRAIN_RESTART_CMD="pm2 restart poker44_miner" /home/me/Poker44-subnet/daily_retrain.sh
```

Check it: `tail -f /home/me/Poker44-subnet/logs/daily_retrain.log`

### Windows Task Scheduler (if you retrain on the dev box)

Create a Basic Task → Daily → "Start a program":

- Program: `C:\Users\1\anaconda3\envs\poker44\python.exe`
- Arguments: `-m p44miner.daily_retrain --force`
- Start in: `C:\Users\1\Desktop\poker44\best-miner`

(Then copy the refreshed `artifacts/` to the server, or run the retrain on the server.)

### Rolling back

```bash
ls artifacts/backups/                       # pick a timestamp
cp artifacts/backups/<ts>/* artifacts/      # restore
pm2 restart poker44_miner
```

---

## 8. Monitoring & health

- **Process**: `pm2 status`, `pm2 logs poker44_miner` (look for `Scored N chunks in Xs`).
- **Model quality over time**: `artifacts/metrics.json` after each retrain; the retrain log
  records `candidate CV reward` vs `live`.
- **On-chain standing**:
  ```bash
  btcli wallet overview --wallet.name my_cold --subtensor.network finney
  ```
  plus the public Poker44 dashboard for rank/incentive.
- **Coverage**: the validator tracks response coverage + latency — make sure the miner
  returns a score for **every** chunk (the contract guard in `forward()` enforces this).

---

## 9. Updating the code

```bash
cd Poker44-subnet
# pull your miner changes (or re-copy the files), then:
git rev-parse HEAD                                  # new commit
export POKER44_MODEL_REPO_COMMIT="$(git rev-parse HEAD)"   # keep manifest honest
pytest tests -q
pm2 restart poker44_miner
```

Whenever the **served code changes**, update the published repo + `POKER44_MODEL_REPO_COMMIT`
so the manifest keeps matching what you actually run.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Manifest status` not `transparent` | Set all `POKER44_MODEL_*` vars; `repo_url` must be yours, `repo_commit` a real hash, `model_name` not the reference name. |
| Reward stuck at 0 | Cold-start buffer still filling, or validators can't reach your axon. Check the port is open and `AXON_PORT` matches your firewall/NAT. |
| Predictor logs "heuristic fallback" | `artifacts/model.joblib` missing or schema mismatch. Re-run training; confirm `feature_schema.json` version matches the code. |
| `daily_retrain` says `skipped_no_new_data` | Expected when no new release landed. Use `--force` to rebuild anyway. |
| Candidate `kept_live` | New model regressed beyond `--tolerance`; the safe gate kept the better live model. Inspect `metrics.json`. |
| bittensor import errors on Py 3.13 | Serve on Python 3.10–3.12; 3.13 is fine for training only. |

---

## 11. Security notes

- Keep the **coldkey** off the server if possible; the miner only needs the **hotkey**.
- Restrict the axon to known validators via `ALLOWED_VALIDATOR_HOTKEYS` once you know them.
- Firewall everything except the axon port and SSH.
- Never commit wallets, `~/.bittensor`, or any private/validator-only data to your repo.
