#!/usr/bin/env bash
# Cron-friendly wrapper around `python -m p44miner.daily_retrain`.
# Run from inside the Poker44-subnet checkout that contains the p44miner/ package
# and artifacts/ directory. Logs to logs/daily_retrain.log.
set -euo pipefail

cd "$(dirname "$0")"

# Activate your venv/conda env (edit to match your host).
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  : # already in a venv
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p logs

# Promote only if the candidate doesn't regress; reload the miner via PM2 on promotion.
exec python -m p44miner.daily_retrain \
  --min-new-releases "${MIN_NEW_RELEASES:-1}" \
  --tolerance "${RETRAIN_TOLERANCE:-0.01}" \
  --keep-backups "${KEEP_BACKUPS:-7}" \
  --restart-cmd "${RETRAIN_RESTART_CMD:-pm2 restart poker44_miner}" \
  >> logs/daily_retrain.log 2>&1
