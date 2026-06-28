#!/usr/bin/env bash
# Launch the Poker44 ensemble miner. Run from inside a Poker44-subnet checkout that has
# the p44miner/ package and artifacts/ directory available (see README "Deploy").
set -euo pipefail

: "${WALLET_NAME:?set WALLET_NAME}"
: "${HOTKEY:?set HOTKEY}"
: "${AXON_PORT:=8091}"
: "${NETUID:=126}"
: "${SUBTENSOR_NETWORK:=finney}"

# Bittensor >=9 disables CLI arg parsing by default (BT_NO_PARSE_CLI_ARGS=true),
# which would drop --netuid/--wallet/--axon flags and break config validation.
export BT_NO_PARSE_CLI_ARGS="${BT_NO_PARSE_CLI_ARGS:-0}"

# ---- model identity / compliance (publish your REAL repo + commit) ----
export POKER44_MODEL_OPEN_SOURCE="${POKER44_MODEL_OPEN_SOURCE:-true}"
export POKER44_MODEL_NAME="${POKER44_MODEL_NAME:-p44-chunk-behavior-ensemble}"
export POKER44_MODEL_VERSION="${POKER44_MODEL_VERSION:-1.0.0}"
export POKER44_MODEL_FRAMEWORK="${POKER44_MODEL_FRAMEWORK:-scikit-learn}"
export POKER44_MODEL_LICENSE="${POKER44_MODEL_LICENSE:-MIT}"
export POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL:?set POKER44_MODEL_REPO_URL to your public repo}"
export POKER44_MODEL_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-$(git rev-parse HEAD)}"
export POKER44_MODEL_TRAINING_DATA_STATEMENT="${POKER44_MODEL_TRAINING_DATA_STATEMENT:-Trained only on the public Poker44 benchmark releases.}"
export POKER44_MODEL_TRAINING_DATA_SOURCES="${POKER44_MODEL_TRAINING_DATA_SOURCES:-Poker44 public benchmark API}"
export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="${POKER44_MODEL_PRIVATE_DATA_ATTESTATION:-No validator-only, private, leaked, or live hidden-label data used.}"

ARGS=(
  neurons/miner.py
  --netuid "${NETUID}"
  --wallet.name "${WALLET_NAME}"
  --wallet.hotkey "${HOTKEY}"
  --subtensor.network "${SUBTENSOR_NETWORK}"
  --axon.port "${AXON_PORT}"
)

if [[ -n "${ALLOWED_VALIDATOR_HOTKEYS:-}" ]]; then
  # shellcheck disable=SC2206
  HOTKEYS=(${ALLOWED_VALIDATOR_HOTKEYS})
  ARGS+=(--blacklist.allowed_validator_hotkeys "${HOTKEYS[@]}")
else
  ARGS+=(--blacklist.force_validator_permit)
fi

# Resolve the Python interpreter without relying on PATH (robust under PM2 / reboots).
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PY="$(pwd)/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

echo "Starting Poker44 miner: netuid=${NETUID} port=${AXON_PORT} python=${PY}"
exec "${PY}" "${ARGS[@]}"
