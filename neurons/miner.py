"""Poker44 production miner: artifact-backed chunk-level bot detector.

Drop-in replacement for ``neurons/miner.py`` in the Poker44-subnet repo. Copy the
sibling ``p44miner/`` package and ``artifacts/`` directory into the subnet checkout
(or keep this folder layout and let the sys.path shim below find them).

Contract (verified against poker44/validator/forward.py):
* read ``synapse.chunks`` (list of chunks; each chunk is a list of hand dicts);
* return ``risk_scores`` with len == number of chunks, each in [0, 1], in input order;
* ``predictions`` and ``model_manifest`` are optional but recommended.
"""

# from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Bittensor >=9 ships BT_NO_PARSE_CLI_ARGS defaulting to "true", which makes
# bt.Config ignore every --flag and drop config.netuid/config.neuron/etc.,
# crashing the subnet's check_config(). Re-enable CLI parsing unless the
# operator has deliberately overridden it.
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")

import bittensor as bt

# Make the sibling p44miner package importable whether this file lives in the subnet
# repo's neurons/ dir or in this deliverable's neurons/ dir.
_THIS = Path(__file__).resolve()
for _candidate in (_THIS.parents[1], _THIS.parents[2]):
    if (_candidate / "p44miner").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from p44miner import __version__ as P44_VERSION
from p44miner.infer import Predictor


class Miner(BaseMinerNeuron):
    """Trained ensemble miner. One bot-risk score per chunk."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        artifact_dir = os.getenv("POKER44_ARTIFACT_DIR")
        self.predictor = Predictor(artifact_dir) if artifact_dir else Predictor()
        bt.logging.info(f"🤖 Poker44 ensemble miner started | predictor: {self.predictor.reason}")

        repo_root = _THIS.parents[1]
        impl_files = [
            _THIS,
            repo_root / "p44miner" / "features.py",
            repo_root / "p44miner" / "model.py",
            repo_root / "p44miner" / "infer.py",
        ]
        impl_files = [p for p in impl_files if p.exists()]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=impl_files,
            defaults={
                # NOTE: override these via POKER44_MODEL_* env vars before going live.
                # repo_url MUST be your own public repo (not the reference repo) and
                # model_name MUST NOT be the reference name, or compliance flags it.
                "model_name": "p44-chunk-behavior-ensemble",
                "model_version": P44_VERSION,
                "framework": "scikit-learn",
                "license": "MIT",
                "repo_url": "https://github.com/CHANGE_ME/poker44-best-miner",
                "open_source": True,
                "inference_mode": "remote",
                "notes": "Release-robust chunk-level behavioral ensemble (HGB+ExtraTrees+LogReg).",
                "training_data_statement": (
                    "Trained only on the public Poker44 benchmark releases "
                    "(https://api.poker44.net/api/v1/benchmark)."
                ),
                "training_data_sources": ["Poker44 public benchmark API"],
                "private_data_attestation": (
                    "No validator-only, private, leaked, or live hidden-label data was used."
                ),
                "data_attestation": (
                    "Features are derived only from miner-visible canonicalized payloads."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Manifest status={self.manifest_compliance['status']} "
            f"missing={self.manifest_compliance['missing_fields']} "
            f"violations={self.manifest_compliance['policy_violations']} "
            f"digest={self.manifest_digest}"
        )
        if self.manifest_compliance["status"] != "transparent":
            bt.logging.warning(
                "Model manifest is NOT transparent yet. Set POKER44_MODEL_REPO_URL, "
                "POKER44_MODEL_REPO_COMMIT (real git hash), and related vars before "
                "competing at the top of the board."
            )
        bt.logging.info(f"Axon created: {self.axon}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        start = time.perf_counter()
        chunks = synapse.chunks or []
        scores = self.predictor.score_chunks(chunks)

        # Hard contract guarantee: exactly one score per chunk, clamped to [0, 1].
        if len(scores) != len(chunks):
            scores = (scores + [0.5] * len(chunks))[: len(chunks)]
        scores = [float(max(0.0, min(1.0, s))) for s in scores]

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        elapsed = time.perf_counter() - start
        self._log_request(synapse, chunks, scores, elapsed)
        return synapse

    def _log_request(self, synapse, chunks, scores, elapsed) -> None:
        """Per-request validator + chunk telemetry.

        One concise INFO line per query: which validator sent it, how many
        chunks/hands arrived, and the score distribution we returned. Set
        POKER44_LOG_CHUNKS=1 to additionally dump the full raw chunk payload
        per chunk at DEBUG (verbose; for inspection only).
        """
        hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
        try:
            vuid = self.metagraph.hotkeys.index(hotkey) if hotkey in self.metagraph.hotkeys else -1
        except Exception:
            vuid = -1
        sizes = [len(c) for c in chunks]
        total_hands = sum(sizes)
        flagged = sum(1 for s in scores if s >= 0.5)
        smin = min(scores) if scores else 0.0
        smax = max(scores) if scores else 0.0
        smean = (sum(scores) / len(scores)) if scores else 0.0
        head = sizes[:10]
        more = "..." if len(sizes) > 10 else ""
        bt.logging.info(
            f"[req] validator uid={vuid} hk={str(hotkey)[:10]} | "
            f"chunks={len(chunks)} total_hands={total_hands} hands/chunk={head}{more} | "
            f"flagged>=0.5={flagged}/{len(scores)} "
            f"score[min/mean/max]={smin:.3f}/{smean:.3f}/{smax:.3f} | {elapsed:.3f}s"
        )
        if os.getenv("POKER44_LOG_CHUNKS", "0").strip().lower() not in ("0", "", "false", "no"):
            for i, (chunk, score) in enumerate(zip(chunks, scores)):
                bt.logging.debug(
                    f"[req][chunk {i}] hands={len(chunk)} score={score:.4f} payload={chunk}"
                )

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 ensemble miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
