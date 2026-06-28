"""Daily retrain loop for the Poker44 miner — safe, idempotent, no-regression.

What it does, in order:
    1. Download any NEW public benchmark releases into the cache (incremental).
    2. If nothing new arrived (and --force not given), exit 0 without touching the
       live artifact.
    3. Retrain into a *staging* directory (the live artifact is never mid-write).
    4. Compare the staged model's CV reward against the live artifact's. Promote the
       new model only if it is within --tolerance of (>=) the current one; otherwise
       keep the live artifact and log the skip.
    5. On promotion: back up the previous artifact (timestamped, last N kept) and copy
       the staged files into artifacts/.
    6. Optionally run a restart command (e.g. `pm2 restart poker44_miner`) so the
       running miner reloads the new model.

Run manually:
    python -m p44miner.daily_retrain
Cron / Task Scheduler example in DEPLOYMENT.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .benchmark_client import DEFAULT_CACHE_DIR, BenchmarkClient
from .train import ARTIFACT_DIR, train

ARTIFACT_FILES = ("model.joblib", "feature_schema.json", "metrics.json")
PROMOTE_METRIC = "blend_cv_mean_per_release_reward"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[daily_retrain {ts}] {msg}", flush=True)


def _read_metric(metrics_path: Path) -> Optional[float]:
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(data.get(PROMOTE_METRIC))
    except Exception:  # noqa: BLE001
        return None


def _cache_stems(cache_dir: Path) -> set[str]:
    return {p.stem for p in cache_dir.glob("*.json")}


def _backup_current(artifact_dir: Path, keep: int) -> Optional[Path]:
    """Copy the current artifact into artifacts/backups/<utc-ts>/. Prune to `keep`."""
    if not (artifact_dir / "model.joblib").exists():
        return None
    backups = artifact_dir / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backups / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        src = artifact_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    # prune oldest
    existing = sorted([d for d in backups.iterdir() if d.is_dir()])
    for old in existing[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def _promote(staging_dir: Path, artifact_dir: Path) -> None:
    for name in ARTIFACT_FILES:
        src = staging_dir / name
        if src.exists():
            shutil.copy2(src, artifact_dir / name)


def daily_retrain(
    cache_dir: Optional[str] = None,
    artifact_dir: Path = ARTIFACT_DIR,
    min_new_releases: int = 1,
    tolerance: float = 0.0,
    keep_backups: int = 7,
    force: bool = False,
    restart_cmd: Optional[str] = None,
    n_splits: int = 5,
    seed: int = 17,
) -> Dict[str, object]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    client = BenchmarkClient(cache_dir=cache_dir) if cache_dir else BenchmarkClient()
    resolved_cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    # 1) incremental download
    before = _cache_stems(resolved_cache)
    _log(f"cache has {len(before)} chunk files; checking for new releases...")
    try:
        client.download_all(refresh=False)
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR during download: {exc}")
        if not force:
            return {"status": "error", "stage": "download", "error": str(exc)}
    after = _cache_stems(resolved_cache)
    new_files = len(after - before)
    _log(f"{new_files} new chunk file(s) downloaded (cache now {len(after)}).")

    if new_files < min_new_releases and not force:
        _log(f"fewer than {min_new_releases} new file(s); nothing to do. Use --force to retrain anyway.")
        return {"status": "skipped_no_new_data", "new_files": new_files}

    # 2) train into staging
    staging = artifact_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    _log("training candidate model into staging...")
    try:
        new_metrics = train(
            cache_dir=cache_dir,
            download=False,
            n_splits=n_splits,
            seed=seed,
            artifact_dir=staging,
        )
    except SystemExit as exc:
        _log(f"training aborted: {exc}")
        return {"status": "error", "stage": "train", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR during training: {exc}")
        return {"status": "error", "stage": "train", "error": str(exc)}

    new_reward = float(new_metrics.get(PROMOTE_METRIC) or 0.0)
    live_reward = _read_metric(artifact_dir / "metrics.json")
    _log(f"candidate CV reward = {new_reward:.4f} | live = "
         f"{'n/a' if live_reward is None else f'{live_reward:.4f}'}")

    # 3) no-regression gate
    promote = (
        live_reward is None
        or force
        or new_reward >= live_reward - tolerance
    )
    if not promote:
        _log(f"candidate regressed beyond tolerance ({tolerance}); keeping live artifact.")
        shutil.rmtree(staging, ignore_errors=True)
        return {
            "status": "kept_live",
            "new_reward": new_reward,
            "live_reward": live_reward,
            "new_files": new_files,
        }

    # 4) backup + promote
    backup = _backup_current(artifact_dir, keep=keep_backups)
    _promote(staging, artifact_dir)
    shutil.rmtree(staging, ignore_errors=True)
    _log(f"promoted new artifact (backup: {backup}).")

    # 5) optional restart
    if restart_cmd:
        _log(f"running restart command: {restart_cmd}")
        try:
            subprocess.run(restart_cmd, shell=True, check=True)
            _log("restart command succeeded.")
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING: restart command failed: {exc}")

    return {
        "status": "promoted",
        "new_reward": new_reward,
        "live_reward": live_reward,
        "new_files": new_files,
        "backup": str(backup) if backup else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Daily safe retrain for the Poker44 miner.")
    p.add_argument("--cache-dir", default=None, help="benchmark cache dir (default: package data dir)")
    p.add_argument("--artifact-dir", default=str(ARTIFACT_DIR), help="live artifact dir")
    p.add_argument("--min-new-releases", type=int, default=1,
                   help="minimum new cache files required to trigger a retrain")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="allowed CV-reward regression before refusing to promote")
    p.add_argument("--keep-backups", type=int, default=7, help="how many old artifacts to keep")
    p.add_argument("--force", action="store_true", help="retrain + promote even without new data")
    p.add_argument("--restart-cmd", default=None,
                   help="shell command to reload the miner, e.g. 'pm2 restart poker44_miner'")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    result = daily_retrain(
        cache_dir=args.cache_dir,
        artifact_dir=Path(args.artifact_dir),
        min_new_releases=args.min_new_releases,
        tolerance=args.tolerance,
        keep_backups=args.keep_backups,
        force=args.force,
        restart_cmd=args.restart_cmd,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    _log(f"result: {json.dumps(result)}")
    # exit non-zero only on hard errors, so cron mail flags real failures
    sys.exit(1 if result.get("status") == "error" else 0)


if __name__ == "__main__":
    main()
