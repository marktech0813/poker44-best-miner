"""Compare the p44 ensemble against baselines across ALL benchmark releases.

Baselines (the concrete things we can actually run; the real top miner's code is not
public):
  * reference     - the subnet's shipped heuristic miner (faithful re-implementation)
  * best_feature  - best single behavioral feature per release with optimal sign
                    (an optimistic upper bound for "simple" miners; ties to audit topSingleAp)
  * audit_comboAp - the subnet's own internal combined-feature AP per release (from /releases)

Our model is scored two honest ways:
  * walk_forward  - train ONLY on releases strictly before the target date (mirrors live)
  * loro          - leave-one-release-out (train on every other release)

All "reward" numbers use the exact on-chain formula (0.75*AP + 0.25*recall@FPR<=0.05),
computed independently per release (mirrors the validator's per-window scoring).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import clone

from .benchmark_client import BenchmarkClient
from .features import FEATURE_NAMES, extract_chunk_features, vectorize
from .model import _proba1, build_base_models
from .reward import reward

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


# --------------------------------------------------------------------------- #
# Reference heuristic miner (verbatim logic from ref neurons/miner.py)
# --------------------------------------------------------------------------- #
def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _ref_score_hand(hand: dict) -> float:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}
    counts = Counter(a.get("action_type") for a in actions)
    meaningful = max(1, sum(counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
    call_ratio = counts.get("call", 0) / meaningful
    check_ratio = counts.get("check", 0) / meaningful
    fold_ratio = counts.get("fold", 0) / meaningful
    raise_ratio = counts.get("raise", 0) / meaningful
    street_depth = len(streets) / 3.0
    showdown = 1.0 if outcome.get("showdown") else 0.0
    pcs = (6 - min(len(players), 6)) / 4.0 if players else 0.0
    score = 0.0
    score += 0.32 * street_depth
    score += 0.22 * showdown
    score += 0.18 * _clamp01(call_ratio / 0.35)
    score += 0.12 * _clamp01(check_ratio / 0.30)
    score += 0.08 * _clamp01(pcs)
    score -= 0.18 * _clamp01(fold_ratio / 0.55)
    score -= 0.10 * _clamp01(raise_ratio / 0.20)
    return _clamp01(score)


def reference_score_chunk(chunk: List[dict]) -> float:
    if not chunk:
        return 0.5
    return round(_clamp01(sum(_ref_score_hand(h) for h in chunk) / len(chunk)), 6)


# --------------------------------------------------------------------------- #
# Data loading (single aligned pass: features + reference score + label + date)
# --------------------------------------------------------------------------- #
def load_aligned(client: BenchmarkClient):
    X: List[List[float]] = []
    ref: List[float] = []
    y: List[int] = []
    rel: List[str] = []
    for obj in client.iter_cached():
        groups = obj.get("chunks") or []
        gt = obj.get("groundTruth")
        if gt is None:
            labels = obj.get("groundTruthLabels") or []
            gt = [1 if str(v).strip().lower() in ("bot", "ai", "1", "true") else 0 for v in labels]
        if len(groups) != len(gt):
            continue
        date = str(obj.get("sourceDate") or "")
        for g, label in zip(groups, gt):
            X.append(vectorize(extract_chunk_features(g)))
            ref.append(reference_score_chunk(g))
            y.append(int(label))
            rel.append(date)
    return (
        np.asarray(X, dtype=float),
        np.asarray(ref, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(rel, dtype=object),
    )


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #
def blended_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Equal-weight probability blend of the base learners (matches BlendModel default)."""
    if len(set(ytr.tolist())) < 2 or Xtr.shape[0] < 4:
        return np.full(Xte.shape[0], 0.5)
    probs = []
    for est in build_base_models(random_state=17).values():
        m = clone(est)
        m.fit(Xtr, ytr)
        probs.append(_proba1(m, Xte))
    return np.mean(probs, axis=0)


def best_single_feature_reward(Xte: np.ndarray, yte: np.ndarray) -> Tuple[float, str]:
    """Optimistic per-release upper bound: best feature * best sign (in-sample)."""
    best = -1.0
    best_name = ""
    for j, name in enumerate(FEATURE_NAMES):
        col = Xte[:, j]
        for sign in (1.0, -1.0):
            rew, _ = reward(sign * col, yte)
            if rew > best:
                best = rew
                best_name = f"{'+' if sign > 0 else '-'}{name}"
    return best, best_name


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #
def evaluate(cache_dir: str | None = None, download: bool = False, max_releases: int = 400):
    client = BenchmarkClient(cache_dir=cache_dir) if cache_dir else BenchmarkClient()
    if download:
        client.download_all(max_releases=max_releases)

    audit_map: Dict[str, float] = {}
    try:
        for r in client.list_releases(max_releases=max_releases):
            audit = r.get("audit") or {}
            audit_map[str(r.get("sourceDate"))] = float(audit.get("comboAp") or 0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] WARN: could not fetch audit comboAp: {exc}")

    X, ref, y, rel = load_aligned(client)
    dates = sorted(set(rel.tolist()))
    print(f"[eval] {X.shape[0]} groups across {len(dates)} releases, {X.shape[1]} features")
    if X.shape[0] < 20:
        raise SystemExit("Not enough cached data. Run: python -m p44miner.evaluate --download")

    rows: List[Dict[str, object]] = []
    for i, d in enumerate(dates):
        te = rel == d
        Xte, yte, refte = X[te], y[te], ref[te]
        if len(set(yte.tolist())) < 2:
            continue  # reward undefined without both classes

        # reference miner
        ref_rew, _ = reward(refte, yte)
        # best single feature (optimistic)
        bsf_rew, bsf_name = best_single_feature_reward(Xte, yte)
        # our model, leave-one-release-out
        loro_mask = rel != d
        loro_pred = blended_predict(X[loro_mask], y[loro_mask], Xte)
        loro_rew, loro_m = reward(loro_pred, yte)
        # our model, walk-forward (train on strictly earlier dates only)
        wf_mask = np.isin(rel, dates[:i])
        if wf_mask.sum() >= 8 and len(set(y[wf_mask].tolist())) == 2:
            wf_pred = blended_predict(X[wf_mask], y[wf_mask], Xte)
            wf_rew, wf_m = reward(wf_pred, yte)
            wf_ap = wf_m["ap_score"]
        else:
            wf_rew, wf_ap = float("nan"), float("nan")

        rows.append(
            {
                "date": d,
                "n": int(te.sum()),
                "bots": int(yte.sum()),
                "reference_reward": round(ref_rew, 4),
                "best_feature_reward": round(bsf_rew, 4),
                "best_feature": bsf_name,
                "audit_comboAp": round(audit_map.get(d, float("nan")), 4),
                "loro_reward": round(loro_rew, 4),
                "loro_ap": round(loro_m["ap_score"], 4),
                "walkforward_reward": round(wf_rew, 4),
                "walkforward_ap": round(wf_ap, 4),
            }
        )

    # ---- aggregate ----
    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if not np.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "releases_scored": len(rows),
        "mean_reference_reward": round(_mean("reference_reward"), 4),
        "mean_best_feature_reward": round(_mean("best_feature_reward"), 4),
        "mean_audit_comboAp": round(_mean("audit_comboAp"), 4),
        "mean_loro_reward": round(_mean("loro_reward"), 4),
        "mean_loro_ap": round(_mean("loro_ap"), 4),
        "mean_walkforward_reward": round(_mean("walkforward_reward"), 4),
        "mean_walkforward_ap": round(_mean("walkforward_ap"), 4),
        "loro_wins_vs_reference": sum(
            1 for r in rows if float(r["loro_reward"]) > float(r["reference_reward"])
        ),
        "loro_beats_best_feature": sum(
            1 for r in rows if float(r["loro_reward"]) >= float(r["best_feature_reward"])
        ),
    }

    # ---- write artifacts ----
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps({"summary": summary, "per_release": rows}, indent=2), encoding="utf-8"
    )
    header = (
        "date,n,bots,reference,best_feat,audit_comboAp,loro_rew,loro_ap,walkfwd_rew,walkfwd_ap"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['date']},{r['n']},{r['bots']},{r['reference_reward']},"
            f"{r['best_feature_reward']},{r['audit_comboAp']},{r['loro_reward']},"
            f"{r['loro_ap']},{r['walkforward_reward']},{r['walkforward_ap']}"
        )
    (ARTIFACT_DIR / "evaluation.csv").write_text("\n".join(lines), encoding="utf-8")

    # ---- print compact report ----
    print("\n=== PER-RELEASE (reward = 0.75*AP + 0.25*recall@FPR<=0.05) ===")
    print(f"{'date':<12}{'n':>4}{'ref':>8}{'bestF':>8}{'cmboAP':>8}{'LORO':>8}{'walkfwd':>9}")
    for r in rows:
        wf = r["walkforward_reward"]
        wf_s = "  n/a" if isinstance(wf, float) and np.isnan(wf) else f"{wf:.3f}"
        print(
            f"{r['date']:<12}{r['n']:>4}{r['reference_reward']:>8.3f}"
            f"{r['best_feature_reward']:>8.3f}{r['audit_comboAp']:>8.3f}"
            f"{r['loro_reward']:>8.3f}{wf_s:>9}"
        )
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    print(f"\n[eval] wrote {ARTIFACT_DIR / 'evaluation.csv'} and evaluation.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate p44 miner vs baselines.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-releases", type=int, default=400)
    args = parser.parse_args()
    evaluate(cache_dir=args.cache_dir, download=args.download, max_releases=args.max_releases)


if __name__ == "__main__":
    main()
