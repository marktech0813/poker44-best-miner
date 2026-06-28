"""Chunk-level behavioral feature extraction for Poker44 bot detection.

A *chunk* (a.k.a. group / batch) is a list of poker-hand dicts that all belong to
ONE entity (one player) and therefore share ONE label: bot (1) or human (0).
The validator sends ``DetectionSynapse.chunks`` (a list of such chunks) and expects
exactly one risk score per chunk, so we collapse each chunk into a single fixed-length
feature row.

Design constraints derived from the real subnet code + live benchmark payloads:

* The miner-visible payload is canonicalized by ``poker44/validator/payload_view.py``.
  These fields are CONSTANT / empty and carry no signal -> we ignore them:
    - ``outcome`` (winners/payouts/total_pot/showdown all zeroed)
    - ``streets[].board_cards`` (always [])  and the top-level ``streets`` array
      (empty in real benchmark payloads -> street info is derived from action.street)
    - ``metadata.sb / bb / ante`` (constant 0.01 / 0.02 / 0.0)
    - hole cards (always null)
* Actions are sampled to a small window (5-12) per hand and amounts are bucketed with
  deterministic noise, so we lean on robust aggregate statistics, not exact values.
* The on-chain audit publishes the discriminative single features (mean_starting_stack,
  aggression_rate, check_rate, fold_rate, mean_action_count, mean_pot_growth,
  zero_amount_share, mean_player_count, action_entropy, mean_street_count,
  hero_seat_mean). Their *direction flips between releases*, so we expose magnitudes
  and let a cross-release-trained model decide; we do NOT hardcode a direction.

Everything here is pure Python + numpy and exception-safe: malformed input must never
raise, it must degrade to neutral feature values.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np

# Bump when the feature layout changes so artifacts and runtime stay in sync.
FEATURE_SCHEMA_VERSION = "p44-feat-v1"

# big-blind unit used by the canonicalizer (metadata.bb is constant 0.02).
_VISIBLE_BB = 0.02

_DECISION_ACTIONS = ("check", "call", "bet", "raise", "fold")
_AGGRESSIVE = ("bet", "raise")
_PASSIVE = ("check", "call")
_STREET_ORDER = ("preflop", "flop", "turn", "river")


# --------------------------------------------------------------------------- #
# Small numeric helpers (all NaN/None safe)
# --------------------------------------------------------------------------- #
def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _to_bb(value: Any) -> float:
    """Convert a visible chip amount back to big blinds (bb == 0.02)."""
    return _f(value) / _VISIBLE_BB if _VISIBLE_BB > 0 else 0.0


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    """mean / std / min / max / p25 / p50 / p75 for a list of values."""
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
    }


def _entropy(counts: Sequence[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


# --------------------------------------------------------------------------- #
# Per-hand reduction
# --------------------------------------------------------------------------- #
def _hand_summary(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one hand to the primitive signals we aggregate over the chunk."""
    if not isinstance(hand, dict):
        hand = {}

    players = hand.get("players") if isinstance(hand.get("players"), list) else []
    actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}

    # players / stacks (in BB)
    stacks_bb: List[float] = []
    for p in players:
        if isinstance(p, dict):
            stacks_bb.append(_to_bb(p.get("starting_stack")))

    # action-level aggregation
    type_counts: Dict[str, int] = {k: 0 for k in _DECISION_ACTIONS}
    streets_seen: set[str] = set()
    amounts_bb: List[float] = []          # nonzero aggressive amounts (already in bb)
    pot_growth: List[float] = []
    zero_amount = 0
    raise_to_used = 0
    call_to_used = 0
    seq: List[str] = []
    distinct_amount_vals: set[float] = set()

    for a in actions:
        if not isinstance(a, dict):
            continue
        atype = str(a.get("action_type") or "").strip().lower()
        if atype in type_counts:
            type_counts[atype] += 1
        seq.append(atype if atype else "?")

        street = str(a.get("street") or "").strip().lower()
        if street:
            streets_seen.add(street)

        amt_bb = _f(a.get("normalized_amount_bb"))
        if amt_bb <= 0:
            zero_amount += 1
        else:
            amounts_bb.append(amt_bb)
            distinct_amount_vals.add(round(amt_bb, 2))

        pb = _f(a.get("pot_before"))
        pa = _f(a.get("pot_after"))
        pot_growth.append(_to_bb(pa) - _to_bb(pb))

        if a.get("raise_to") not in (None, 0, 0.0):
            raise_to_used += 1
        if a.get("call_to") not in (None, 0, 0.0):
            call_to_used += 1

    n_actions = len([a for a in actions if isinstance(a, dict)])
    decision_total = sum(type_counts.values())
    aggressive = type_counts["bet"] + type_counts["raise"]
    passive = type_counts["check"] + type_counts["call"]

    # street depth: how far the hand progressed (preflop..river)
    depth = 0
    for i, s in enumerate(_STREET_ORDER, start=1):
        if s in streets_seen:
            depth = max(depth, i)

    return {
        "n_players": float(len(players)),
        "n_actions": float(n_actions),
        "hero_seat": _f(metadata.get("hero_seat")),
        "max_seats": _f(metadata.get("max_seats")),
        "stacks_bb": stacks_bb,
        "type_counts": type_counts,
        "decision_total": float(decision_total),
        "aggressive": float(aggressive),
        "passive": float(passive),
        "amounts_bb": amounts_bb,
        "pot_growth": pot_growth,
        "zero_amount": float(zero_amount),
        "raise_to_used": float(raise_to_used),
        "call_to_used": float(call_to_used),
        "street_count": float(len(streets_seen)),
        "street_depth": float(depth),
        "reached_flop": 1.0 if depth >= 2 else 0.0,
        "reached_turn": 1.0 if depth >= 3 else 0.0,
        "reached_river": 1.0 if depth >= 4 else 0.0,
        "seq": tuple(seq),
        "distinct_amount_vals": len(distinct_amount_vals),
        "empty_actions": 1.0 if n_actions == 0 else 0.0,
        "empty_players": 1.0 if len(players) == 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Chunk-level feature dict
# --------------------------------------------------------------------------- #
def extract_chunk_features(chunk: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Collapse one chunk (list of hands) into a named feature dict.

    Never raises: bad input yields a neutral (mostly-zero) feature dict.
    """
    if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    hands = [_hand_summary(h) for h in chunk]
    n_hands = float(len(hands))

    # ---- pooled action-type totals across the whole chunk ----
    pooled = {k: 0.0 for k in _DECISION_ACTIONS}
    for h in hands:
        for k in _DECISION_ACTIONS:
            pooled[k] += h["type_counts"][k]
    pooled_total = sum(pooled.values())
    pooled_aggr = pooled["bet"] + pooled["raise"]
    pooled_zero = sum(h["zero_amount"] for h in hands)
    pooled_nonzero_amt = sum(len(h["amounts_bb"]) for h in hands)
    pooled_actions = sum(h["n_actions"] for h in hands)
    pooled_raise_to = sum(h["raise_to_used"] for h in hands)
    pooled_call_to = sum(h["call_to_used"] for h in hands)

    feats: Dict[str, float] = {}

    # ---- audit-proven core rates (pooled) ----
    feats["aggression_rate"] = _safe_div(pooled_aggr, pooled_total)
    feats["check_rate"] = _safe_div(pooled["check"], pooled_total)
    feats["call_rate"] = _safe_div(pooled["call"], pooled_total)
    feats["bet_rate"] = _safe_div(pooled["bet"], pooled_total)
    feats["raise_rate"] = _safe_div(pooled["raise"], pooled_total)
    feats["fold_rate"] = _safe_div(pooled["fold"], pooled_total)
    feats["passive_rate"] = _safe_div(pooled["check"] + pooled["call"], pooled_total)
    feats["aggression_factor"] = _safe_div(pooled_aggr, pooled["call"] + 1.0)
    feats["zero_amount_share"] = _safe_div(pooled_zero, pooled_actions)
    feats["action_entropy"] = _entropy([pooled[k] for k in _DECISION_ACTIONS])
    feats["raise_to_share"] = _safe_div(pooled_raise_to, pooled_actions)
    feats["call_to_share"] = _safe_div(pooled_call_to, pooled_actions)

    # ---- per-hand distributions (dispersion captures bot rigidity) ----
    feats.update(_stats([h["n_actions"] for h in hands], "action_count"))
    feats.update(_stats([h["street_count"] for h in hands], "street_count"))
    feats.update(_stats([h["street_depth"] for h in hands], "street_depth"))
    feats.update(_stats([h["n_players"] for h in hands], "player_count"))
    feats.update(_stats([h["hero_seat"] for h in hands], "hero_seat"))

    # starting stacks (BB), pooled across all players in all hands
    all_stacks = [s for h in hands for s in h["stacks_bb"]]
    feats.update(_stats(all_stacks, "starting_stack"))

    # pot growth (BB) per action, pooled
    all_growth = [g for h in hands for g in h["pot_growth"]]
    feats.update(_stats(all_growth, "pot_growth"))

    # aggressive bet sizing (BB), pooled nonzero amounts
    all_amounts = [a for h in hands for a in h["amounts_bb"]]
    feats.update(_stats(all_amounts, "bet_size_bb"))

    # ---- street-depth reach rates ----
    feats["reach_flop_rate"] = _safe_div(sum(h["reached_flop"] for h in hands), n_hands)
    feats["reach_turn_rate"] = _safe_div(sum(h["reached_turn"] for h in hands), n_hands)
    feats["reach_river_rate"] = _safe_div(sum(h["reached_river"] for h in hands), n_hands)

    # ---- scripted-behavior / rigidity signals ----
    seqs = [h["seq"] for h in hands if h["seq"]]
    feats["seq_unique_ratio"] = _safe_div(len(set(seqs)), len(seqs)) if seqs else 0.0
    # most-common identical action-sequence share (repetition -> scripted)
    if seqs:
        most = Counter(seqs).most_common(1)[0][1]
        feats["seq_top_share"] = _safe_div(float(most), float(len(seqs)))
    else:
        feats["seq_top_share"] = 0.0
    # distinct bet-size diversity: few distinct sizes across many bets -> scripted
    feats["amount_diversity"] = _safe_div(
        float(len({round(a, 2) for a in all_amounts})), float(max(1, len(all_amounts)))
    )
    feats["mean_distinct_amounts"] = _safe_div(
        sum(h["distinct_amount_vals"] for h in hands), n_hands
    )

    # ---- context / robustness ----
    feats["n_hands"] = n_hands
    feats["mean_action_count"] = _safe_div(pooled_actions, n_hands)
    feats["empty_actions_share"] = _safe_div(sum(h["empty_actions"] for h in hands), n_hands)
    feats["empty_players_share"] = _safe_div(sum(h["empty_players"] for h in hands), n_hands)
    feats["actions_per_player"] = _safe_div(
        pooled_actions, max(1.0, sum(h["n_players"] for h in hands))
    )

    # guard: make sure every declared feature exists and is finite
    out: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        out[name] = _f(feats.get(name, 0.0))
    return out


def _build_feature_names() -> List[str]:
    names: List[str] = [
        "aggression_rate",
        "check_rate",
        "call_rate",
        "bet_rate",
        "raise_rate",
        "fold_rate",
        "passive_rate",
        "aggression_factor",
        "zero_amount_share",
        "action_entropy",
        "raise_to_share",
        "call_to_share",
    ]
    for prefix in (
        "action_count",
        "street_count",
        "street_depth",
        "player_count",
        "hero_seat",
        "starting_stack",
        "pot_growth",
        "bet_size_bb",
    ):
        for suffix in ("mean", "std", "min", "max", "p25", "p50", "p75"):
            names.append(f"{prefix}_{suffix}")
    names += [
        "reach_flop_rate",
        "reach_turn_rate",
        "reach_river_rate",
        "seq_unique_ratio",
        "seq_top_share",
        "amount_diversity",
        "mean_distinct_amounts",
        "n_hands",
        "mean_action_count",
        "empty_actions_share",
        "empty_players_share",
        "actions_per_player",
    ]
    return names


FEATURE_NAMES: List[str] = _build_feature_names()


def vectorize(features: Dict[str, float]) -> List[float]:
    """Turn a feature dict into a list ordered by ``FEATURE_NAMES``."""
    return [_f(features.get(name, 0.0)) for name in FEATURE_NAMES]


def transform_chunks(chunks: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
    """Vectorize a list of chunks into a (n_chunks, n_features) float array."""
    if not chunks:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    rows = [vectorize(extract_chunk_features(chunk)) for chunk in chunks]
    return np.asarray(rows, dtype=float)
