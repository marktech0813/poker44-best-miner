"""Shared test fixtures: synthetic chunks shaped like the real benchmark payload."""

from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest


def _make_hand(rng: random.Random, *, botlike: bool) -> Dict[str, Any]:
    n_players = rng.choice([4, 5, 6])
    players = [
        {
            "player_uid": f"seat_{i}",
            "seat": i,
            "starting_stack": round(rng.uniform(1.0, 6.0), 4),
            "hole_cards": None,
            "showed_hand": False,
        }
        for i in range(1, n_players + 1)
    ]
    streets = ["preflop", "flop", "turn", "river"]
    depth = rng.randint(1, 4) if not botlike else rng.randint(1, 2)
    actions: List[Dict[str, Any]] = []
    for k in range(rng.randint(3, 8)):
        street = streets[min(depth - 1, rng.randint(0, depth - 1))]
        if botlike:
            atype = rng.choice(["fold", "call", "check", "raise"])
            amt = rng.choice([2.0, 2.0, 3.0])  # rigid sizing
        else:
            atype = rng.choice(["fold", "call", "check", "bet", "raise"])
            amt = round(rng.uniform(1.0, 30.0), 2)
        amount_bb = amt if atype in ("bet", "raise") else 0.0
        actions.append(
            {
                "action_id": str(k + 1),
                "street": street,
                "actor_seat": rng.randint(1, n_players),
                "action_type": atype,
                "amount": round(amount_bb * 0.02, 4),
                "raise_to": None,
                "call_to": None,
                "normalized_amount_bb": amount_bb,
                "pot_before": round(rng.uniform(0.1, 1.0), 4),
                "pot_after": round(rng.uniform(1.0, 2.0), 4),
            }
        )
    return {
        "hand_id": f"h{rng.random()}",
        "metadata": {
            "game_type": "Hold'em",
            "limit_type": "No Limit",
            "max_seats": 6,
            "hero_seat": rng.randint(1, n_players),
            "sb": 0.01,
            "bb": 0.02,
            "ante": 0.0,
        },
        "players": players,
        "streets": [],
        "actions": actions,
        "outcome": {"winners": [], "payouts": {}, "total_pot": 0.0, "showdown": False},
    }


def _make_chunk(seed: int, botlike: bool, n_hands: int = 30) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [_make_hand(rng, botlike=botlike) for _ in range(n_hands)]


@pytest.fixture
def human_chunk() -> List[Dict[str, Any]]:
    return _make_chunk(seed=1, botlike=False)


@pytest.fixture
def bot_chunk() -> List[Dict[str, Any]]:
    return _make_chunk(seed=2, botlike=True)


@pytest.fixture
def mixed_chunks():
    chunks = []
    labels = []
    for i in range(20):
        botlike = i % 2 == 0
        chunks.append(_make_chunk(seed=100 + i, botlike=botlike))
        labels.append(1 if botlike else 0)
    return chunks, labels
