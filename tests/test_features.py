"""Feature-extraction contract tests."""

from __future__ import annotations

import math

from p44miner.features import (
    FEATURE_NAMES,
    extract_chunk_features,
    transform_chunks,
    vectorize,
)


def test_feature_names_unique_and_nonempty():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert len(FEATURE_NAMES) > 20


def test_empty_chunk_is_neutral():
    feats = extract_chunk_features([])
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert all(v == 0.0 for v in feats.values())


def test_malformed_input_never_raises():
    for bad in [None, [None], [{}], [{"actions": "nope"}], [{"players": 5}], "garbage"]:
        feats = extract_chunk_features(bad)  # type: ignore[arg-type]
        assert len(feats) == len(FEATURE_NAMES)
        assert all(math.isfinite(v) for v in feats.values())


def test_vectorize_order_matches_schema(human_chunk):
    feats = extract_chunk_features(human_chunk)
    vec = vectorize(feats)
    assert len(vec) == len(FEATURE_NAMES)
    for name, value in zip(FEATURE_NAMES, vec):
        assert value == feats[name]


def test_transform_chunks_shape(mixed_chunks):
    chunks, _ = mixed_chunks
    X = transform_chunks(chunks)
    assert X.shape == (len(chunks), len(FEATURE_NAMES))
    assert math.isfinite(float(X.sum()))


def test_bot_and_human_differ(bot_chunk, human_chunk):
    fb = extract_chunk_features(bot_chunk)
    fh = extract_chunk_features(human_chunk)
    # rigid bot sizing should reduce bet-size dispersion vs human
    assert fb["bet_size_bb_std"] <= fh["bet_size_bb_std"] + 1e-9
