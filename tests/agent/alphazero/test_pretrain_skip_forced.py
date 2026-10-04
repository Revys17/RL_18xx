"""Human-game conversion drops forced (single-legal-action) positions.

A position whose legal set has exactly one action index carries no decision:
its policy target is trivial and its value target is one more correlated copy
of the game's outcome. Self-play never records such positions (the MCTS
applies forced actions without searching), so ``convert_game_to_training_data``
skips them by default while still replaying the action. The fixture is the
anonymized, finished 2-player game ``router_d_trains.json``.
"""

import io
import json
import logging
import sys
from pathlib import Path

import lmdb
import lz4.frame
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3]))
logging.disable(logging.CRITICAL)

from rl18xx.agent.alphazero.config import TrainingConfig  # noqa: E402
from rl18xx.agent.alphazero.encoder import Encoder_Transformer  # noqa: E402
from rl18xx.agent.alphazero.pretraining import (  # noqa: E402
    _get_game_object_for_game_with_reason,
    _load_cleaned_game_via_rust,
    convert_game_to_training_data,
    convert_games_to_training_dataset,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "1830" / "router_d_trains.json"
ALL_TRAINING = TrainingConfig(pretrain_validation_percentage=0.0)


@pytest.fixture(scope="module")
def cleaned_dict():
    with open(FIXTURE) as f:
        game, reason = _get_game_object_for_game_with_reason(json.load(f), use_rust=True)
    assert reason is None
    return game.to_dict()


@pytest.fixture(scope="module")
def converted(cleaned_dict):
    """{skip_forced: (training rows, stats)} for the same cleaned game."""
    out = {}
    for skip_forced in (False, True):
        game = _load_cleaned_game_via_rust(cleaned_dict)
        assert game is not None and game.finished
        stats = {}
        train, val = convert_game_to_training_data(
            game, Encoder_Transformer(), config=ALL_TRAINING, skip_forced=skip_forced, stats=stats
        )
        assert val == []
        out[skip_forced] = (train, stats)
    return out


def _same_row(a, b):
    (state_a, legal_a, pi_a, value_a, price_a), (state_b, legal_b, pi_b, value_b, price_b) = a, b
    assert len(state_a) == len(state_b)
    for x, y in zip(state_a, state_b):
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y
    # Same smoothed one-hot over the same legal set. Which legal index carries
    # the mass isn't compared: when a human place_token / buy_train matches
    # several legal actions, the label is the first match in the Rust engine's
    # per-instance (unordered) choice list, so it varies between replays.
    same_pi = torch.equal(pi_a > 0, pi_b > 0) and torch.equal(pi_a.sort().values, pi_b.sort().values)
    return legal_a == legal_b and same_pi and torch.equal(value_a, value_b) and price_a == price_b


def test_forced_positions_are_dropped_and_decisions_kept(converted):
    all_rows, all_stats = converted[False]
    kept_rows, kept_stats = converted[True]
    forced = [row for row in all_rows if len(row[1]) == 1]
    decisions = [row for row in all_rows if len(row[1]) > 1]
    assert forced and decisions

    assert all(len(row[1]) > 1 for row in kept_rows)
    assert len(kept_rows) == len(decisions)
    assert all(_same_row(a, b) for a, b in zip(kept_rows, decisions))

    assert all_stats == {"positions": len(all_rows), "forced_skipped": 0}
    assert kept_stats == {"positions": len(kept_rows), "forced_skipped": len(forced)}


def test_value_target_is_unchanged(converted):
    (all_rows, _), (kept_rows, _) = converted[False], converted[True]
    value = all_rows[0][3]
    assert torch.isclose(value.sum(), torch.tensor(1.0))
    assert all(torch.equal(row[3], value) for row in all_rows + kept_rows)


def _read_lmdb(path):
    env = lmdb.open(str(path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            return [torch.load(io.BytesIO(lz4.frame.decompress(v)), weights_only=False) for _, v in txn.cursor()]
    finally:
        env.close()


@pytest.mark.parametrize("skip_forced", [True, False], ids=["skip", "keep"])
def test_dataset_conversion_threads_skip_forced(tmp_path, cleaned_dict, converted, skip_forced):
    games = tmp_path / "games"
    games.mkdir()
    (games / "234317.json").write_text(json.dumps(cleaned_dict))
    out = tmp_path / "lmdb"
    convert_games_to_training_dataset(
        str(games), Encoder_Transformer(), out, config=ALL_TRAINING, skip_forced=skip_forced
    )
    rows = _read_lmdb(out / "training")
    assert [row[1] for row in rows] == [row[1] for row in converted[skip_forced][0]]
