"""Auction learning (search_learning ``auction_only``): episodes are the waterfall auction
alone, scored by a frozen network's estimate of the first Stock Round position."""

import json
import math

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero.lockstep import ModelEvaluator  # noqa: E402

from .search_test_utils import HashNet  # noqa: E402
from .test_search_learning import _save_checkpoints  # noqa: E402


def _settings(**overrides):
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="v", learner_temperature=1.0, prior_temperature=1.0,
        n_full=6, p_full=1.0, games_per_worker=3, max_decisions=200, parallel_readouts=4,
        auction_only=True, terminal_value="t", full_move_temperature=1.0,
    )  # fmt: skip
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return sl.worker_settings(cfg, version=3)


def _generate(settings, seconds=8.0):
    gen = sl.Generator(settings, {"learner": ModelEvaluator(HashNet())})
    return gen.run(seconds)


def test_episodes_are_the_auction_alone_and_end_at_the_first_stock_round():
    out = _generate(_settings())
    games = [g for g in out["games"] if g["termination"] == "auction_end"]
    assert games, out["stats"]
    for g in games:
        assert g["start_kind"] == "auction" and g["rows"] and g["searched"] == len(g["rows"])
        state, nodes, rotation, players = g["terminal"]
        assert state.dtype == np.float32 and nodes.ndim == 2 and players == 4 and 0 <= rotation < 4
        # Every private sold (the SV maybe handed over at $0), each at its face value or more.
        sales = g["auction"]
        assert set(sales) - {"SV"} == {"CS", "DH", "MH", "CA", "BO"}
        for company, (buyer, price) in sales.items():
            assert 1 <= buyer <= 4 and (price >= sl.AUCTION_FACE[company] or company == "SV")
        for row in g["rows"]:
            assert row.critic and row.win_target is None  # the trainer scores them
            assert len(row.legal) > 1 and float(row.target.sum()) == pytest.approx(1.0, abs=1e-5)


def test_a_stalled_auction_is_dropped():
    out = _generate(_settings(auction_max_decisions=2), seconds=4.0)
    stalled = [g for g in out["games"] if g["termination"] == "auction_stall"]
    assert stalled and out["stats"]["auction_stalls"] == len(stalled)
    assert all(g["rows"] == [] and g["terminal"] is None for g in stalled)


class _FixedValue(torch.nn.Module):
    """A value network whose win head says (0.4, 0.3, 0.2, 0.1) in the mover's frame."""

    def forward(self, states, nodes, value_only=True):
        p = torch.tensor([0.4, 0.3, 0.2, 0.1, 1e-9, 1e-9]).log().expand(states.shape[0], 6)
        return None, p, None, None


def test_outcomes_are_the_terminal_networks_estimate_in_player_order():
    out = _generate(_settings())
    games = [g for g in out["games"] if g["terminal"] is not None]
    sl.auction_outcomes(games, _FixedValue(), "cpu")
    for g in games:
        rotation = g["terminal"][2]
        expected = np.roll([0.4, 0.3, 0.2, 0.1], rotation)  # the mover is seat ``rotation``
        assert np.allclose(g["win_share"], expected, atol=1e-5) and g["fractions"] == g["win_share"]
        for row in g["rows"]:
            assert np.allclose(row.win_target, expected, atol=1e-5) and row.score_target is None
        assert g["root_value_ce"][1] == len(g["rows"])


def test_the_auction_options_are_checked():
    cfg = sl.SearchLearningConfig(policy_checkpoint="p", value_checkpoint="v", auction_only=True)
    with pytest.raises(SystemExit, match="terminal-value"):
        sl._validate(cfg)
    cfg.terminal_value = "t"
    cfg.save_data = "somewhere"
    with pytest.raises(SystemExit, match="save-data"):
        sl._validate(cfg)


def test_a_tiny_cpu_auction_run_trains_on_scored_auctions(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    policy, value = _save_checkpoints(tmp_path)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint=str(policy), value_checkpoint=str(value), out_dir=str(tmp_path / "runs"), run_name="auc",
        p_full=1.0, n_full=4, parallel_readouts=2, workers=2, games_per_worker=2, task_seconds=3.0,
        max_decisions=200, inference_batch_size=64, window_rows=400, rows_per_update=10, min_window_rows=10,
        steps_per_update=2, minibatch=8, microbatch=4, max_updates=2, snapshot_every=1, anchor_every=0,
        kl_target=0.05, auction_only=True, terminal_value=str(value), score_loss_weight=0.0,
    )  # fmt: skip
    run_dir = sl.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    for h in history:
        assert math.isfinite(h["ce"]) and math.isfinite(h["critic_ce"]) and h["rows_trained"] == 16
        assert set(h["terminations"]) <= {"auction_end", "auction_stall"} and h["starts"] == {"auction": h["games"]}
        assert h["auction_prices"] and all(p >= sl.AUCTION_FACE[c] for c, p in h["auction_prices"].items() if c != "SV")
