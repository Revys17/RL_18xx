"""Tests for Phase 2 multiplayer consensus resign (mcts.py + self_play.py + loop.py).

Covers:
- ``MCTSPlayer.check_resign`` window/threshold logic
- Holdout sampling (``noresign_holdout``) suppresses resign decisions
- Termination metadata is set correctly on game end
- ``loop.calibrate_resign_threshold`` tightens / loosens / clamps as specified
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero.config import SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE, MCTSNode
from rl18xx.agent.alphazero.self_play import MCTSPlayer
from rl18xx.agent.alphazero import loop as loop_mod


@pytest.fixture(autouse=True)
def restore_mcts_node_q():
    """Several tests below stub ``MCTSNode.Q`` to a fixed property so they
    can drive ``check_resign`` with known Q vectors. Save and restore the
    original descriptor so the class-level patch doesn't leak into other
    tests run in the same pytest session.
    """
    original_q = MCTSNode.__dict__["Q"]
    try:
        yield
    finally:
        # ``property`` lives in the class dict; reassign directly to avoid
        # the data-descriptor protection on __setattr__.
        type.__setattr__(MCTSNode, "Q", original_q)


class DummyNet:
    def encoder_type(self):
        return "GNN"

    def run_encoded(self, encoded_game_state):
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        return priors, torch.log(priors), value

    def run_many_encoded(self, encoded_game_states):
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        n = len(encoded_game_states)
        return [priors] * n, [torch.log(priors)] * n, [value] * n


def _make_player(**hyperparam_overrides) -> MCTSPlayer:
    defaults = dict(
        network=DummyNet(),
        use_score_values=False,
        backup_discount=1.0,
        enable_resign=True,
        resign_window=3,
        resign_high_threshold=0.65,
        resign_gap_threshold=0.30,
        noresign_holdout_rate=0.0,
        resign_high_threshold_min=0.45,
    )
    defaults.update(hyperparam_overrides)
    return MCTSPlayer(SelfPlayConfig(**defaults))


# ----------------------------- check_resign --------------------------------


def _inject_q_window(player: MCTSPlayer, q_vectors: list[np.ndarray]):
    """Push raw Q vectors into the player's rolling window for tests that
    want full control of the consensus history."""
    player._q_window = deque(q_vectors, maxlen=player.config.resign_window)


def _stub_root_q(player: MCTSPlayer, q: np.ndarray):
    """Replace ``player.root.Q`` with a fixed array so ``check_resign`` reads
    a known per-player Q vector instead of the empty MCTS state."""
    full = np.zeros(VALUE_SIZE, dtype=np.float32)
    full[: len(q)] = q
    type(player.root).Q = property(lambda self: full)


def test_check_resign_disabled_returns_false():
    player = _make_player(enable_resign=False)
    should, info = player.check_resign()
    assert should is False
    assert info is None


def test_check_resign_window_not_full_returns_false():
    """First few moves shouldn't trigger — the window has to fill first."""
    player = _make_player(resign_window=3)
    _stub_root_q(player, np.array([0.9, 0.05, 0.05, 0.0], dtype=np.float32))
    # Push 2 entries (window not full yet).
    should, info = player.check_resign()
    assert should is False
    should, info = player.check_resign()
    assert should is False


def test_check_resign_unstable_leader_returns_false():
    """argmax must be the same player across the full window."""
    player = _make_player(resign_window=3)
    _stub_root_q(player, np.array([0.9, 0.05, 0.0, 0.05], dtype=np.float32))
    _inject_q_window(
        player,
        [
            np.array([0.9, 0.05, 0.0, 0.05], dtype=np.float32),
            np.array([0.05, 0.9, 0.0, 0.05], dtype=np.float32),  # leader flips
        ],
    )
    should, info = player.check_resign()
    assert should is False


def test_check_resign_below_high_threshold():
    """Even with a stable leader and good gap, ``Q_leader < high_threshold``
    should not trigger."""
    player = _make_player(resign_window=3, resign_high_threshold=0.65)
    _stub_root_q(player, np.array([0.55, 0.20, 0.15, 0.10], dtype=np.float32))
    _inject_q_window(
        player,
        [np.array([0.55, 0.20, 0.15, 0.10], dtype=np.float32)] * 2,
    )
    should, info = player.check_resign()
    assert should is False


def test_check_resign_below_gap_threshold():
    """Stable leader above high threshold but gap too narrow → no resign."""
    player = _make_player(resign_window=3, resign_high_threshold=0.65,
                          resign_gap_threshold=0.30)
    _stub_root_q(player, np.array([0.70, 0.50, 0.10, 0.10], dtype=np.float32))
    _inject_q_window(
        player,
        [np.array([0.70, 0.50, 0.10, 0.10], dtype=np.float32)] * 2,
    )
    should, info = player.check_resign()
    # gap = 0.20 < 0.30 → no resign
    assert should is False


def test_check_resign_triggers_when_conditions_hold():
    player = _make_player(resign_window=3, resign_high_threshold=0.65,
                          resign_gap_threshold=0.30)
    decisive = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(player, decisive)
    _inject_q_window(player, [decisive] * 2)
    should, info = player.check_resign()
    assert should is True
    assert info is not None
    assert info["leader"] == 0
    assert info["q_leader_min"] == pytest.approx(0.80)
    assert info["gap_min"] == pytest.approx(0.70)


def test_check_resign_holdout_suppresses_resign_but_records():
    """Holdout games don't actually resign but DO stash the would-have-resigned
    info for the loop calibrator."""
    player = _make_player(resign_window=3)
    player._noresign_holdout = True
    decisive = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(player, decisive)
    _inject_q_window(player, [decisive] * 2)
    should, info = player.check_resign()
    assert should is False
    assert info is not None  # info is still returned so caller can log
    assert player._would_have_resigned_info is not None
    assert player._would_have_resigned_info["leader"] == 0


def test_would_have_resigned_recorded_only_once():
    """The first triggering moment is stashed; later triggers don't overwrite."""
    player = _make_player(resign_window=3)
    decisive_a = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(player, decisive_a)
    _inject_q_window(player, [decisive_a] * 2)
    player.check_resign()
    first = dict(player._would_have_resigned_info)

    decisive_b = np.array([0.85, 0.05, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(player, decisive_b)
    _inject_q_window(player, [decisive_b] * 2)
    player.check_resign()
    assert player._would_have_resigned_info == first


# ----------------------- calibrate_resign_threshold ------------------------


def _write_holdout_game(
    tmpdir: Path,
    *,
    loop: int,
    game_idx: int,
    holdout: bool,
    would_have_resigned: dict | None,
    result_per_player: list[float],
    termination: str = "finished",
):
    file = tmpdir / f"L{loop}_G{game_idx}.json"
    data = {
        "loop_number": loop,
        "game_number": game_idx,
        "status": "Completed",
        "noresign_holdout": holdout,
        "would_have_resigned": would_have_resigned,
        "result_per_player": result_per_player,
        "termination": termination,
    }
    file.write_text(json.dumps(data))


def test_calibration_tightens_on_high_fp_rate(tmp_path, monkeypatch):
    """fp_rate > 5% should bump the threshold up by 0.05."""
    monkeypatch.setattr(loop_mod, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path)
    # 10 holdouts that would have resigned; 5 wrong → fp_rate=0.5
    for i in range(5):
        _write_holdout_game(
            tmp_path, loop=0, game_idx=i, holdout=True,
            would_have_resigned={"leader": 0, "move_number": 100},
            result_per_player=[0.9, 0.0, 0.05, 0.05],  # leader (0) IS winner
        )
    for i in range(5, 10):
        _write_holdout_game(
            tmp_path, loop=0, game_idx=i, holdout=True,
            would_have_resigned={"leader": 0, "move_number": 100},
            result_per_player=[0.1, 0.9, 0.0, 0.0],  # leader (0) is NOT winner
        )
    cfg = loop_mod.LoopConfig(
        num_loop_iterations=1, num_games_per_iteration=10, num_threads=1,
        training_config=TrainingConfig(), num_readouts=8,
        resign_high_threshold=0.65, resign_high_threshold_min=0.45,
    )
    metrics = loop_mod.LoopMetrics()
    # Don't actually persist to LOOP_CONFIG_PATH during the test.
    with mock.patch.object(loop_mod, "atomic_write_json"):
        info = loop_mod.calibrate_resign_threshold(0, cfg, metrics)
    assert info["fp_rate"] == pytest.approx(0.5)
    assert info["adjustment"] == pytest.approx(0.05)
    assert cfg.resign_high_threshold == pytest.approx(0.70)


def test_calibration_loosens_after_3_low_fp_iterations(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path)
    cfg = loop_mod.LoopConfig(
        num_loop_iterations=3, num_games_per_iteration=10, num_threads=1,
        training_config=TrainingConfig(), num_readouts=8,
        resign_high_threshold=0.65, resign_high_threshold_min=0.45,
    )
    metrics = loop_mod.LoopMetrics()
    # Pre-seed two iterations of low fp_rate; the third should trigger the loosen.
    metrics.recent_resign_fp_rates = [0.0, 0.0]
    # Write only "correct" holdouts for loop=2 (fp_rate = 0.0)
    for i in range(8):
        _write_holdout_game(
            tmp_path, loop=2, game_idx=i, holdout=True,
            would_have_resigned={"leader": 0, "move_number": 100},
            result_per_player=[0.9, 0.05, 0.0, 0.05],
        )
    with mock.patch.object(loop_mod, "atomic_write_json"):
        info = loop_mod.calibrate_resign_threshold(2, cfg, metrics)
    assert info["fp_rate"] == pytest.approx(0.0)
    assert info["adjustment"] == pytest.approx(-0.05)
    assert cfg.resign_high_threshold == pytest.approx(0.60)
    # History should be cleared after a loosen.
    assert metrics.recent_resign_fp_rates == []


def test_calibration_clamps_at_min(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path)
    cfg = loop_mod.LoopConfig(
        num_loop_iterations=1, num_games_per_iteration=10, num_threads=1,
        training_config=TrainingConfig(), num_readouts=8,
        resign_high_threshold=0.45,  # already at floor
        resign_high_threshold_min=0.45,
    )
    metrics = loop_mod.LoopMetrics()
    metrics.recent_resign_fp_rates = [0.0, 0.0]
    for i in range(5):
        _write_holdout_game(
            tmp_path, loop=0, game_idx=i, holdout=True,
            would_have_resigned={"leader": 0, "move_number": 100},
            result_per_player=[0.9, 0.05, 0.0, 0.05],
        )
    with mock.patch.object(loop_mod, "atomic_write_json"):
        info = loop_mod.calibrate_resign_threshold(0, cfg, metrics)
    # Loosen would push to 0.40 but the clamp keeps it at 0.45.
    assert cfg.resign_high_threshold == pytest.approx(0.45)


def test_calibration_holds_when_no_holdouts(tmp_path, monkeypatch):
    """No holdouts → fp_rate=None, no adjustment, threshold unchanged."""
    monkeypatch.setattr(loop_mod, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path)
    cfg = loop_mod.LoopConfig(
        num_loop_iterations=1, num_games_per_iteration=10, num_threads=1,
        training_config=TrainingConfig(), num_readouts=8,
        resign_high_threshold=0.65,
    )
    metrics = loop_mod.LoopMetrics()
    with mock.patch.object(loop_mod, "atomic_write_json"):
        info = loop_mod.calibrate_resign_threshold(0, cfg, metrics)
    assert info["fp_rate"] is None
    assert info["adjustment"] == 0.0
    assert cfg.resign_high_threshold == pytest.approx(0.65)


def test_calibration_persists_threshold_to_disk(tmp_path, monkeypatch):
    """The updated threshold must be written back to LOOP_CONFIG_PATH so the
    next iteration's workers pick it up."""
    monkeypatch.setattr(loop_mod, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path)
    config_path = tmp_path / "loop_config.json"
    monkeypatch.setattr(loop_mod, "LOOP_CONFIG_PATH", config_path)
    cfg = loop_mod.LoopConfig(
        num_loop_iterations=1, num_games_per_iteration=10, num_threads=1,
        training_config=TrainingConfig(), num_readouts=8,
        resign_high_threshold=0.65, resign_high_threshold_min=0.45,
    )
    # 5 holdouts, all wrong → fp_rate=1.0 → tighten
    for i in range(5):
        _write_holdout_game(
            tmp_path, loop=0, game_idx=i, holdout=True,
            would_have_resigned={"leader": 0, "move_number": 100},
            result_per_player=[0.1, 0.9, 0.0, 0.0],
        )
    metrics = loop_mod.LoopMetrics()
    loop_mod.calibrate_resign_threshold(0, cfg, metrics)
    assert config_path.exists()
    persisted = json.loads(config_path.read_text())
    assert persisted["resign_high_threshold"] == pytest.approx(0.70)


# --------------------- holdout sampling ---------------------------------


def test_holdout_flag_set_by_rate():
    """At rate=1.0 every game is a holdout; at rate=0.0 no game is."""
    p1 = _make_player(noresign_holdout_rate=1.0)
    assert p1._noresign_holdout is True
    p0 = _make_player(noresign_holdout_rate=0.0)
    assert p0._noresign_holdout is False


def test_termination_defaults_to_none():
    """Before any end-of-game routing, ``termination`` is None."""
    player = _make_player()
    assert player.termination is None
