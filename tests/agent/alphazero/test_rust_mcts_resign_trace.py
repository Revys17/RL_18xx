"""Phase 1 PlayoutTrace + Phase 2 consensus-resign parity on the Rust MCTS path.

Confirms ``RustMCTSPlayer`` (the Python adapter around the Rust tree)
mirrors the Python ``MCTSPlayer`` behavior for both features end-to-end.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero.config import SelfPlayConfig, TraceConfig
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE, PlayoutTrace

pytest.importorskip("engine_rs")
from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer  # noqa: E402


class DummyNet:
    """Uniform prior + zero value stub model."""

    def encoder_type(self) -> str:
        return "GNN"

    def run_encoded(self, encoded_game_state):
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        return priors, torch.log(priors), value

    def run_many_encoded(self, encoded_game_states):
        n = len(encoded_game_states)
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        return [priors] * n, [torch.log(priors)] * n, [value] * n


def _make_player(**overrides) -> RustMCTSPlayer:
    defaults = dict(
        network=DummyNet(),
        use_rust_mcts=True,
        num_readouts=8,
        parallel_readouts=2,
        min_readouts=2,
        backup_discount=1.0,
        use_score_values=False,
        use_fp16_inference=False,
        enable_resign=True,
        resign_window=3,
        resign_high_threshold=0.65,
        resign_gap_threshold=0.30,
        noresign_holdout_rate=0.0,
        resign_high_threshold_min=0.45,
        dirichlet_noise_weight=0.0,
    )
    defaults.update(overrides)
    return RustMCTSPlayer(SelfPlayConfig(**defaults))


# --------------------------- Resign -----------------------------------------


def _inject_q_window(player: RustMCTSPlayer, vectors: list[np.ndarray]):
    """Replace the player's rolling Q window so ``check_resign`` reads a known
    history. The next ``check_resign`` call also appends the live root Q, so
    we leave one slot of headroom (size = resign_window - 1)."""
    cap = player._q_window.maxlen
    needed = max(0, cap - 1)
    player._q_window = deque(vectors[-needed:], maxlen=cap) if needed else deque(maxlen=cap)


def _stub_root_q(player: RustMCTSPlayer, q: np.ndarray):
    """Monkey-patch the Python-side ``_root_q_vector_for_resign`` helper so
    tests can drive ``check_resign`` deterministically without running MCTS.
    (The PyO3 Rust object doesn't allow attribute injection, so we go through
    the Python-resident indirection.)"""
    arr = [float(x) for x in q]
    player._root_q_vector_for_resign = lambda: arr  # type: ignore


def test_rust_check_resign_disabled_returns_false():
    p = _make_player(enable_resign=False)
    assert p.check_resign() == (False, None)


def test_rust_check_resign_window_not_full_no_fire():
    """Window=3; only 1 vector recorded -> should not fire."""
    p = _make_player(resign_window=3)
    _stub_root_q(p, np.array([0.9, 0.05, 0.0, 0.05], dtype=np.float32))
    assert p.check_resign() == (False, None)


def test_rust_check_resign_below_threshold_no_fire():
    p = _make_player(resign_window=3, resign_high_threshold=0.65)
    _stub_root_q(p, np.array([0.55, 0.20, 0.15, 0.10], dtype=np.float32))
    _inject_q_window(
        p, [np.array([0.55, 0.20, 0.15, 0.10], dtype=np.float32)] * 2
    )
    assert p.check_resign() == (False, None)


def test_rust_check_resign_below_gap_no_fire():
    p = _make_player(resign_window=3, resign_high_threshold=0.65, resign_gap_threshold=0.30)
    _stub_root_q(p, np.array([0.70, 0.50, 0.10, 0.10], dtype=np.float32))
    _inject_q_window(
        p, [np.array([0.70, 0.50, 0.10, 0.10], dtype=np.float32)] * 2
    )
    assert p.check_resign() == (False, None)


def test_rust_check_resign_unstable_leader_no_fire():
    p = _make_player(resign_window=3)
    _stub_root_q(p, np.array([0.9, 0.05, 0.0, 0.05], dtype=np.float32))
    _inject_q_window(
        p,
        [
            np.array([0.9, 0.05, 0.0, 0.05], dtype=np.float32),
            # Leader flips in this window slot.
            np.array([0.05, 0.9, 0.0, 0.05], dtype=np.float32),
        ],
    )
    should, info = p.check_resign()
    assert should is False
    assert info is None


def test_rust_check_resign_triggers_with_stable_decisive_leader():
    p = _make_player(resign_window=3, resign_high_threshold=0.65, resign_gap_threshold=0.30)
    decisive = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(p, decisive)
    _inject_q_window(p, [decisive] * 2)
    should, info = p.check_resign()
    assert should is True
    assert info is not None
    assert info["leader"] == 0
    assert info["q_leader_min"] == pytest.approx(0.80)
    assert info["gap_min"] == pytest.approx(0.70)


def test_rust_check_resign_holdout_suppresses_but_records():
    p = _make_player(resign_window=3)
    p._noresign_holdout = True
    decisive = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(p, decisive)
    _inject_q_window(p, [decisive] * 2)
    should, info = p.check_resign()
    assert should is False  # holdout never actually resigns
    assert info is not None  # but the would-have-resigned info is reported
    assert p._would_have_resigned_info is not None
    assert p._would_have_resigned_info["leader"] == 0


def test_rust_would_have_resigned_recorded_only_once():
    p = _make_player(resign_window=3)
    decisive_a = np.array([0.80, 0.10, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(p, decisive_a)
    _inject_q_window(p, [decisive_a] * 2)
    p.check_resign()
    first = dict(p._would_have_resigned_info)

    decisive_b = np.array([0.85, 0.05, 0.05, 0.05], dtype=np.float32)
    _stub_root_q(p, decisive_b)
    _inject_q_window(p, [decisive_b] * 2)
    p.check_resign()
    assert p._would_have_resigned_info == first


def test_rust_root_q_vector_matches_w_over_n_plus_1():
    """The Rust accessor implements Q = root_w / (1 + root_n) — sanity-check
    that after a handful of readouts the per-player Q is consistent with
    backed-up values from the DummyNet (zero value vector → Q stays 0)."""
    p = _make_player()
    for _ in range(3):
        p.tree_search(parallel_readouts=2)
    q = p._rust_player.root_q_vector()
    num_players = len(p._rust_player.root_game_object().players)
    assert len(q) == num_players
    # DummyNet's zero value vector means Q stays at zero regardless of N.
    assert all(abs(v) < 1e-5 for v in q)


# ------------------------------ Tracing -------------------------------------


def test_rust_tracing_disabled_by_default():
    p = _make_player()
    assert p._tracing_enabled is False
    assert p.traces == []
    p.tree_search(parallel_readouts=2)
    assert p.traces == []


def test_rust_tracing_enabled_records_traces():
    trace_cfg = TraceConfig(trace_game_rate=1.0, trace_every_n_moves=1, traces_per_move=4)
    p = _make_player(trace=trace_cfg)
    assert p._tracing_enabled is True
    p.tree_search(parallel_readouts=4)
    assert len(p.traces) > 0, "tracing was on but no traces captured"
    for tr in p.traces:
        assert isinstance(tr, PlayoutTrace)
        assert tr.move_idx == 0
        # action_path / pw_grandchild_path / forced_chain_lengths are parallel.
        assert len(tr.action_path) == tr.leaf_depth
        assert len(tr.pw_grandchild_path) == tr.leaf_depth
        assert len(tr.forced_chain_lengths) == tr.leaf_depth
        assert tr.nn_value is not None
        assert tr.nn_value.shape == (VALUE_SIZE,)


def test_rust_traces_per_move_budget_respected():
    trace_cfg = TraceConfig(trace_game_rate=1.0, trace_every_n_moves=1, traces_per_move=3)
    p = _make_player(trace=trace_cfg)
    p.tree_search(parallel_readouts=4)
    first_count = len([t for t in p.traces if t.move_idx == 0])
    assert first_count <= 3
    p.tree_search(parallel_readouts=4)
    total = len([t for t in p.traces if t.move_idx == 0])
    assert total <= 3, f"expected <=3 traces for move 0, got {total}"


def test_rust_dump_traces_writes_jsonl(tmp_path: Path):
    trace_cfg = TraceConfig(trace_game_rate=1.0, trace_every_n_moves=1, traces_per_move=4)
    p = _make_player(trace=trace_cfg)
    p.tree_search(parallel_readouts=4)
    p.tree_search(parallel_readouts=4)

    out = p.dump_traces(output_dir=tmp_path)
    assert out is not None
    assert out.exists()
    assert out.parent.name == str(int(p.config.global_step))

    lines = out.read_text().strip().split("\n")
    header = json.loads(lines[0])
    assert header["kind"] == "header"
    assert header["num_traces"] == len(p.traces)
    assert header["engine"] == "rust"
    assert "players" in header and len(header["players"]) > 0

    trace_lines = [json.loads(ln) for ln in lines[1:]]
    assert len(trace_lines) == len(p.traces)
    for entry in trace_lines:
        assert "action_path" in entry
        assert "pw_grandchild_path" in entry
        assert "forced_chain_lengths" in entry
        assert "leaf_q_perspective" in entry
        assert "nn_value" in entry
        if entry["nn_value"] is not None:
            assert len(entry["nn_value"]) == VALUE_SIZE


def test_rust_dump_traces_noop_when_tracing_off(tmp_path: Path):
    p = _make_player()
    p.tree_search(parallel_readouts=4)
    assert p.traces == []
    assert p.dump_traces(output_dir=tmp_path) is None


def test_rust_pw_grandchild_marker_matches_truth():
    """When tracing is on AND PW slots are present, ``pw_grandchild_path[i]``
    must be True iff the step's action descended into a price-bearing slot.
    Auction state is dominated by Bid (a PW slot) so we should see at least
    one True marker after some readouts.
    """
    trace_cfg = TraceConfig(trace_game_rate=1.0, trace_every_n_moves=1, traces_per_move=8)
    p = _make_player(trace=trace_cfg)
    # Drive several readouts so the tree has descents to record.
    for _ in range(4):
        p.tree_search(parallel_readouts=4)
    assert p.traces, "expected at least one trace recorded"
    # At least one step on any trace should have descended through PW (the
    # initial Bid slots are all price-bearing).
    any_pw = any(any(tr.pw_grandchild_path) for tr in p.traces)
    assert any_pw, "expected at least one PW grandchild descent on the Rust path"
