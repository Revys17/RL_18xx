"""Tests for Phase 1 PlayoutTrace instrumentation (mcts.py + self_play.py).

These tests run with the GNN-stub network so a CPU-only environment can
exercise the full MCTS path. They do not depend on a CUDA GPU.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero.config import SelfPlayConfig, TraceConfig
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE, MCTSNode, PlayoutTrace
from rl18xx.agent.alphazero.self_play import MCTSPlayer


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


def _make_player(traces_per_move: int = 4) -> MCTSPlayer:
    trace_cfg = TraceConfig(
        trace_game_rate=1.0,
        trace_every_n_moves=1,
        traces_per_move=traces_per_move,
    )
    cfg = SelfPlayConfig(
        network=DummyNet(),
        use_score_values=False,
        backup_discount=1.0,
        trace=trace_cfg,
    )
    return MCTSPlayer(cfg)


def _expand_root(player: MCTSPlayer):
    """Replicate SelfPlay.play()'s first-node expansion (no tracing through it)."""
    first = player.root.select_leaf()
    first.ensure_encoded()
    probs, _, val = player.config.network.run_encoded(first.encoded_game_state)
    first.incorporate_results(probs, val, up_to=player.root)


def test_tracing_enabled_with_rate_1():
    player = _make_player()
    assert player._tracing_enabled is True
    assert player.traces == []


def test_tracing_disabled_by_default():
    cfg = SelfPlayConfig(network=DummyNet(), use_score_values=False, backup_discount=1.0)
    player = MCTSPlayer(cfg)
    assert player._tracing_enabled is False


def test_tree_search_populates_traces():
    player = _make_player(traces_per_move=4)
    _expand_root(player)

    player.tree_search(parallel_readouts=4)
    assert len(player.traces) > 0, "tracing was on but no traces captured"
    assert len(player.traces) <= 4

    for tr in player.traces:
        assert isinstance(tr, PlayoutTrace)
        assert tr.move_idx == 0
        # action_path / pw_grandchild_path / forced_chain_lengths are parallel.
        assert len(tr.action_path) == tr.leaf_depth
        assert len(tr.pw_grandchild_path) == tr.leaf_depth
        assert len(tr.forced_chain_lengths) == tr.leaf_depth
        # nn_value is the network output; value head emits VALUE_SIZE entries.
        assert tr.nn_value is not None
        assert tr.nn_value.shape == (VALUE_SIZE,)


def test_pw_grandchild_marker_matches_price_range():
    """``pw_grandchild_path[i]`` is True iff the corresponding step descended
    into a price-bearing slot (non-degenerate price range) on the parent."""
    player = _make_player()
    _expand_root(player)
    # A few searches so we have a variety of action_paths to validate.
    for _ in range(4):
        player.tree_search(parallel_readouts=4)

    assert player.traces, "no traces to inspect"

    # Walk each trace and verify the PW marker matches the parent's
    # ``price_ranges_by_idx`` for the action taken. Reconstruction walks the
    # tree from the player's root.
    for tr in player.traces:
        node = player.root
        for step_idx, action in enumerate(tr.action_path):
            price_range = node.price_ranges_by_idx.get(action)
            is_pw_slot = price_range is not None and price_range[0] != price_range[1]
            assert tr.pw_grandchild_path[step_idx] == is_pw_slot, (
                f"pw_grandchild_path mismatch at step {step_idx} of trace "
                f"action={action}, expected {is_pw_slot}, got "
                f"{tr.pw_grandchild_path[step_idx]}"
            )
            # Descend to the chosen child for the next step.
            if is_pw_slot:
                grandchildren = node.price_children.get(action, {})
                if not grandchildren:
                    break
                # Pick any child whose chain prefix matches the remainder
                # of the trace; with a single-step lookahead we just take
                # the first available grandchild.
                node = next(iter(grandchildren.values()))
            else:
                child = node.children.get(action)
                if child is None:
                    break
                node = child


def test_most_visited_child_appears_in_some_trace():
    """At least one playout's first step should match the most-visited child of
    the root after the search completes — i.e. MCTS actually descended along the
    eventually-most-visited path at some point during the playout sequence."""
    player = _make_player(traces_per_move=4)
    _expand_root(player)
    for _ in range(8):
        player.tree_search(parallel_readouts=4)

    most_visited_path = player.root.most_visited_path_nodes()
    if not most_visited_path:
        pytest.skip("MCTS produced no fully-expanded categorical child to follow")

    target_fmove = most_visited_path[0].fmove

    matched = [
        tr for tr in player.traces if tr.action_path and tr.action_path[0] == target_fmove
    ]
    assert matched, (
        f"no trace's action_path started with the most-visited child fmove="
        f"{target_fmove}; saw firsts="
        f"{sorted({tr.action_path[0] for tr in player.traces if tr.action_path})}"
    )


def test_dump_traces_writes_jsonl(tmp_path: Path):
    player = _make_player()
    _expand_root(player)
    player.tree_search(parallel_readouts=4)
    player.tree_search(parallel_readouts=4)

    out = player.dump_traces(output_dir=tmp_path)
    assert out is not None
    assert out.exists()
    assert out.parent.name == str(int(player.config.global_step))

    lines = out.read_text().strip().split("\n")
    header = json.loads(lines[0])
    assert header["kind"] == "header"
    assert header["num_traces"] == len(player.traces)
    assert header["game_id"] == str(player.config.game_id)
    assert "players" in header and len(header["players"]) > 0

    trace_lines = [json.loads(ln) for ln in lines[1:]]
    assert len(trace_lines) == len(player.traces)
    for entry in trace_lines:
        assert "action_path" in entry
        assert "pw_grandchild_path" in entry
        assert "forced_chain_lengths" in entry
        assert "leaf_q_perspective" in entry
        assert "nn_value" in entry
        if entry["nn_value"] is not None:
            assert len(entry["nn_value"]) == VALUE_SIZE


def test_dump_traces_noop_when_tracing_off(tmp_path: Path):
    """When tracing is disabled the dump should be a no-op (return None)."""
    cfg = SelfPlayConfig(network=DummyNet(), use_score_values=False, backup_discount=1.0)
    player = MCTSPlayer(cfg)
    # Run a search to confirm normal path is unaffected.
    _expand_root(player)
    player.tree_search(parallel_readouts=4)
    assert player.traces == []
    assert player.dump_traces(output_dir=tmp_path) is None


def test_traces_per_move_cap_respected_across_calls():
    """``traces_per_move`` is a per-move budget enforced across all
    ``tree_search`` calls until ``play_move`` resets it."""
    player = _make_player(traces_per_move=3)
    _expand_root(player)
    player.tree_search(parallel_readouts=4)
    first_count = len([t for t in player.traces if t.move_idx == 0])
    assert first_count <= 3

    # Second search at the same move should not add more than the remaining
    # budget (so total still <= 3).
    player.tree_search(parallel_readouts=4)
    total_count = len([t for t in player.traces if t.move_idx == 0])
    assert total_count <= 3, f"expected <=3 traces for move 0, got {total_count}"
