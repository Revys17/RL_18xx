"""Phase 3.5 thin-shim parity tests for ``ActionMapper.get_legal_actions_factored``.

Confirms that the Rust path (``RustGameAdapter`` -> Rust ``get_factored_choices``)
and the Python fallback path (``BaseGame`` -> ``FactoredActionHelper``) produce
identical (indices, price_ranges, action_types) outputs across a handful of
representative game states.

This is a regression-safety test: today the two paths *should* match (the
factored corpus audit covered 3243 games with 0 divergences), and any future
drift between Python `index_for_factored` and the Rust enumeration is caught
here.
"""
from __future__ import annotations

import pytest

from engine_rs import BaseGame as RustBaseGame
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter


PLAYERS = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}


def _fresh_python_game():
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    return game_class(PLAYERS)


def _fresh_rust_game():
    return RustGameAdapter(RustBaseGame(PLAYERS))


def _step_both(py_game, rust_game, action_dict_picker):
    """Apply the same action (chosen by ``action_dict_picker(py_game)``) to
    both engines so they stay in sync."""
    helper = ActionHelper()
    choices = helper.get_all_choices(py_game)
    action = action_dict_picker(py_game, choices)
    action_dict = action.to_dict() if hasattr(action, "to_dict") else action
    py_game.process_action(action_dict)
    rust_game.process_action(action_dict)


def _assert_factored_parity(py_game, rust_game, label: str):
    mapper = ActionMapper()
    py_out = mapper.get_legal_actions_factored(py_game)
    rust_out = mapper.get_legal_actions_factored(rust_game)
    py_indices, py_prices, py_types = py_out
    rust_indices, rust_prices, rust_types = rust_out

    assert py_indices == rust_indices, (
        f"[{label}] indices diverged:\n"
        f"  python: {py_indices}\n"
        f"  rust:   {rust_indices}\n"
        f"  py - rust: {sorted(set(py_indices) - set(rust_indices))}\n"
        f"  rust - py: {sorted(set(rust_indices) - set(py_indices))}"
    )
    assert py_prices == rust_prices, (
        f"[{label}] price_ranges diverged:\n"
        f"  python: {py_prices}\n"
        f"  rust:   {rust_prices}"
    )
    assert py_types == rust_types, (
        f"[{label}] action_types diverged:\n"
        f"  python: {py_types}\n"
        f"  rust:   {rust_types}"
    )


# ----- States -------------------------------------------------------------


def test_initial_auction_parity():
    """Fresh 1830 game: all legal actions are Bid (price-bearing PW slots)."""
    py_game = _fresh_python_game()
    rust_game = _fresh_rust_game()
    _assert_factored_parity(py_game, rust_game, "initial_auction")


def test_after_buy_company_in_auction():
    """A few steps into the waterfall — gets a mix of Bid + BuyCompany."""
    py_game = _fresh_python_game()
    rust_game = _fresh_rust_game()
    # Player 1 buys SV (the first index-0 choice = "buy cheapest").
    _step_both(py_game, rust_game, lambda g, c: c[0])
    _assert_factored_parity(py_game, rust_game, "after_p1_buys_sv")


def test_mid_auction_after_passes():
    """Multiple players pass — legal set still includes Bids on remaining companies."""
    py_game = _fresh_python_game()
    rust_game = _fresh_rust_game()
    # Walk a few buys/passes (mirror what the existing fixture in
    # mcts_test.py does to reach an interesting auction-mid state).
    _step_both(py_game, rust_game, lambda g, c: c[-2])  # bid on BO
    _step_both(py_game, rust_game, lambda g, c: c[0])
    _step_both(py_game, rust_game, lambda g, c: c[0])
    _step_both(py_game, rust_game, lambda g, c: c[0])
    _assert_factored_parity(py_game, rust_game, "after_bo_bid_then_buys")


def test_legal_action_indices_helper_matches():
    """``get_legal_action_indices`` is a thin wrapper — same parity expected."""
    py_game = _fresh_python_game()
    rust_game = _fresh_rust_game()
    mapper = ActionMapper()
    py_idx = mapper.get_legal_action_indices(py_game)
    rust_idx = mapper.get_legal_action_indices(rust_game)
    assert py_idx == rust_idx


def test_python_fallback_logs_once_per_process(caplog):
    """The Phase 3.5 boundary warning fires the first time the Python
    fallback engages and stays silent on later calls."""
    mapper = ActionMapper()
    # Force the warning gate fresh so any test ordering doesn't matter.
    if hasattr(mapper, "_warned_python_fallback"):
        del mapper._warned_python_fallback

    py_game = _fresh_python_game()
    import logging
    with caplog.at_level(logging.WARNING, logger="rl18xx.agent.alphazero.action_mapper"):
        mapper.get_legal_actions_factored(py_game)
        first_count = sum(
            1 for r in caplog.records if "falling back to Python" in r.message
        )
        mapper.get_legal_actions_factored(py_game)
        second_count = sum(
            1 for r in caplog.records if "falling back to Python" in r.message
        )
    assert first_count == 1, f"expected exactly 1 fallback warning, got {first_count}"
    assert second_count == 1, "second call must not re-warn (once-per-process gate)"


def test_rust_path_does_not_warn(caplog):
    """The Rust path is the production path and must not trip the fallback warning."""
    rust_game = _fresh_rust_game()
    mapper = ActionMapper()
    if hasattr(mapper, "_warned_python_fallback"):
        del mapper._warned_python_fallback
    import logging
    with caplog.at_level(logging.WARNING, logger="rl18xx.agent.alphazero.action_mapper"):
        mapper.get_legal_actions_factored(rust_game)
    assert not any(
        "falling back to Python" in r.message for r in caplog.records
    )
