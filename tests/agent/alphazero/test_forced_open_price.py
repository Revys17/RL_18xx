"""A played move's forced chain that reaches a single legal move with an open price
(``RustMCTSPlayer.advance_root``'s ``stop_at_open_price``, ``SelfPlayConfig.stop_chain_at_open_price``):
the root stops there, so the move is priced from its own position's price head."""

import random

import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.policy_selfplay import has_price_choice  # noqa: E402
from rl18xx.agent.alphazero.start_positions import new_game  # noqa: E402

from .search_test_utils import HashNet, expand_root, make_player  # noqa: E402

DEFAULT = dict(parallel_readouts=8, num_readouts=16, min_readouts=16)


def _fixed_price(rs, move) -> bool:
    price_range = rs.price_range_for_index(move)
    return price_range is None or price_range[0] == price_range[1]


def _move_into_forced_open_price():
    """``(game before, move, game after)``: a random walk's fixed-price move after which the next
    position's only legal move has an open price."""
    for seed in range(60):
        rng = random.Random(seed)
        rs = new_game(4, auction_unlock=True)._game
        for _ in range(4000):
            if rs.finished:
                break
            legal = [int(i) for i in rs.factored_legal_indices()]
            if not legal:
                break
            idx = rng.choice(legal)
            price_range = rs.price_range_for_index(idx)
            price = None if price_range is None else rng.randint(int(price_range[0]), int(price_range[1]))
            before = rs.pickle_clone()
            try:
                rs.apply_action_index(idx, price)
            except Exception:
                break
            after = [int(i) for i in rs.factored_legal_indices()]
            if len(legal) > 1 and _fixed_price(before, idx) and len(after) == 1 and has_price_choice(rs, after[0]):
                return before, idx, rs
    pytest.skip("no fixed-price move into a forced open-price move found")


def test_the_root_stops_at_a_forced_open_price_move():
    before, move, after = _move_into_forced_open_price()
    single = int(after.factored_legal_indices()[0])

    stopping = make_player(HashNet(), before, **DEFAULT)._rust_player
    stopping.advance_root(move, True)
    root = stopping.root_game_object()
    assert len(root.raw_actions) == len(after.raw_actions)  # the forced move is not played
    assert [int(i) for i in stopping.legal_action_indices_at_root()] == [single]
    assert stopping.root_move_has_open_price(single) and not stopping.is_expanded(stopping.root_idx)

    collapsing = make_player(HashNet(), before, **DEFAULT)._rust_player
    collapsing.advance_root(move)  # the default: the chain collapses it, priced from the previous position
    assert len(collapsing.root_game_object().raw_actions) > len(after.raw_actions)


def test_the_player_prices_it_at_its_own_position():
    before, move, after = _move_into_forced_open_price()
    single = int(after.factored_legal_indices()[0])
    lo, hi = after.price_range_for_index(single)

    player = make_player(HashNet(), before, stop_chain_at_open_price=True, **DEFAULT)
    expand_root(player)
    player.play_move(move)
    assert player.forced_action_dicts[-1] == []  # the open-price move is not swallowed into the chain
    assert player.start_search() is not None  # the lockstep path: a 0-readout root evaluation
    assert player.suggest_move() == single
    assert player._rust_player.is_expanded(player._rust_player.root_idx)  # its price row is the root's own
    player.play_move(single)
    price = player.played_actions[-1].get("price")
    assert price is None or lo <= price <= hi
    assert len(player.get_game_state().raw_actions) > len(after.raw_actions)
