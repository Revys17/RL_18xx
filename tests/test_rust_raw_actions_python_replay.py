"""Natively-decoded self-play logs replay in the Python engine and in Rust.

Self-play applies policy indices natively (``BaseGame.apply_action_index``);
the Rust engine logs each move to ``raw_actions`` as an 18xx.games action dict.
That log is the game record: it must replay in the Python reference engine
(which reads the 18xx.games format the browser replay tooling uses) and in a
fresh Rust engine, reproducing the original game exactly.

Each case plays a seeded random native game to completion (or a move cap),
then replays its log through both engines, comparing an engine-neutral state
fingerprint (cash, every certificate's owner, share prices, trains, tokens,
laid tiles, depot, phase, round) at regular checkpoints and at the end, plus
the final result. The Rust replay must also match the full GNN encoding.

Run routes are additionally rebuilt in the Python engine from their logged
``connections`` / ``nodes`` with the revenue stripped, so Python recomputes
each route itself — the logged chains must name real routes, not just carry a
total. Python's route validation (a port of Ruby's route.rb) must accept every
route the Rust router picks: no stop visited twice, no track reused within or
across the run's routes, at most one stop per group (both Canada offboards).
"""

import copy
import logging
import random
import re

import pytest

import engine_rs
from rl18xx.game.engine.actions import BaseAction
from rl18xx.game.engine.core import GameError
from rl18xx.game.engine.entities import Player, SharePool
from rl18xx.game.gamemap import GameMap

logging.disable(logging.CRITICAL)

MAX_MOVES = 3000
CHECKPOINT_EVERY = 100
# The pinned cases are long games; guard against a case silently degenerating.
MIN_ACTIONS = 500
MIN_ROUTES_REBUILT = 20

# (seed, players, auction_unlock) — every player count, both auction rules.
CASES = [
    (0, 4, False),
    (1, 3, True),
    (2, 5, False),
    (3, 2, True),
    (4, 6, False),
    (12, 4, True),
]


def _players(n):
    return {i: f"Player {i}" for i in range(1, n + 1)}


def _owner_key(owner):
    """Normalize a certificate owner across engines: a player, the market, or
    the corporation's IPO (Rust: ``ipo:SYM`` / unissued ``""``)."""
    if isinstance(owner, str):  # Rust EntityId string
        if owner.startswith("player:") or owner == "market":
            return owner
        return "ipo"
    if isinstance(owner, Player):
        return f"player:{owner.id}"
    if isinstance(owner, SharePool):
        return "market"
    return "ipo"


def _rust_fingerprint(g):
    corps = {}
    for c in g.corporations:
        corps[c.sym] = (
            c.cash,
            c.share_price.price if c.share_price else None,
            tuple(sorted(t.id for t in c.trains)),
            tuple(_owner_key(s.owner) for s in sorted(c.shares, key=lambda s: s.index)),
            tuple(sorted(t.city_hex_id for t in c.tokens if t.used)),
        )
    # Laid tiles only: the lay handler names a tile by its instance id;
    # preprinted tiles have a "preprinted_" id and blank hexes an empty one.
    tiles = {
        h.id: (h.tile.name, h.tile.rotation) for h in g.hexes if h.tile.name and not h.tile.id.startswith("preprinted_")
    }
    return {
        "players": {p.id: p.cash for p in g.players},
        "bank": g.bank.cash,
        "corps": corps,
        "tiles": tiles,
        "depot": (tuple(sorted(t.id for t in g.depot.trains)), tuple(sorted(t.id for t in g.depot.discarded))),
        "phase": g.phase.name,
        "round": g.round.round_type,
        "finished": g.finished,
    }


def _py_fingerprint(pg):
    certs = {}
    for s in pg.shares:
        certs.setdefault(s.corporation().id, {})[s.index] = _owner_key(s.owner)
    corps = {}
    for c in pg.corporations:
        corps[c.id] = (
            c.cash,
            c.share_price.price if c.share_price else None,
            tuple(sorted(t.id for t in c.trains)),
            tuple(owner for _, owner in sorted(certs.get(c.id, {}).items())),
            tuple(sorted(t.city.hex.id for t in c.tokens if t.used and t.city)),
        )
    tiles = {h.id: (h.tile.id, h.tile.rotation) for h in pg.hexes if not h.tile.preprinted}
    return {
        "players": {p.id: p.cash for p in pg.players},
        "bank": pg.bank.cash,
        "corps": corps,
        "tiles": tiles,
        "depot": (tuple(sorted(t.id for t in pg.depot.upcoming)), tuple(sorted(t.id for t in pg.depot.discarded))),
        "phase": pg.phase.name,
        "round": type(pg.round).__name__,
        "finished": pg.finished,
    }


def _play_native(seed, players, auction_unlock):
    """A seeded random native game, with Rust fingerprints at checkpoints."""
    rng = random.Random(seed)
    game = engine_rs.BaseGame(players)
    game.set_auction_unlock(auction_unlock)
    checkpoints = {}
    for move in range(1, MAX_MOVES + 1):
        if game.finished:
            break
        idx = int(rng.choice(game.factored_legal_indices()))
        price_range = game.price_range_for_index(idx)
        price = None
        if price_range is not None and price_range[0] != price_range[1]:
            price = rng.randint(price_range[0], price_range[1])
        game.apply_action_index(idx, price)
        if move % CHECKPOINT_EVERY == 0:
            checkpoints[move] = (_rust_fingerprint(game), game.encode_for_gnn())
    return game, checkpoints


def _rebuild_routes(action_dict, pg):
    """Rebuild a logged run_routes in the Python engine with each route's
    revenue stripped; return (logged, recomputed) revenue pairs. Validating a
    route 1830's rules forbid raises a GameError ("Cannot use X twice",
    "Route cannot reuse track on X", "Cannot use group X more than once")."""
    stripped = copy.deepcopy(action_dict)
    stripped["routes"] = [{k: v for k, v in r.items() if k != "revenue"} for r in action_dict["routes"]]
    rebuilt = BaseAction.action_from_dict(stripped, pg)
    return [(r["revenue"], route.revenue()) for r, route in zip(action_dict["routes"], rebuilt.routes)]


@pytest.mark.parametrize("seed,num_players,auction_unlock", CASES)
def test_native_log_replays_in_python_and_rust(seed, num_players, auction_unlock):
    players = _players(num_players)
    game, checkpoints = _play_native(seed, players, auction_unlock)
    actions = game.raw_actions
    assert len(actions) > MIN_ACTIONS, "game ended suspiciously early"

    py_game = GameMap().game_by_title("1830")(players)
    py_game.auction_unlock = auction_unlock
    rust_game = engine_rs.BaseGame(players)
    rust_game.set_auction_unlock(auction_unlock)

    routes_checked = 0
    for move, action in enumerate(actions, start=1):
        if action["type"] == "run_routes":
            try:
                rebuilt = _rebuild_routes(action, py_game)
            except GameError as exc:
                pytest.fail(f"move {move}: Python rejects a logged route ({exc}): {action}")
            for logged, recomputed in rebuilt:
                assert recomputed == logged, f"move {move}: route recomputes to {recomputed}: {action}"
                routes_checked += 1
        try:
            py_game.process_action(BaseAction.action_from_dict(copy.deepcopy(action), py_game))
        except Exception as exc:
            pytest.fail(f"Python engine rejected move {move} {action}: {type(exc).__name__}: {exc}")
        rust_game.process_action(copy.deepcopy(action))

        if move in checkpoints:
            fingerprint, encoding = checkpoints[move]
            assert _py_fingerprint(py_game) == fingerprint, f"Python diverged by move {move}"
            assert _rust_fingerprint(rust_game) == fingerprint, f"Rust replay diverged by move {move}"
            assert rust_game.encode_for_gnn() == encoding, f"Rust replay encoding diverged by move {move}"

    final = _rust_fingerprint(game)
    assert _py_fingerprint(py_game) == final
    assert _rust_fingerprint(rust_game) == final
    assert rust_game.encode_for_gnn() == game.encode_for_gnn()
    assert rust_game.raw_actions == actions
    if game.finished:
        assert py_game.result() == dict(game.result())
        assert dict(rust_game.result()) == dict(game.result())
    assert routes_checked > MIN_ROUTES_REBUILT, "too few routes rebuilt for the route check to mean anything"


def test_native_log_uses_18xx_games_format():
    """The fields the native log writes are the 18xx.games ones, which the
    Python engine's ``to_dict`` also emits — e.g. a par names its market cell,
    a share trade names the exact certificates, a token names its city."""
    game, _ = _play_native(0, _players(4), False)
    by_type = {}
    for action in game.raw_actions:
        by_type.setdefault(action["type"], []).append(action)

    for par in by_type["par"]:
        assert re.fullmatch(r"\d+,\d+,\d+", par["share_price"]), par
    for kind in ("buy_shares", "sell_shares"):
        for trade in by_type[kind]:
            assert trade["shares"] and all(re.fullmatch(r"[A-Z&]+_\d+", s) for s in trade["shares"]), trade
    for token in by_type["place_token"]:
        assert re.fullmatch(r"\w+-\d+-\d+", token["city"]) and {"slot", "tokener"} <= set(token), token
    for buy in by_type["buy_train"]:
        assert re.fullmatch(r"\w+-\d+", buy["train"]) and buy["variant"] == buy["train"].split("-")[0], buy
    for discard in by_type.get("discard_train", []):
        assert re.fullmatch(r"\w+-\d+", discard["train"]), discard
    for run in by_type["run_routes"]:
        for route in run["routes"]:
            assert {"train", "connections", "hexes", "revenue", "revenue_str", "nodes"} <= set(route), run
            assert re.fullmatch(r"\w+-\d+", route["train"]), run
            assert len(route["nodes"]) == len(route["connections"]) + 1, run
