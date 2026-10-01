"""Every legal index decodes to a processable action on the RustGameAdapter.

The pure-Python MCTS path (``MCTSPlayer`` with ``use_rust_mcts=False`` — gating
and the arena) enumerates with the Rust factored helper but decodes with
``ActionMapper.map_index_to_action`` through the adapter. Any index the helper
reports as legal that the adapter can't materialize crashes MCTS expansion.

``test_mh_exchange_before_nyc_par`` pins the original crash (MH -> NYC exchange
in SR1, before NYC pars). The random sweep checks every legal index at every
state of natively-advanced games: the ActionMapper decode must process on a
clone AND reach the same state as the native Rust decode (``apply_action_index``,
the Rust MCTS / self-play path). ``test_sweep_covers_known_gaps`` keeps the sweep
seeds honest — each decode gap this suite was written for must actually occur.
"""

import functools
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.disable(logging.CRITICAL)

import pytest

from engine_rs import BaseGame as RustGame
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter

PLAYERS = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}


def _bid(player, company, price):
    return {"type": "bid", "entity": player, "entity_type": "player", "company": company, "price": price}


def _pass(player):
    return {"type": "pass", "entity": player, "entity_type": "player"}


# Private auction (p4 ends up with MH at 375), then p2 pars B&O for the BO
# private — the first SR1 decision belongs to p4, the MH owner, with NYC unparred.
# fmt: off
MH_EXCHANGE_OPENING = [
    _bid(1, "CA", 285), _bid(2, "SV", 20), _bid(3, "BO", 490), _pass(4),
    _bid(1, "CS", 40), _bid(2, "BO", 500), _pass(3), _bid(4, "MH", 115),
    _bid(1, "MH", 120), _pass(2), _bid(3, "DH", 70), _bid(4, "MH", 125),
    _bid(1, "MH", 175), _bid(4, "MH", 375), _pass(1), _pass(3),
    {"type": "par", "entity": 2, "entity_type": "player", "corporation": "B&O", "share_price": "100,0,6"},
]
# fmt: on

MH_EXCHANGE_IPO = ActionMapper().action_offsets["CompanyBuyShares"]


def _decode(mapper, index, adapter, price):
    if price is None:
        return mapper.map_index_to_action(index, adapter)
    return mapper.map_index_to_action_with_price(index, adapter, int(price))


def _prices(price_range):
    """Both ends and the midpoint of a price range; [None] for categorical slots."""
    if price_range is None:
        return [None]
    lo, hi = price_range
    return sorted({lo, hi, (lo + hi) // 2})


def _encoding(game):
    game_state, node_features, *_ = game.encode_for_gnn()
    return tuple(game_state), tuple(node_features)


def test_mh_exchange_before_nyc_par():
    game = RustGame(PLAYERS)
    for action in MH_EXCHANGE_OPENING:
        game.process_action(action)
    adapter = RustGameAdapter(game)
    mapper = ActionMapper()
    assert game.corporation_by_id("NYC").ipo_price is None

    indices, price_ranges, _ = mapper.get_legal_actions_factored(adapter)
    assert MH_EXCHANGE_IPO in indices
    for index in indices:
        for price in _prices(price_ranges.get(index)):
            action = _decode(mapper, index, adapter, price)
            game.pickle_clone().process_action(action.to_dict())

    exchange = mapper.map_index_to_action(MH_EXCHANGE_IPO, adapter).to_dict()
    assert (exchange["entity"], exchange["shares"]) == ("MH", ["NYC_1"])
    after = game.pickle_clone()
    after.process_action(exchange)
    assert after.company_by_id("MH").closed
    assert after.corporation_by_id("NYC").percent_owned_by_entity("player:4") == 10

    # The Python engine decodes the same exchange at the same state.
    py_game = GameMap().game_by_title("1830")(PLAYERS)
    for action in MH_EXCHANGE_OPENING:
        py_game = py_game.process_action(action)
    py_exchange = mapper.map_index_to_action(MH_EXCHANGE_IPO, py_game).to_dict()
    assert (py_exchange["entity"], py_exchange["shares"]) == ("MH", ["NYC_1"])


def _gaps_exercised(game, adapter, mapper, index, action_type, decoded):
    """Which of the decode gaps this suite guards does this decode exercise?"""
    offsets = mapper.action_offsets
    gaps = []
    if action_type == "CompanyBuyShares" and game.corporation_by_id("NYC").ipo_price is None:
        gaps.append("exchange_before_par")
    if action_type == "SellShares" and adapter.round.round_type == "Operating":
        gaps.append("emergency_sell")
    if index == offsets["BuyTrainDTradeIn"] and decoded.get("exchange"):
        gaps.append("d_trade_in_non_4_donor" if not decoded["exchange"].startswith("4-") else "d_trade_in")
    if action_type == "BuyTrain" and index < offsets["BuyTrainDFull"] and decoded.get("exchange"):
        gaps.append("discard_slot_trade_in")
    if action_type == "Pass" and (game.acting_entity_id() or "").startswith("company:"):
        gaps.append("teleport_pass")
    if action_type == "PlaceToken":
        hex_id = mapper.actions[index][1][-2]  # args: [hex, city] or [company, hex, city]
        tile = adapter.hex_by_id(hex_id).tile.name.split("-")[0]
        if sum(h.tile.name.split("-")[0] == tile for h in game.hexes) > 1:
            gaps.append("token_on_duplicate_tile")
    return gaps


@functools.lru_cache(maxsize=None)
def _sweep(seed, max_steps=3000):
    """Play a random game advancing natively; check every legal index at every state."""
    rng = random.Random(seed)
    num_players = rng.choice([2, 3, 4, 5, 6])
    game = RustGame({i: f"P{i}" for i in range(1, num_players + 1)})
    mapper = ActionMapper()
    failures = []
    gaps = Counter()
    for step in range(max_steps):
        if game.finished:
            break
        adapter = RustGameAdapter(game)
        indices, price_ranges, action_types = mapper.get_legal_actions_factored(adapter)
        assert indices, f"seed={seed} step={step}: no legal indices"
        for index in indices:
            for price in _prices(price_ranges.get(index)):
                where = f"seed={seed} step={step} index={index} ({action_types.get(index)}) price={price}"
                try:
                    decoded = _decode(mapper, index, adapter, price).to_dict()
                    via_mapper = game.pickle_clone()
                    via_mapper.process_action(decoded)
                except Exception as exc:
                    failures.append(f"{where}: {type(exc).__name__}: {exc}")
                    continue
                native = game.pickle_clone()
                native.apply_action_index(index, price)
                if _encoding(native) != _encoding(via_mapper):
                    failures.append(f"{where}: mapper decode {decoded} diverges from native decode")
                gaps.update(_gaps_exercised(game, adapter, mapper, index, action_types.get(index), decoded))
        index = rng.choice(indices)
        price_range = price_ranges.get(index)
        game.apply_action_index(index, rng.randint(*price_range) if price_range else None)
    return failures, gaps


SWEEP_SEEDS = [35, 49]  # 49 exercises every gap below on its own (~17s); 35 is a short second trajectory


@pytest.mark.parametrize("seed", SWEEP_SEEDS)
def test_every_legal_index_decodes_on_adapter(seed):
    failures, _ = _sweep(seed)
    assert not failures, f"{len(failures)} decode failure(s):\n" + "\n".join(failures[:10])


def test_sweep_covers_known_gaps():
    covered = Counter()
    for seed in SWEEP_SEEDS:
        covered.update(_sweep(seed)[1])
    expected = {
        "exchange_before_par",
        "emergency_sell",
        "d_trade_in_non_4_donor",
        "discard_slot_trade_in",
        "teleport_pass",
        "token_on_duplicate_tile",
    }
    assert expected <= set(covered), f"sweep seeds no longer exercise: {sorted(expected - set(covered))}"
