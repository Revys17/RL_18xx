"""Regression states where the pure-Python MCTS path crashed on an enumerated index.

``MCTSNode.maybe_add_child`` decodes an index with the Python ``ActionMapper``
and applies it to the game. On a ``RustGameAdapter`` that decode used to fail
for indices the factored enumeration correctly reports as legal (the native
Rust decode handled all of them), crashing gating and arena games:

* ``teleport_pass`` — while a DH teleport token is pending, Ruby/Python's
  ``SpecialToken`` step accepts a Pass only from the teleported COMPANY
  (``actions << 'pass' if entity == @round.teleported``); Python's
  ``current_entity`` is DH there. The adapter reported the operating corp, so
  the decoded Pass came from the corp and both engines rejected it.
* ``mh_exchange_pre_par`` — MH may be exchanged for an NYC IPO share before NYC
  pars. The adapter's ``exchangeable_shares`` skipped unparred corps.
* ``d_trade_in`` — any 4, 5 or 6 trades $300 off a D; the adapter's
  ``discountable_trains_for`` only listed a 4.
* ``discarded_d_trade_in`` — with a D in the discard pool, a D trade-in indexes
  into the discard slot; the mapper decoded the slot as a face-value buy.
* ``emergency_sell`` — in an operating round the seller is the operating
  corp's president; the adapter looked up the corp as if it were a player.

Each state is a Python-format action log that replays into both engines.
"""

import json
from pathlib import Path

import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import MCTSNode
from rl18xx.game.engine.round import SpecialToken
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "python_mcts_decode_states.json"
STATES = json.loads(FIXTURE.read_text())

# The decoded action each state must produce (subset of the action dict).
EXPECTED = {
    "teleport_pass": {"type": "pass", "entity": "DH", "entity_type": "company"},
    "mh_exchange_pre_par": {"type": "buy_shares", "entity": "MH", "entity_type": "company", "shares": ["NYC_1"]},
    "d_trade_in": {"type": "buy_train", "entity": "CPR", "train": "D-0", "price": 800, "exchange": "5-2"},
    "discarded_d_trade_in": {"type": "buy_train", "entity": "B&M", "train": "D-0", "price": 800, "exchange": "6-0"},
    "emergency_sell": {"type": "sell_shares", "entity": 1, "entity_type": "player", "shares": ["B&O_1"]},
}


def _players(state):
    return {i + 1: f"P{i + 1}" for i in range(state["players"])}


def _rust(name):
    state = STATES[name]
    game = engine_rs.BaseGame(_players(state))
    for action in state["actions"]:
        game.process_action(dict(action))
    return game


def _python(name):
    state = STATES[name]
    game = GameMap().game_by_title("1830")(_players(state))
    for action in state["actions"]:
        game.process_action(dict(action))
    return game


def _decode(game, name):
    state = STATES[name]
    mapper = ActionMapper()
    if state["price"] is not None:
        return mapper.map_index_to_action_with_price(state["index"], game, state["price"])
    return mapper.map_index_to_action(state["index"], game)


def _strip(action_dict):
    return {k: v for k, v in action_dict.items() if v not in (None, []) and k != "created_at"}


def _encoded(rust_game):
    game_state, node_features, *_ = rust_game.encode_for_gnn()
    return list(game_state), list(node_features)


@pytest.mark.parametrize("name", list(STATES))
def test_index_is_legal_in_both_engines(name):
    mapper = ActionMapper()
    rust_indices = mapper.get_legal_actions_factored(RustGameAdapter(_rust(name)))[0]
    python_indices = mapper.get_legal_actions_factored(_python(name))[0]
    assert STATES[name]["index"] in rust_indices
    assert rust_indices == python_indices


@pytest.mark.parametrize("name", list(STATES))
def test_maybe_add_child_applies_the_index(name):
    adapter = RustGameAdapter(_rust(name))
    node = MCTSNode(adapter, config=SelfPlayConfig(network=None))
    child = node.maybe_add_child(STATES[name]["index"])
    assert child.game_object.move_number > adapter.move_number


@pytest.mark.parametrize("name", list(STATES))
def test_adapter_decode_matches_native_decode(name):
    """Python decode on the adapter lands in the same state as the native decode."""
    rust = _rust(name)
    adapter = RustGameAdapter(rust.pickle_clone())
    action = _decode(adapter, name)
    assert EXPECTED[name].items() <= _strip(action.to_dict()).items()
    adapter.process_action(action)

    native = rust.pickle_clone()
    native.apply_action_index(STATES[name]["index"], STATES[name]["price"])
    assert _encoded(adapter._game) == _encoded(native)


@pytest.mark.parametrize("name", list(STATES))
def test_adapter_decode_matches_python_engine_decode(name):
    """The Python engine (the oracle) decodes the index to the same action and accepts it."""
    adapter_dict = _strip(_decode(RustGameAdapter(_rust(name)), name).to_dict())
    python = _python(name)
    python_action = _decode(python, name)
    python_dict = _strip(python_action.to_dict())
    python.process_action(python_action)
    assert adapter_dict == python_dict


def test_discarded_d_trade_in_buys_the_discarded_train():
    rust = _rust("discarded_d_trade_in")
    assert "D-0" in [t.id for t in rust.depot.discarded]
    action = _decode(RustGameAdapter(rust), "discarded_d_trade_in")
    # Face value (1100) is more than the corporation has; the slot means the trade-in.
    assert action.price == 800 and action.exchange is not None


def test_teleport_actor_is_the_company_in_both_engines():
    rust = _rust("teleport_pass")
    adapter = RustGameAdapter(rust)
    python = _python("teleport_pass")

    assert rust.acting_entity_id() == "company:DH"
    assert adapter.current_entity.sym == "DH"
    assert [e.sym for e in adapter.round.active_entities] == ["DH"]
    assert isinstance(adapter.active_step(), SpecialToken)
    assert python.current_entity.id == "DH"
    assert isinstance(python.active_step(), SpecialToken)


def test_teleport_pass_from_the_corporation_is_rejected_by_both_engines():
    """Only the teleported company may decline the token (Ruby special_token.rb)."""
    rust = _rust("teleport_pass")
    corp = rust.current_entity_id.split(":", 1)[1]
    corp_pass = {"type": "pass", "entity": corp, "entity_type": "corporation"}
    with pytest.raises(Exception, match="Place teleport token"):
        rust.process_action(dict(corp_pass))
    with pytest.raises(Exception, match="Place teleport token"):
        _python("teleport_pass").process_action(dict(corp_pass))

    rust.process_action({"type": "pass", "entity": "DH", "entity_type": "company"})
    assert not rust.teleport_pending()
