"""1830 ``multiple_brown_from_ipo`` (Ruby G1830 ``multiple_buy_only_from_market?``)
in both engines and through the human-game import.

A stock turn may buy several shares of a corporation whose price is in the
brown ("multiple_buy") zone -- under the base rules only from the market pool,
with the rule from the IPO as well. The Python engine is the oracle; the Rust
engine pins the same cases on rigged states in its own tests
(``rounds::stock::multiple_brown_from_ipo_tests``), and the fixture -- an
anonymized 4-player game truncated right after a turn that bought brown shares
from the IPO twice (see its ``description``) -- checks both engines in
lockstep.
"""

import copy
import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.engine.entities import ShareBundle  # noqa: E402
from rl18xx.game.factored_action_helper import FactoredActionHelper  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.agent.alphazero.pretraining import (  # noqa: E402
    ENGINE_OPTIONAL_RULES,
    _get_game_object_for_game_with_reason,
    _load_cleaned_game_via_rust,
    engine_optional_rules,
)
from tests import cleaning_diff  # noqa: E402
from tests.validate_rust_engine import compare_state  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "1830" / "multiple_brown_from_ipo.json"
RULE = "multiple_brown_from_ipo"
PLAYERS = {i: f"Player {i}" for i in range(1, 5)}
# 1830 market cells: the $67 par, $27 brown ("multiple_buy") and $39 orange
# ("unlimited": ownership-exempt, but one buy per turn).
PAR, BROWN, ORANGE = (5, 6), (6, 1), (3, 0)


def _nynh_game(rules, cell=BROWN):
    """A 4-player 1830 stock round (the first: no sales yet) with NYNH parred at
    $67 and parked on market cell ``cell``: player 2 is president with certs 0-3
    (50%), certs 4-6 are in the IPO and 7-8 in the market pool. Player 1 acts.
    Same rig as the Rust tests."""
    game = GameMap().game_by_title("1830")(PLAYERS, optional_rules=rules)
    game.round = game.new_stock_round()
    corp = game.corporation_by_id("NYNH")
    game.stock_market.set_par(corp, game.stock_market.share_price(PAR))
    shares = [game.share_by_id(f"NYNH_{i}") for i in range(9)]
    game.share_pool.transfer_shares(ShareBundle(shares[0:4]), game.player_by_id(2))
    game.share_pool.transfer_shares(ShareBundle(shares[7:9]), game.share_pool)
    corp.ipoed = True
    game.stock_market.move(corp, cell, force=True)
    assert game.current_entity.id == 1 and corp.owner.id == 2
    return game


def _sources(game):
    """NYNH's legal BuyShares sources in the factored enumeration."""
    if not game.round.actions_for(game.current_entity):
        # A rigged turn with nothing left to do (the engine would have moved
        # on); the helper's no-action fallback isn't meant for it.
        return []
    return sorted(
        la.params["source"]
        for la in FactoredActionHelper().get_choices(game)
        if la.type == "BuyShares" and la.entity.get("corp") == "NYNH"
    )


def _buy(game, source):
    corp = game.corporation_by_id("NYNH")
    pool = corp.ipo_shares if source == "ipo" else corp.market_shares
    share = min(pool, key=lambda s: s.index)
    pid = game.current_entity.id
    game.process_action(
        {"type": "buy_shares", "entity": pid, "entity_type": "player", "shares": [share.id], "percent": 10}
    )


def _player_1_percent(game):
    return game.player_by_id(1).percent_of(game.corporation_by_id("NYNH"))


def _mark_bought(game, corp_sym=None, par=False):
    """Make the current turn look like it already bought ``corp_sym`` (or parred)
    without moving a certificate -- a turn that stays open after its buy."""
    from rl18xx.game.engine.actions import BuyShares, Par

    step = game.round.active_step()
    if par:
        action = Par(game.current_entity, game.corporation_by_id("PRR"), game.stock_market.share_price(PAR))
    else:
        action = BuyShares(game.current_entity, game.share_by_id(f"{corp_sym}_8"))
    step.round.current_actions.append(action)


def test_rule_is_in_both_engines_and_the_importer():
    assert RULE in ENGINE_OPTIONAL_RULES
    assert engine_optional_rules({"settings": {"optional_rules": [RULE, "optional_6_train"]}}) == [
        RULE,
        "optional_6_train",
    ]
    assert list(RustGame(PLAYERS, optional_rules=[RULE]).optional_rules) == [RULE]
    py = GameMap().game_by_title("1830")(PLAYERS, optional_rules=[RULE])
    assert not py.multiple_buy_only_from_market()
    assert GameMap().game_by_title("1830")(PLAYERS).multiple_buy_only_from_market()


def test_base_rules_rebuy_brown_shares_from_the_pool_only():
    game = _nynh_game([])
    assert _sources(game) == ["ipo", "market"]
    _buy(game, "ipo")
    assert game.current_entity.id == 1, "a brown-zone buy keeps the turn open"
    assert _sources(game) == ["market"]
    with pytest.raises(Exception, match="Cannot buy a share of NYNH"):
        _buy(game, "ipo")
    # The market path is unchanged -- including after an IPO buy: the Python
    # port's IPO is the bank, so ``round.bought_from_ipo`` (set only for a
    # corporation-owned bundle) never fires.
    assert game.round.bought_from_ipo is False
    _buy(game, "market")
    _buy(game, "market")
    assert _player_1_percent(game) == 30
    # Pool empty, IPO closed to re-buys, no sales in SR1: the turn passes on.
    assert game.current_entity.id == 2


def test_rule_rebuys_brown_shares_from_the_ipo_too():
    game = _nynh_game([RULE])
    _buy(game, "ipo")
    assert _sources(game) == ["ipo", "market"]
    _buy(game, "ipo")
    _buy(game, "market")
    _buy(game, "ipo")
    assert _player_1_percent(game) == 40
    assert _sources(game) == ["market"], "the IPO is empty"
    _buy(game, "market")
    assert _player_1_percent(game) == 50
    assert game.current_entity.id == 2


@pytest.mark.parametrize("rules, stays_open", [([], False), ([RULE], True)])
def test_empty_pool_ends_the_turn_only_under_the_base_rules(rules, stays_open):
    game = _nynh_game(rules)
    corp = game.corporation_by_id("NYNH")
    game.share_pool.transfer_shares(ShareBundle(corp.market_shares), game.player_by_id(3))
    _buy(game, "ipo")
    assert (game.current_entity.id == 1) == stays_open


@pytest.mark.parametrize("rules", [[], [RULE]], ids=["base", "rule"])
@pytest.mark.parametrize("cell", [PAR, ORANGE], ids=["par", "orange"])
def test_non_brown_zones_stay_one_buy_per_turn(rules, cell):
    game = _nynh_game(rules, cell)
    assert _sources(game) == ["ipo", "market"]
    _buy(game, "ipo")
    assert game.current_entity.id == 2, "one buy ends the SR1 turn"

    game = _nynh_game(rules, cell)
    _mark_bought(game, "NYNH")
    assert _sources(game) == []
    for source in ("ipo", "market"):
        with pytest.raises(Exception):
            _buy(game, source)


@pytest.mark.parametrize("par", [True, False], ids=["after_par", "after_other_corp"])
def test_rule_keeps_the_par_and_same_corp_restrictions(par):
    game = _nynh_game([RULE])
    _mark_bought(game, "B&O", par=par)
    assert _sources(game) == []
    for source in ("ipo", "market"):
        with pytest.raises(Exception):
            _buy(game, source)


@pytest.mark.parametrize("rules", [[], [RULE]], ids=["base", "rule"])
def test_cert_limit_does_not_end_a_brown_multi_buy(rules):
    """Brown-zone certs are cert-limit exempt (CERT_LIMIT_TYPES / ``can_gain``),
    so a player at the limit keeps buying them and the turn stays open."""
    game = _nynh_game(rules)
    game.cert_limit = lambda *args, **kwargs: 0
    _buy(game, "market")
    assert game.current_entity.id == 1
    assert "market" in _sources(game)
    _buy(game, "market")


# ----- the recorded game: both engines, through the human-game import -----


def _fixture(rules=None):
    with open(FIXTURE) as f:
        game = json.load(f)
    if rules is not None:
        game["settings"]["optional_rules"] = rules
    return game


@pytest.fixture(scope="module")
def cleaned():
    """The fixture cleaned on each engine: {use_rust: (game, reason)}."""
    return {
        use_rust: _get_game_object_for_game_with_reason(_fixture(), use_rust=use_rust) for use_rust in (True, False)
    }


@pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
def test_cleaner_replays_the_rule_game(cleaned, use_rust):
    game, reason = cleaned[use_rust]
    assert reason is None
    assert len(game.raw_actions) > 80


def test_engines_agree_after_cleaning(cleaned):
    (rust, _), (py, _) = cleaned[True], cleaned[False]
    assert compare_state(rust._game, py) == []
    # The last turn took Player 3 to 70% of NYNH: three pool shares, then
    # two from the IPO.
    assert py.player_by_id(3).percent_of(py.corporation_by_id("NYNH")) == 70
    assert rust._game.corporation_by_id("NYNH").percent_owned_by_entity("player:3") == 70


@pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
def test_base_rules_drop_the_game(use_rust):
    """Replayed under the base rules, the first IPO buy after the pool buys
    is illegal -- in both engines, for the same reason."""
    assert _get_game_object_for_game_with_reason(_fixture([]), use_rust=use_rust) == (None, "illegal_share_buy")


@pytest.mark.parametrize(
    "rules, outcome",
    [([RULE], {"status": "parity"}), ([], {"status": "dropped", "reason": "illegal_share_buy"})],
    ids=["rule", "base"],
)
def test_strict_lockstep_parity(rules, outcome):
    """State, legal-set and acceptance parity at every step of the import."""
    result = cleaning_diff.diagnose_game(_fixture(rules), strict=True)
    assert {k: result.get(k) for k in outcome} == outcome


@pytest.mark.parametrize("use_rust", [True, False], ids=["rust", "python"])
def test_cleaned_game_keeps_its_optional_rules(cleaned, use_rust):
    """``to_dict`` -> ``_load_cleaned_game_via_rust`` is how cleaned games
    reach training-data conversion; the rule must survive that hop."""
    game_dict = cleaned[use_rust][0].to_dict()
    assert game_dict["settings"]["optional_rules"] == [RULE]
    reloaded = _load_cleaned_game_via_rust(game_dict)
    assert reloaded is not None
    assert list(reloaded.optional_rules) == [RULE]
    assert reloaded._game.corporation_by_id("NYNH").percent_owned_by_entity("player:3") == 70
