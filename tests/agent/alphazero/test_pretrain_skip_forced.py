"""Human-game conversion: forced positions, policy labels, and the train/validation split.

A position whose legal set has exactly one action index carries no decision:
its policy target is trivial and its value target is one more correlated copy
of the game's outcome. Self-play never records such positions (the MCTS
applies forced actions without searching), so ``convert_game_to_training_data``
skips them by default while still replaying the action.

Every recorded human action must match exactly one policy index. Token
placements name their city by tile id ("57-1-0"), which only the board
resolves to a hex; train purchases name the train, whose current owner is the
seller; company abilities (CS / DH) act as the private, not the corporation.

The fixtures are anonymized 2-player games: ``router_d_trains.json`` (finished)
and ``router_two_d_trains.json`` (has a purchase from the depot's discard pool).
"""

import io
import json
import logging
import sys
from pathlib import Path

import lmdb
import lz4.frame
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[3]))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.agent.alphazero.action_mapper import ActionMapper  # noqa: E402
from rl18xx.agent.alphazero.config import TrainingConfig  # noqa: E402
from rl18xx.agent.alphazero.encoder import Encoder_Transformer  # noqa: E402
from rl18xx.agent.alphazero.pretraining import (  # noqa: E402
    AmbiguousActionMatch,
    _action_dict_to_factored_index,
    _matching_choices,
    _get_game_object_for_game_with_reason,
    _load_cleaned_game_via_rust,
    convert_game_to_training_data,
    convert_games_to_training_dataset,
    in_validation_split,
)
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

FIXTURES = Path(__file__).parents[2] / "fixtures" / "1830"
ALL_TRAINING = TrainingConfig(pretrain_validation_percentage=0.0)


def _cleaned(name):
    with open(FIXTURES / name) as f:
        game, reason = _get_game_object_for_game_with_reason(json.load(f), use_rust=True)
    assert reason is None
    return game.to_dict()


@pytest.fixture(scope="module")
def cleaned_dict():
    return _cleaned("router_d_trains.json")


@pytest.fixture(scope="module")
def converted(cleaned_dict):
    """{skip_forced: (training rows, stats)} for the same cleaned game."""
    out = {}
    for skip_forced in (False, True):
        game = _load_cleaned_game_via_rust(cleaned_dict)
        assert game is not None and game.finished
        stats = {}
        train, val = convert_game_to_training_data(
            game, Encoder_Transformer(), config=ALL_TRAINING, skip_forced=skip_forced, stats=stats
        )
        assert val == []
        out[skip_forced] = (train, stats)
    return out


def _same_row(a, b):
    (state_a, legal_a, pi_a, value_a, price_a), (state_b, legal_b, pi_b, value_b, price_b) = a, b
    assert len(state_a) == len(state_b)
    for x, y in zip(state_a, state_b):
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y
    return legal_a == legal_b and torch.equal(pi_a, pi_b) and torch.equal(value_a, value_b) and price_a == price_b


def test_forced_positions_are_dropped_and_decisions_kept(converted):
    all_rows, all_stats = converted[False]
    kept_rows, kept_stats = converted[True]
    forced = [row for row in all_rows if len(row[1]) == 1]
    decisions = [row for row in all_rows if len(row[1]) > 1]
    assert forced and decisions

    assert all(len(row[1]) > 1 for row in kept_rows)
    assert len(kept_rows) == len(decisions)
    assert all(_same_row(a, b) for a, b in zip(kept_rows, decisions))

    no_skipped_labels = {"unmatched": {}, "ambiguous": {}}
    assert all_stats == {"positions": len(all_rows), "forced_skipped": 0, **no_skipped_labels}
    assert kept_stats == {"positions": len(kept_rows), "forced_skipped": len(forced), **no_skipped_labels}


def test_value_target_is_unchanged(converted):
    (all_rows, _), (kept_rows, _) = converted[False], converted[True]
    value = all_rows[0][3]
    assert torch.isclose(value.sum(), torch.tensor(1.0))
    assert all(torch.equal(row[3], value) for row in all_rows + kept_rows)


def _read_lmdb(path):
    env = lmdb.open(str(path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            return [torch.load(io.BytesIO(lz4.frame.decompress(v)), weights_only=False) for _, v in txn.cursor()]
    finally:
        env.close()


@pytest.mark.parametrize("skip_forced", [True, False], ids=["skip", "keep"])
def test_dataset_conversion_threads_skip_forced(tmp_path, cleaned_dict, converted, skip_forced):
    games = tmp_path / "games"
    games.mkdir()
    (games / "234317.json").write_text(json.dumps(cleaned_dict))
    out = tmp_path / "lmdb"
    convert_games_to_training_dataset(
        str(games), Encoder_Transformer(), out, config=ALL_TRAINING, skip_forced=skip_forced
    )
    rows = _read_lmdb(out / "training")
    assert [row[1] for row in rows] == [row[1] for row in converted[skip_forced][0]]
    assert all(_same_row(a, b) for a, b in zip(rows, converted[skip_forced][0]))


# --- labels ---------------------------------------------------------------------------------------


def _board_tokens(game):
    return {
        (h.id, i): [t and t.corporation_id for t in city.tokens]
        for h in game._game.hexes
        for i, city in enumerate(h.tile.cities)
    }


def _train_holder(game, train_id):
    """Who holds ``train_id``, read off the engine: "depot" (incl. its discard pool) or a corporation."""
    depot = game._game.depot
    if any(t.id == train_id for t in list(depot.trains) + list(depot.discarded)):
        return "depot"
    return next(c.sym for c in game._game.corporations if any(t.id == train_id for t in c.trains))


def _fresh_state(game):
    players = {i + 1: f"Player {i + 1}" for i in range(len(game.players))}
    return RustGameAdapter(RustGame(players, optional_rules=list(game.optional_rules)))


def _replay_labels(cleaned):
    """One record per human action: for decisions, its label and the matching choices; ground truth from the engine."""
    mapper = ActionMapper()
    game = _load_cleaned_game_via_rust(cleaned)
    state = _fresh_state(game)
    records = []
    for pos, action in enumerate(game.raw_actions):
        choices = state.get_factored_choices()
        legal = {int(mapper.index_for_factored(la, state)) for la in choices}
        record = {"pos": pos, "action": action, "matches": []}
        if len(legal) > 1:
            record["index"] = _action_dict_to_factored_index(action, state, choices, mapper)
            record["matches"] = [
                (la, int(mapper.index_for_factored(la, state))) for la in _matching_choices(action, state, choices)
            ]
            record["choices"] = choices
        if action["type"] == "buy_train":
            record["seller"] = _train_holder(state, action["train"])
        tokens_before = _board_tokens(state) if action["type"] == "place_token" else None
        state.process_action(action)
        if tokens_before is not None:
            after = _board_tokens(state)
            record["token_city"] = [key for key, tokens in after.items() if tokens != tokens_before.get(key)]
        records.append(record)
    return records


@pytest.fixture(scope="module")
def labels(cleaned_dict):
    return {
        "router_d_trains": _replay_labels(cleaned_dict),
        "router_two_d_trains": _replay_labels(_cleaned("router_two_d_trains.json")),
    }


def _decisions(labels, action_type):
    return [r for game in labels.values() for r in game if r["action"]["type"] == action_type and "index" in r]


def _record(labels, game, action_id):
    return next(r for r in labels[game] if r["action"].get("id") == action_id)


def test_every_decision_matches_exactly_one_index(labels):
    decisions = [r for game in labels.values() for r in game if "index" in r]
    assert decisions
    for r in decisions:
        assert r["index"] is not None, r["action"]
        assert {index for _, index in r["matches"]} == {r["index"]}, r["action"]


def test_place_token_label_is_the_city_the_token_lands_in(labels):
    tokens = _decisions(labels, "place_token")
    for r in tokens:
        assert len(r["token_city"]) == 1, r["action"]
        (la, _), *_ = r["matches"]
        assert (la.params["hex"], int(la.params["city"])) == r["token_city"][0], r["action"]
    cities = {r["action"]["city"] for r in tokens}
    assert {"57-1-0", "15-1-0"} <= cities  # laid tiles
    assert {"62-0-1", "67-0-1", "59-0-1"} <= cities  # the second city of an OO tile
    assert {"D2-0-0", "D14-0-0"} <= cities  # preprinted tiles


@pytest.mark.parametrize(
    "game, action_id, city, hex_city, index",
    [
        ("router_d_trains", 260, "15-1-0", ("H16", 0), 125),  # also legal: H10
        ("router_d_trains", 507, "D2-0-0", ("D2", 0), 111),  # also legal: B16
        ("router_d_trains", 396, "67-0-1", ("D10", 1), 130),  # also legal: D10 city 0
        ("router_two_d_trains", 141, "59-0-1", ("E11", 1), 132),  # also legal: E11 city 0
    ],
)
def test_place_token_label_pins(labels, game, action_id, city, hex_city, index):
    r = _record(labels, game, action_id)
    assert r["action"]["city"] == city
    sites = {(la.params["hex"], la.params["city"]) for la in r["choices"] if la.type == "PlaceToken"}
    assert hex_city in sites and len(sites) > 1
    assert r["index"] == index
    assert [(la.params["hex"], la.params["city"]) for la, _ in r["matches"]] == [hex_city]


def test_buy_train_label_names_the_seller(labels):
    buys = _decisions(labels, "buy_train")
    for r in buys:
        (la, _), *_ = r["matches"]
        traded_in = r["action"].get("exchange")
        assert la.entity["source"] == r["seller"], r["action"]
        assert la.entity.get("exchange") == (traded_in.split("-")[0] if traded_in else None), r["action"]
    sellers = {r["seller"] for r in buys}
    assert "depot" in sellers and len(sellers) > 2


@pytest.mark.parametrize(
    "game, action_id, entity, index",
    [
        # From another corporation; ERIE also had a 4-train for sale (the old matcher's pick).
        ("router_d_trains", 244, {"corp": "PRR", "source": "C&O", "train": "4"}, 26178),
        # A D-train for a traded-in 4-train vs. the full-price D also on offer.
        ("router_d_trains", 271, {"corp": "B&O", "source": "depot", "train": "D", "exchange": "4"}, 26536),
        ("router_d_trains", 389, {"corp": "B&O", "source": "depot", "train": "D"}, 26535),
        # From the depot's discard pool (its own per-type slot), and a regular depot buy beside it.
        ("router_two_d_trains", 236, {"corp": "C&O", "source": "depot", "train": "3"}, 25809),
        ("router_two_d_trains", 235, {"corp": "C&O", "source": "depot", "train": "5"}, 25807),
    ],
)
def test_buy_train_label_pins(labels, game, action_id, entity, index):
    r = _record(labels, game, action_id)
    assert [dict(la.entity) for la, _ in r["matches"]] == [entity]
    assert r["index"] == index


def test_company_tile_lay_is_labelled_as_the_private(labels):
    r = _record(labels, "router_d_trains", 49)
    assert r["action"]["entity"] == "CS" and r["action"]["entity_type"] == "company"
    assert [dict(la.entity) for la, _ in r["matches"]] == [{"private": "CS"}]
    assert r["index"] is not None


def test_an_action_matching_several_indices_is_rejected(labels, cleaned_dict):
    """An unknown train id can't name its seller: both 4-train sellers match, so no label."""
    r = _record(labels, "router_d_trains", 244)
    game = _load_cleaned_game_via_rust(cleaned_dict)
    state = _fresh_state(game)
    for action in game.raw_actions[: r["pos"]]:
        state.process_action(action)
    unknown_train = {**r["action"], "train": "4-99"}
    with pytest.raises(AmbiguousActionMatch):
        _action_dict_to_factored_index(unknown_train, state, state.get_factored_choices(), ActionMapper())


# --- validation split -----------------------------------------------------------------------------


def test_validation_split_is_a_stable_function_of_the_game_id():
    ids = [str(i) for i in range(20000)]
    split = [in_validation_split(game_id, 0.05) for game_id in ids]
    assert split == [in_validation_split(game_id, 0.05) for game_id in ids]
    assert 0.04 < sum(split) / len(ids) < 0.06
    # Raising the percentage only moves games into validation.
    assert all(in_validation_split(game_id, 0.10) for game_id, held_out in zip(ids, split) if held_out)


@pytest.mark.parametrize("game_id, held_out", [("234317", False), ("1", True)])
def test_conversion_routes_a_game_by_its_id(cleaned_dict, game_id, held_out):
    assert in_validation_split(game_id, 0.5) == held_out
    config = TrainingConfig(pretrain_validation_percentage=0.5)
    train, val = convert_game_to_training_data(
        _load_cleaned_game_via_rust(cleaned_dict), Encoder_Transformer(), config=config, game_id=game_id
    )
    assert (bool(train), bool(val)) == (not held_out, held_out)
