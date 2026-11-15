"""The advisor's play systems with title-generic checkpoints (``generic:<path>``; rl18xx/agent/alphazero/
game_network.py): policy-only moves (``--play-mode policy``) and searches whose leaves go to the networks as
games, with tiny untrained CPU models. The legacy play systems' behaviour is tested in test_autoplay.py."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl18xx.agent.advisor.advisor import Advisor, AdvisorModels, price_options
from rl18xx.agent.advisor.autoplay import bot_position
from rl18xx.agent.advisor.live_game import LiveGame
from rl18xx.agent.alphazero import model_generic as mg
from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.game_network import GameNetwork, GenericNetwork, absolute_values

from .conftest import truncated
from .test_autoplay import play, site_view, wait

G = "generic:"


def load(advisor_checkpoints, **play) -> AdvisorModels:
    return AdvisorModels.load(*advisor_checkpoints, device="cpu", **play)


@pytest.fixture(scope="module")
def generic_policy_models(advisor_checkpoints, generic_checkpoints):
    """Generic learner + generic critic play every seat, policy-only (the most likely move)."""
    learner, critic = (f"{G}{generic_checkpoints[k]}" for k in ("learner", "critic"))
    return load(
        advisor_checkpoints,
        play_policy=learner,
        play_value=critic,
        play_auction_policy=learner,
        play_auction_value=critic,
        play_mode="policy",
    )


@pytest.fixture(scope="module")
def generic_search_models(generic_policy_models):
    return dataclasses.replace(generic_policy_models, play_mode="search", play_value_head="score")


def positions(game: dict, every: int = 9) -> list:
    """Live games at every ``every``-th action of the test game (decisions only)."""
    out = []
    for count in range(0, len(game["actions"]) + 1, every):
        live = LiveGame(truncated(game, count))
        if live.error is None and not live.finished:
            out.append(live)
    return out


def legal_of(live) -> list:
    return [int(i) for i in bot_position(live)._game.factored_legal_indices()]


# ------------------------------------------------------------------ the bridge
def test_generic_moves_and_prices_are_the_legacy_layouts(game):
    """For 1830 a generic model's legal rows are the frozen layout's indices, a priced row's range is the
    legacy price slot's and its price cells the slot's cells; the encoder's seat rotation is the legacy
    encoder's, so a generic value maps to the players as the legacy unrotation does."""
    from rl18xx.agent.alphazero.mcts import _rust_encode, unrotate_value

    priced = 0
    for live in positions(game, every=5):
        g = live.rust
        legal = mg.legal_actions(g)
        assert sorted(legal.indices.tolist()) == sorted(int(i) for i in g._game.factored_legal_indices())
        arrays = mg.build_inputs([g], [legal])
        encoded = _rust_encode(g)
        rotation, n = int(arrays["meta"][0, 5]), int(arrays["meta"][0, 4])
        assert (rotation, n) == (int(encoded[6]), int(encoded[7]))
        v = np.arange(1, 7, dtype=np.float32)
        expected = unrotate_value(v.copy(), rotation, n)
        expected[n:] = 0
        assert np.allclose(absolute_values(v, rotation, n, [p.id for p in g.players]), expected)
        for k, index in enumerate(legal.indices.tolist()):
            if legal.rows[k, mg.C_PRICE_KIND] < 0:
                continue
            priced += 1
            slot, lo, hi = g._game.price_head_slot_for_index(index)
            assert tuple(int(x) for x in legal.price_range[k, :2]) == (lo, hi)
            ours, theirs = legal.price_cells(k), price_pmf.price_cells(price_pmf.SLOT_TYPES[slot], lo, hi)
            assert np.array_equal(ours.nonempty, theirs.nonempty) and np.array_equal(ours.counts, theirs.counts)
    assert priced > 0


def test_a_game_network_of_legacy_parts_is_the_legacy_composite(advisor_models, game):
    """Leaves handed over as games give the legacy networks' own priors, prices and values (unrotated)."""
    from rl18xx.agent.advisor.advisor import _RoundPolicy
    from rl18xx.agent.alphazero.composite_model import PolicyValueComposite
    from rl18xx.agent.alphazero.mcts import _rust_encode, unrotate_value

    lives = positions(game, every=23)
    games = [live.rust for live in lives]
    legal = [legal_of(live) for live in lives]
    m = advisor_models
    ours = GameNetwork(m.policy, m.value, auction_policy=m.auction_policy, value_head="blend=0.3")
    priors, _, values = ours.run_many_games(games, legal)
    theirs = PolicyValueComposite(_RoundPolicy(m.policy, m.auction_policy), m.value, value_head="blend=0.3")
    encoded = [_rust_encode(g) for g in games]
    with torch.no_grad():
        probs, _, expected = theirs.run_many_encoded(encoded)
    for b, (g, e) in enumerate(zip(games, encoded)):
        p = probs[b].numpy()[legal[b]]
        assert np.allclose(priors[b, legal[b]], p / p.sum(), atol=1e-6)
        assert np.allclose(values[b], unrotate_value(expected[b].numpy(), int(e[6]), int(e[7])), atol=1e-6)
        assert np.allclose(
            ours.last_price_components["price_logits"][b].numpy(),
            theirs.last_price_components["price_logits"][b].numpy(),
        )


def test_a_generic_network_scores_the_legal_moves_and_its_prices_reach_their_slots(generic_checkpoints, game):
    learner = GenericNetwork(generic_checkpoints["learner"])
    critic = GenericNetwork(generic_checkpoints["critic"])
    network = GameNetwork(learner, critic, policy_temperature=0.5, auction_policy_temperature=0.5)
    lives = positions(game, every=11)
    evaluations = network.evaluate([live.rust for live in lives], [legal_of(live) for live in lives])
    slotted = 0
    for live, e in zip(lives, evaluations):
        assert len(e.probs) == len(e.legal) and np.isclose(e.probs.sum(), 1.0)
        n = live.num_players
        assert np.all(e.value[n:] == 0) and np.isclose(e.value[:n].sum(), 1.0, atol=1e-4)  # win shares
        for index in e.legal.tolist():
            slot = live.rust._game.price_head_slot_for_index(index)
            if slot is not None and slot[1] < slot[2]:
                assert np.any(e.price_logits[int(slot[0])] != 0)
                slotted += 1
    assert slotted > 0
    # A critic only values: its forward without legal lists matches its forward with them.
    games = [live.rust for live in lives[:3]]
    values_only = critic.forward(games)
    with_policy = critic.forward(games, [legal_of(live) for live in lives[:3]])
    for a, b in zip(values_only, with_policy):
        assert np.allclose(a.win, b.win, atol=1e-5) and np.allclose(a.score, b.score, atol=1e-5)


def test_generic_checkpoints_serve_the_play_systems_only(advisor_checkpoints, generic_checkpoints):
    policy, auction, value = advisor_checkpoints
    with pytest.raises(ValueError, match="--policy"):
        AdvisorModels.load(f"{G}{generic_checkpoints['learner']}", auction, value, device="cpu")
    with pytest.raises(ValueError, match="--search-value"):
        AdvisorModels.load(policy, auction, value, device="cpu", search_value=f"{G}{generic_checkpoints['critic']}")
    with pytest.raises(ValueError, match="play mode"):
        load(advisor_checkpoints, play_mode="greedy")


def test_the_command_line_serves_generic_play_systems(advisor_checkpoints, generic_checkpoints):
    """``main.py advisor --play-* generic:<path> --play-mode policy --play-temperature T`` -> the models
    (``app.load_models``, what ``run`` serves)."""
    import sys

    from rl18xx.agent.advisor.app import load_models

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    try:
        import main
    finally:
        sys.path.remove(str(root))
    policy, auction, value = (str(p) for p in advisor_checkpoints)
    learner, critic = (f"{G}{generic_checkpoints[k]}" for k in ("learner", "critic"))
    argv = ["advisor", "--policy", policy, "--auction-policy", auction, "--value", value, "--search-value", value]
    argv += ["--play-policy", learner, "--play-value", critic, "--play-auction-policy", learner]
    argv += ["--play-auction-value", critic, "--play-value-head", "score"]
    policy_only = argv + ["--play-mode", "policy", "--play-temperature", "0.5"]
    models = load_models(main.build_parser().parse_args(policy_only), "cpu")
    assert models.play_mode == "policy" and models.play_temperature == 0.5 and models.play_value_head == "score"
    assert isinstance(models.play_auction_value, GenericNetwork) and models.labels["play_mode"] == "policy@0.5"
    default = load_models(main.build_parser().parse_args(argv), "cpu")
    assert default.play_mode == "search" and default.labels["play_mode"] == "search"


def test_the_swap_script_rewrites_a_services_play_options():
    """``scripts/launch/2026-11/bridge/swap_bots_generic.sh``'s rewrite of a service's command line (offline)."""
    import subprocess

    script = Path(__file__).resolve().parents[3] / "scripts" / "launch" / "2026-11" / "bridge" / "swap_bots_generic.sh"
    old = (
        "/old/.venv/bin/python main.py advisor --device cpu --port 5004 --policy /p.pth --play-policy /l.pth "
        "--play-value /c.pth --play-auction-policy /l.pth --play-auction-value /c.pth --bots --play-value-head score"
    )
    out = subprocess.run(
        ["bash", str(script), "/deploy/new", "/run/learner/9.pth", "/run/critic/9.pth", "--policy-only"],
        env={"SWAP_TEST_CMDLINE": old, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[:9] == "/deploy/new/.venv/bin/python main.py advisor --device cpu --port 5004 --policy /p.pth".split()
    assert "--bots" in out and out[out.index("--play-value-head") + 1] == "score"
    for flag, path in (
        ("--play-policy", "/run/learner/9.pth"),
        ("--play-value", "/run/critic/9.pth"),
        ("--play-auction-policy", "/run/learner/9.pth"),
        ("--play-auction-value", "/run/critic/9.pth"),
    ):
        assert out.count(flag) == 1 and out[out.index(flag) + 1] == f"generic:{path}"
    assert out[out.index("--play-mode") + 1] == "policy" and out[out.index("--play-temperature") + 1] == "0"


def test_the_play_systems_load_generic_checkpoints(generic_policy_models, generic_checkpoints):
    m = generic_policy_models
    assert isinstance(m.play_policy, GenericNetwork) and isinstance(m.play_value, GenericNetwork)
    assert m.play_auction_policy is m.play_policy  # one copy of a checkpoint given twice
    assert m.play_policy_temperature == 0.5  # the run's learner_temperature
    assert m.labels["play_policy"].startswith(G) and m.labels["play_policy"].endswith("learner/1")
    assert m.labels["play_mode"] == "policy@0"
    assert m.is_game_system("main") and m.is_game_system("auction") and not m.is_game_system(None)
    assert not isinstance(m.policy, GenericNetwork) and not isinstance(m.value, GenericNetwork)  # advice: legacy


# ------------------------------------------------------------ policy-only play
def test_a_generic_policy_plays_its_most_likely_move(generic_policy_models, game):
    advisor = Advisor(generic_policy_models)
    live = LiveGame(game)
    legal = legal_of(live)
    job = wait(advisor.start_move(live, site_view(live, last_id=267), seats=[3], readouts=400))
    assert job.status == "done", job.error
    result = job.result
    assert result["kind"] == "policy" and result["system"] == "main" and result["visits"] == 0
    evaluation = generic_policy_models.game_network("main").evaluate([bot_position(live)], [legal])[0]
    assert result["index"] == legal[int(np.argmax(evaluation.probs))]
    assert np.isclose(result["share"], evaluation.probs.max())
    action = result["action"]
    assert action["entity"] == live.acting()["id"] and "id" not in action
    assert result["description"] and result["actor"] and len(result["win"]) == live.num_players
    # The site's game takes it.
    game["actions"].append({**action, "id": len(game["actions"]) + 1, "user": 103})
    assert live.update(game) and live.error is None


def test_a_generic_policy_bids_in_the_auction_at_its_price_heads_price(generic_policy_models, game):
    advisor = Advisor(generic_policy_models)
    bids = 0
    for count in range(0, 12):
        live = LiveGame(truncated(game, count))
        if live.rust._game.round.round_type != "Auction":
            break
        seat = live.acting()["player_id"]
        job = wait(advisor.start_move(live, site_view(live, actions=["bid", "pass"]), seats=[seat], readouts=8))
        assert job.status == "done", job.error
        result = job.result
        assert result["system"] == "auction" and result["action"]["entity"] == live.user_id(seat)
        if result["action"]["type"] != "bid":
            continue
        bids += 1
        rs = bot_position(live)._game
        evaluation = generic_policy_models.game_network("auction").evaluate([bot_position(live)])[0]
        lo, hi = rs.price_range_for_index(result["index"])
        assert lo <= result["action"]["price"] <= hi
        if lo < hi:
            assert result["action"]["price"] == price_options(rs, result["index"], evaluation.price_logits)["price"]
    assert bids > 0


def test_a_generic_policy_at_a_temperature_samples_legal_moves(generic_policy_models, game):
    models = dataclasses.replace(generic_policy_models, play_temperature=1.0)
    advisor = Advisor(models)
    live = LiveGame(truncated(game, 0))
    legal = legal_of(live)
    seen = set()
    for i in range(12):
        job = wait(advisor.start_move(live, site_view(live, last_id=i, actions=["bid", "pass"]), seats=[1], readouts=1))
        assert job.status == "done", job.error
        assert job.result["index"] in legal and job.result["temperature"] == 1.0
        seen.add((job.result["index"], job.result["action"].get("price")))
    assert len(seen) > 1


def test_a_legacy_policy_plays_policy_only_too(advisor_checkpoints, game):
    advisor = Advisor(load(advisor_checkpoints, play_mode="policy"))
    live = LiveGame(game)
    job = wait(advisor.start_move(live, site_view(live, last_id=267), seats=[3], readouts=400))
    assert job.status == "done", job.error
    assert job.result["kind"] == "policy" and job.result["index"] in legal_of(live)


SHUFFLE = Path(__file__).resolve().parents[2] / "fixtures" / "1830" / "train_shuffle_game.json"


@pytest.mark.parametrize("mode", ["policy", "search"])
def test_a_generic_bot_obeys_the_rule_against_train_shuffling(generic_policy_models, mode):
    from tests.test_train_shuffle_rule import _expected, _first_refused

    data = json.loads(SHUFFLE.read_text())
    i = _first_refused(data)
    buyer, seller, _ = _expected(data)[i]
    live = LiveGame({"id": "hs_shuffle", "title": "1830", "players": data["players"], "actions": data["actions"][:i]})
    assert live.error is None and live.acting()["id"] == buyer
    advisor = Advisor(dataclasses.replace(generic_policy_models, play_mode=mode))
    site = site_view(live, last_id=i, step="BuyTrain", actions=["buy_train", "pass"])
    job = wait(advisor.start_move(live, site, seats=[live.acting()["player_id"]], readouts=8))
    assert job.status == "done", job.error
    action = job.result["action"]
    if action["type"] == "buy_train":
        assert live.rust.train_by_id(action["train"]).owner.id != seller, action
    assert not live.rust._game.no_buyback_rule


# ---------------------------------------------------------------- search
def test_a_generic_system_searches(generic_search_models, game):
    advisor = Advisor(generic_search_models)
    live = LiveGame(game)
    legal = legal_of(live)
    job = wait(advisor.start_move(live, site_view(live, last_id=267), seats=[3], readouts=24), seconds=120)
    assert job.status == "done", job.error
    result = job.result
    assert result["kind"] == "search" and result["system"] == "main"
    assert result["visits"] >= 24 and result["index"] in legal
    assert result["action"]["entity"] == live.acting()["id"]
    assert advisor._trees[live.game_id][1] == "main"  # its tree kept for the model's next move


def test_a_generic_system_searches_the_auction(generic_search_models, game):
    advisor = Advisor(generic_search_models)
    live = LiveGame(truncated(game, 0))
    job = wait(advisor.start_move(live, site_view(live, actions=["bid", "pass"]), seats=[1, 2, 3, 4], readouts=16))
    assert job.status == "done", job.error
    result = job.result
    assert result["system"] == "auction" and result["visits"] >= 16
    assert result["action"]["entity"] == 101 and result["action"]["type"] in ("bid", "pass")


def test_a_generic_policy_searches_with_a_legacy_value(advisor_checkpoints, generic_checkpoints, game):
    """``--play-policy generic:...`` alone: the play value stays the (legacy) search value."""
    models = load(advisor_checkpoints, play_policy=f"{G}{generic_checkpoints['learner']}")
    assert not isinstance(models.play_value, GenericNetwork) and models.is_game_system("main")
    live = LiveGame(game)
    job = wait(Advisor(models).start_move(live, site_view(live, last_id=267), seats=[3], readouts=16), seconds=120)
    assert job.status == "done", job.error
    assert job.result["kind"] == "search" and job.result["visits"] >= 16


def test_a_generic_systems_next_move_goes_on_with_its_kept_tree(generic_search_models, game):
    advisor = Advisor(generic_search_models)
    live = LiveGame(game)
    searched = play({s: advisor for s in (1, 2, 3, 4)}, live, game, {s: 12 for s in (1, 2, 3, 4)}, 4)
    assert searched[0][1]["inherited"] == 0
    assert any(r["inherited"] > 0 for _, r in searched[1:]), [r["inherited"] for _, r in searched]
