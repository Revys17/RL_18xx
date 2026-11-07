"""Two-player 1830 (``--num-players 2``): the definitions that differ from four players -- placement
points, learner / league seats, eval seatings and their fair score, start files by player count --
plus tiny CPU games of the policy-gradient and search-learning generators, and the four-player
defaults they must leave unchanged."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import policy_gradient as pg  # noqa: E402
from rl18xx.agent.alphazero import search_data as sd  # noqa: E402
from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero import start_positions as sp  # noqa: E402
from rl18xx.agent.alphazero import value_trainer as vt  # noqa: E402
from rl18xx.agent.alphazero.composite_model import terminal_utility  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _script(name):
    """A script under scripts/ as a module (they import their siblings, so scripts/ goes on the path)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_two_player_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def h2h():
    return _script("eval_head_to_head")


@pytest.fixture(scope="module")
def epo():
    return _script("eval_policy_only")


@pytest.fixture(scope="module")
def rt():
    return _script("rollout_targets")


@pytest.fixture
def start_file(tmp_path):
    """An empty start file: every game takes a random auction ending."""
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


# ----------------------------------------------------------------------------- placement points


def test_placement_points_by_player_count():
    assert pg.default_placement_points(4) == (1.0, 0.5, 0.2, 0.0)
    assert pg.default_placement_points(2) == (1.0, 0.0)
    # Four players unchanged; two players: the place is the win, a tie splits it.
    assert pg.placement_points([900, 500, 1200, 100]).tolist() == [0.5, 0.2, 1.0, 0.0]
    assert pg.placement_points([900, 1200]).tolist() == [0.0, 1.0]
    assert pg.placement_points([800, 800]).tolist() == [0.5, 0.5]
    # Explicit points are cut to the seats, as before.
    assert pg.placement_points([900, 1200], (1.0, 0.3, 0.1, 0.0)).tolist() == [0.3, 1.0]
    # Other counts keep the four-player points (padded / cut).
    assert pg.placement_points([3, 2, 1]).tolist() == [1.0, 0.5, 0.2]


def test_rewards_and_score_targets_use_the_player_counts_points():
    win = np.array([0.0, 1.0])
    assert pg.shaped_reward("placement", win, [900, 1200], [False, False], {}, [0, 0], {0: []}).tolist() == [0, 1]
    assert sl.score_target(np.array([1500.0, 700.0]), "placement").tolist() == [1.0, 0.0]
    assert sl.score_target(np.array([1500.0, 700.0, 900, 100]), "placement") == pytest.approx([1.0, 0.2, 0.5, 0.0])
    assert sl.score_target(np.array([1500.0, 700.0]), "win") is None


def test_terminal_utility_by_player_count():
    assert terminal_utility("score", "placement") == (1.0, (1.0, 0.5, 0.2, 0.0))
    assert terminal_utility("score", "placement", 4) == (1.0, (1.0, 0.5, 0.2, 0.0))
    assert terminal_utility("score", "placement", 2) == (1.0, (1.0, 0.0))
    assert terminal_utility("blend=0.3", "placement:1,0.3,0.1,0", 2) == (0.3, (1.0, 0.3, 0.1, 0.0))
    assert terminal_utility("win", "placement", 2) == (0.0, ())
    s = dict(
        target="cq",
        value_head="score",
        reward="placement",
        n_full=8,
        parallel_readouts=4,
        c_puct_init=0.05,
        mean_q=True,
        fpu_reduction=0.0,
        dedupe_leaves=True,
        prior_temperature=1.0,
        gumbel_k=4,
        gumbel_c_visit=50.0,
        gumbel_c_scale=1.0,
        gumbel_noise_scale=1.0,
        max_decisions=100,
    )
    four, two = sl.search_config(s), sl.search_config(dict(s, num_players=2))
    assert four.player_count_distribution == {4: 1.0} and four.terminal_score_points == (1.0, 0.5, 0.2, 0.0)
    assert two.player_count_distribution == {2: 1.0} and two.terminal_score_points == (1.0, 0.0)


# ----------------------------------------------------------------------------- value targets


def test_value_targets_follow_the_datas_seats(rt):
    # Two players: one column per seat, canonical frame = the mover first.
    assert rt.canon(np.array([0.25, 0.75]), 1).tolist() == [0.75, 0.25]
    assert rt.canon(np.array([1.0, 2.0, 3.0, 4.0]), 1).tolist() == [2.0, 3.0, 4.0, 1.0]
    worth = np.array([[[900.0, 1200.0], [np.nan, np.nan]]])
    assert np.array_equal(rt.placement_rows(worth), np.array([[[0.0, 1.0], [np.nan, np.nan]]]), equal_nan=True)
    assert rt.canon_rows(np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([0, 1])).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    win, place = vt.canonical_targets(
        np.array([[0.0, 1.0]], np.float32), np.array([[700.0, 1500.0]]), np.array([0, 0]), np.array([0, 1])
    )
    assert win.tolist() == [[0.0, 1.0], [1.0, 0.0]] and place.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_value_trainer_counts_only_the_runs_player_count():
    index = vt.GameIndex(
        paths=["a"], shard=np.zeros(4, np.int32), slot=np.arange(4, dtype=np.int32),
        crc=np.array([900, 901, 902, 5], np.uint32), players=np.array([4, 2, 2, 2], np.int8),
    )  # fmt: skip
    assert index.trainable(30).tolist() == [0]
    assert index.trainable(30, 2).tolist() == [1, 2]
    cfg = vt.ValueTrainerConfig(run_dir="x", num_players=2, rolling_games=10, held_out_per_mille=30)
    assert vt.rolling_games(index, cfg).tolist() == [3]
    with pytest.raises(ValueError, match="4-player positions"):
        vt.check_players({"win": np.zeros((3, 4))}, 2, "--anchor-data")
    vt.check_players({"win": np.zeros((3, 2))}, 2, "--anchor-data")


# ----------------------------------------------------------------------------- seats


def test_policy_gradient_seats():
    assert pg.default_learner_seats(4) == 2 and pg.default_learner_seats(2) == 1
    assert pg.default_learner_seat_weights(4) == pg.DEFAULT_LEARNER_SEAT_WEIGHTS
    assert pg.default_learner_seat_weights(2) == {1: 1.0}
    rng = random.Random(0)
    league = {"slots": ["opp0"], "slot_labels": {"opp0": "m"}, "learner_seat_weights": {1: 1.0}}
    for _ in range(50):
        learner, models, labels = pg._deal_seats(
            {"num_players": 2, "pool_ready": True, "sl_opponent_fraction": 0.5, "learner_seats": 1}, rng
        )
        assert len(learner) == 1 and len(models) == 2 and models.count("learner") == 1
        learner, models, labels = pg._deal_seats({"num_players": 2, "league": league}, rng)
        assert len(learner) == 1 and models.count("opp0") == 1 and list(labels.values()) == ["m"]


def test_policy_gradient_rejects_seats_that_dont_fit(tmp_path):
    cfg = pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", num_players=2, learner_seats=2)
    with pytest.raises(SystemExit, match="learner-seats"):
        pg.check_players(cfg)
    cfg.learner_seats = 1
    cfg.population, cfg.learner_seat_weights = "pop.json", {1: 0.5, 2: 0.5}
    with pytest.raises(SystemExit, match="learner-seat-weights"):
        pg.check_players(cfg)
    cfg.learner_seat_weights = None
    cfg.start_positions = str(tmp_path / "missing.jsonl")  # not there: not checked
    pg.check_players(cfg)


def test_league_scores_take_the_games_player_count():
    weights = sl.default_league_seat_weights(2)
    assert weights == {1: 1.0} and sl.default_league_seat_weights(4) == sl.DEFAULT_LEAGUE_SEAT_WEIGHTS
    league = {"members": {"m": {"server": "lg0"}}, "weights": {"m": 1.0}, "share": 1.0, "seat_weights": weights}
    rng = random.Random(1)
    for _ in range(20):
        servers, members = sl.deal_seats({"num_players": 2, "league": league}, rng)
        assert sorted(servers) == ["learner", "lg0"] and sorted(members) == ["m", sl.MAIN]
    servers, _ = sl.deal_seats({"num_players": 2, "opponent_servers": ["x"]}, rng)
    assert sorted(servers) == ["learner", "x"]  # the older hook: half the table
    assert sl.deal_seats({"num_players": 2}, rng) == (["learner"] * 2, [sl.MAIN] * 2)
    result = sl.league_result({"seat_members": ["m", sl.MAIN], "fractions": [0.4, 0.6], "win_share": [0.0, 1.0]})
    assert result == {"member": "m", "players": 2, "main_seats": 1, "win": 1.0, "worth": 0.6, "pairwise": 1.0}
    scores = {}
    sl.add_league_score(scores, result, "fixed")
    assert scores["m"]["fair"] == 0.5 and scores["m"]["win_excess_sq"] == 0.25
    # A league learner seat's score is a multiple of its fair share: 1 of 2 seats is fair at 0.5.
    s = pg._Scores(10)
    s.add({"win_share": [1.0, 0.0], "learner_seats": [0], "opponent": "league"})
    s.add({"win_share": [0.5, 0.5], "learner_seats": [1], "opponent": "league"})
    assert s.summary()["score_vs_league"] == pytest.approx(1.5)


# ----------------------------------------------------------------------------- eval seatings


def test_eval_seatings(h2h, epo):
    for module in (h2h, epo):
        assert module.arrangements(1, 2) == [("A", "B"), ("B", "A")]
        assert len(module.arrangements(2, 4)) == 6 and len(module.arrangements(1, 4)) == 4
        assert module.arrangements(2) == module.arrangements(2, 4)  # the four-player default
        with pytest.raises(ValueError):
            module.arrangements(2, 2)
    records = [
        {"seats": ["A", "B"], "win_share": [1.0, 0.0], "worth_share": [0.6, 0.4], "termination": "finished",
         "engine_moves": 10, "decisions": {}},
        {"seats": ["B", "A"], "win_share": [1.0, 0.0], "worth_share": [0.6, 0.4], "termination": "finished",
         "engine_moves": 10, "decisions": {}},
    ]  # fmt: skip
    summary = h2h.summarize("a vs b", {"A": "a", "B": "b"}, records)
    assert summary["A_score"] == 0.5 and summary["A_score_equal"] == 0.5
    assert summary["by_arrangement"] == {"AB": 1.0, "BA": 0.0}
    assert summary["seat_win_rate"] == {"A0": 1.0, "A1": 0.0, "B0": 1.0, "B1": 0.0}
    summary = epo.summarize("a", "b", [dict(r, decisions=5) for r in records])
    assert summary["players"] == 2 and summary["a_seats"] == 1 and summary["A_score_equal"] == 0.5


# ----------------------------------------------------------------------------- start files


def test_start_files_by_player_count(tmp_path):
    assert sp.default_start_positions(4) == "human_games/start_positions_1830_4p_v2.jsonl"
    assert sp.default_midgame_positions(4) == "human_games/midgame_positions_1830_4p_v2.jsonl"
    assert sp.default_start_positions(2) == "human_games/start_positions_1830_2p_v2.jsonl"
    assert sp.default_midgame_positions(2) == "human_games/midgame_positions_1830_2p_v2.jsonl"
    four = tmp_path / "four.jsonl"
    four.write_text(json.dumps({"id": "1", "num_players": 4, "actions": []}) + "\n")
    two = tmp_path / "two.jsonl"
    two.write_text("".join(json.dumps({"id": str(i), "num_players": 2, "actions": []}) + "\n" for i in range(3)))
    assert sp.file_player_counts(four) == {4: 1} and sp.file_player_counts(two) == {2: 3}
    sp.check_start_files(4, four, 0.2)
    sp.check_start_files(2, two, 0.2, two, 0.25)
    with pytest.raises(SystemExit, match="no 2-player games"):
        sp.check_start_files(2, four, 0.2)
    with pytest.raises(SystemExit, match="midgame_positions_1830_2p_v2"):
        sp.check_start_files(2, two, 0.2, four, 0.25)
    sp.check_start_files(2, four, 1.0)  # only random starts: the file isn't drawn from
    sp.check_start_files(2, two, 0.2, four, 0.0)  # no mid-game starts
    sp.check_start_files(2, tmp_path / "missing.jsonl", 0.2)  # not there: the generator fails as before


def test_two_player_starts(tmp_path):
    # Random auction endings: both players hold the privates they paid for, from $1,200 each.
    for seed in range(40):
        actions = sp.random_start_actions(2, random.Random(seed))
        game = sp.apply_actions(sp.new_game(2), actions)
        summary = sp.position_summary(game)
        assert summary["round"] == "Stock" and set(summary["cash"]) == {1, 2}
        assert set(summary["owners"].values()) <= {1, 2}
    # A file of 2-player starts: only those are drawn (by label), the rest random.
    actions = sp.random_start_actions(2, random.Random(99))
    path = tmp_path / "start_positions_1830_2p_v2.jsonl"
    path.write_text(json.dumps({"id": "g2", "num_players": 2, "actions": actions}) + "\n")
    labels = set()
    for i in range(30):
        start, game, skipped = sp.replayed_start(2, str(path), 0.3, rng=random.Random(i))
        labels.add(start.label)
        assert skipped == 0 and len(game.players) == 2 and game._game.round.round_type == "Stock"
    assert labels == {"human:g2", sp.RANDOM_START}
    start, game, _ = sp.replayed_start(2, str(path), 0.3, rng=random.Random(0), auction_fraction=1.0)
    assert start.label == sp.AUCTION_START and game._game.round.round_type == "Auction"


# ----------------------------------------------------------------------------- generators


def test_policy_gradient_two_player_games(monkeypatch, start_file):
    from .test_policy_gradient import CRITIC_VALUE, _Client, _settings

    monkeypatch.setattr(pg, "_CLIENTS", {"learner": _Client(), "sl": _Client(), "pool": _Client()})
    result = pg.play_pg_games(3, _settings(start_file, num_players=2, learner_seats=1, reward="placement"))
    assert len(result["games"]) == 3
    for game in result["games"]:
        assert len(game["learner_seats"]) == 1 and len(game["win_share"]) == 2
        assert sorted(game["reward"]) in ([0.0, 1.0], [0.5, 0.5])  # placement points 1 / 0 (a tie splits)
        for encoded, _, _, _, _, advantage, fractions, score_target in game["rows"]:
            assert int(encoded[6]) in game["learner_seats"] and int(encoded[7]) == 2
            assert fractions.shape == (2,) and score_target.shape == (2,)
            assert advantage == pytest.approx(game["reward"][int(encoded[6])] - CRITIC_VALUE)


def test_search_learning_two_player_games_save_and_rebuild(tmp_path, start_file):
    from rl18xx.agent.alphazero.lockstep import ModelEvaluator

    from .search_test_utils import HashNet
    from .test_search_data import _config, _save, assert_same_row

    cfg = _config(start_file, num_players=2, n_fast=2, auction_fraction=0.3, max_decisions=60)
    settings = sl.worker_settings(cfg, version=3)
    assert settings["num_players"] == 2
    gen = sl.Generator(settings, {"learner": ModelEvaluator(HashNet())})
    games = gen.run(6.0)["games"]
    assert games
    rows = [r for g in games for r in g["rows"]]
    assert rows and all(r.fractions.shape == (2,) and r.score_target.shape == (2,) for r in rows)
    assert all(sorted(r.score_target.tolist()) in ([0.0, 1.0], [0.5, 0.5]) for r in rows)
    assert all(len(g["seat_servers"]) == 2 and len(g["win_share"]) == 2 for g in games)
    saved = _save(games, tmp_path / "data", cfg)
    assert {s.meta["num_players"] for s in saved} == {2}
    for g, s in zip(games, saved):
        rebuilt = sd.rebuild_rows(s)
        assert len(rebuilt) == len(g["rows"])
        for a, b in zip(g["rows"], rebuilt):
            assert_same_row(a, b)
