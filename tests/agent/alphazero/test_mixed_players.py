"""Mixed player counts in one policy-gradient run (``--player-mix``, gen6): the per-task count schedule,
per-count start files, rewards, seats and limits, and training batches whose rows mix counts (padded to
the network's 6-player layout, the same outputs as one-count batches)."""

import json
import math

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import policy_gradient as pg  # noqa: E402

from .test_policy_gradient import _Client, _settings  # noqa: E402


@pytest.fixture
def start_file(tmp_path):
    """An empty start file: every game takes a random auction ending."""
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


def test_count_schedule_hands_out_the_mix():
    shares = {2: 0.1, 3: 0.2, 4: 0.4, 5: 0.2, 6: 0.1}
    schedule = pg.CountSchedule(shares)
    for k in range(1, 201):
        schedule.next()
        # Never more than one task away from the mix, at every point.
        assert all(abs(schedule.handed[n] - share * k) <= 1.0 for n, share in shares.items())
    assert dict(schedule.handed) == {2: 20, 3: 40, 4: 80, 5: 40, 6: 20}
    one = pg.CountSchedule({4: 1.0})
    assert {one.next() for _ in range(5)} == {4}


def test_config_mix_and_paths():
    cfg = pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", player_mix={"2": 1, "4": 3})
    assert cfg.player_mix == {2: 1, 4: 3}  # JSON's string keys back to counts
    assert pg.player_counts(cfg) == {2: 0.25, 4: 0.75}
    assert pg.player_counts(pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", num_players=2)) == {2: 1.0}
    assert pg.count_path("human_games/start_positions_1830_{n}p_v2.jsonl", 5) == (
        "human_games/start_positions_1830_5p_v2.jsonl"
    )
    assert pg.count_path("plain.jsonl", 5) == "plain.jsonl" and pg.count_path(None, 5) is None
    cfg = pg.PGConfig(
        policy_checkpoint="p", value_checkpoint="v", player_mix={3: 1, 6: 1},
        max_decisions=1000, max_decisions_by_players={"6": 1500}, auction_stall_by_players={6: 200},
        auction_stall_decisions=150,
    )
    assert pg.count_limits(cfg, 6) == (1500, 200) and pg.count_limits(cfg, 3) == (1000, 150)
    assert pg.learner_seats_for(cfg, 3) == 1 and pg.learner_seats_for(cfg, 6) == 3


def test_check_players_mixed(tmp_path):
    def starts(n):
        path = tmp_path / f"s{n}.jsonl"
        path.write_text(json.dumps({"id": "g", "num_players": n, "actions": []}) + "\n")

    for n in (3, 5):
        starts(n)
    template = str(tmp_path / "s{n}.jsonl")
    cfg = pg.PGConfig(
        policy_checkpoint="p", value_checkpoint="v", player_mix={3: 1, 5: 1}, start_positions=template,
        population="pop.json",
    )
    pg.check_players(cfg)
    cfg.start_positions = str(tmp_path / "s3.jsonl")  # one file for every count
    with pytest.raises(SystemExit, match="template"):
        pg.check_players(cfg)
    cfg.start_positions = template
    cfg.learner_seat_weights = {1: 1.0}
    with pytest.raises(SystemExit, match="learner-seat-weights"):
        pg.check_players(cfg)
    cfg.learner_seat_weights = None
    (tmp_path / "s5.jsonl").write_text(json.dumps({"id": "g", "num_players": 4, "actions": []}) + "\n")
    with pytest.raises(SystemExit, match="no 5-player games"):
        pg.check_players(cfg)
    cfg.player_mix = {3: 1, 7: 1}
    with pytest.raises(SystemExit, match="2-6"):
        pg.check_players(cfg)


def test_points_and_seats_for_every_count():
    for n in range(2, 7):
        points = pg.default_placement_points(n)
        assert len(points) == n and points[0] == 1.0 and points[-1] == 0.0
        assert all(a > b for a, b in zip(points, points[1:]))  # a place higher is always worth more
        weights = pg.default_learner_seat_weights(n)
        assert all(1 <= k <= n - 1 for k in weights) and sum(weights.values()) == pytest.approx(1.0)
    assert pg.default_placement_points(4) == (1.0, 0.5, 0.2, 0.0)  # unchanged
    assert pg.default_learner_seat_weights(4) == pg.DEFAULT_LEARNER_SEAT_WEIGHTS
    assert pg.placement_points([3, 2, 1]).tolist() == [1.0, 0.4, 0.0]
    assert pg.placement_points([5, 5, 1]).tolist() == [0.7, 0.7, 0.0]  # a tie shares 1st and 2nd


def _rows(monkeypatch, start_file, counts=(3, 5)):
    monkeypatch.setattr(pg, "_CLIENTS", {"learner": _Client(), "sl": _Client(), "pool": _Client()})
    rows = []
    for n in counts:
        settings = _settings(start_file, num_players=n, learner_seats=pg.default_learner_seats(n), max_decisions=12)
        for game in pg.play_pg_games(1, settings)["games"]:
            assert game["num_players"] == n and len(game["win_share"]) == n
            rows.extend(game["rows"][:3])
    return rows


def test_mixed_rows_batch_like_one_count_batches(monkeypatch, start_file):
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel, _layout_for

    rows = _rows(monkeypatch, start_file)
    counts = [int(r[0][7]) for r in rows]
    assert sorted(set(counts)) == [3, 5]
    batch = pg.batch_tensors(rows, torch.device("cpu"))
    assert batch["game_state"].shape == (len(rows), _layout_for(6)[2])
    assert batch["num_players"].tolist() == counts
    torch.manual_seed(0)
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu"))).eval()
    with torch.no_grad():
        mixed = model(batch["game_state"], batch["nodes"], num_players=batch["num_players"])
        for n in (3, 5):
            idx = [i for i, c in enumerate(counts) if c == n]
            alone = pg.batch_tensors([rows[i] for i in idx], torch.device("cpu"))
            assert alone["num_players"] is None and alone["game_state"].shape[1] == _layout_for(n)[2]
            single = model(alone["game_state"], alone["nodes"])
            for a, b in zip(mixed[:3], single[:3]):  # policy logits, win logits, score
                assert torch.allclose(a[idx], b, atol=1e-4), float((a[idx] - b).abs().max())


def test_update_on_mixed_rows(monkeypatch, start_file):
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    rows = _rows(monkeypatch, start_file, counts=(2, 6))
    torch.manual_seed(0)
    config = ModelTransformerConfig(device=torch.device("cpu"))
    learner, critic, anchor = (AlphaZeroTransformerModel(config) for _ in range(3))
    anchor.load_state_dict(learner.state_dict())
    cfg = pg.PGConfig(policy_checkpoint="", value_checkpoint="", minibatch=4, microbatch=4, lr=1e-3, critic_lr=1e-3)
    rows, more = pg.subtract_baselines(critic, rows, cfg, torch.device("cpu"))
    assert math.isfinite(more["critic_row_brier"])
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0) for m in (learner, critic)]
    stats = pg._update(learner, critic, anchor, *opts, rows, cfg, 0.02, torch.device("cpu"), np.random.default_rng(0))
    assert all(math.isfinite(v) for v in stats.values()) and stats["rows"] == len(rows)


def test_row_players_from_width():
    from rl18xx.agent.alphazero.model_transformer import _layout_for

    for n in range(2, 7):
        state = np.zeros(_layout_for(n)[2], dtype=np.float32)
        assert pg._row_players(((state, None, None, None, 0, 0, 0),)) == n


# ------------------------------------------------------------------ PFSP per player count (2026-11-12)
def test_count_league_rates_are_per_player_count():
    league = pg.CountLeague([2, 4], ["a", "b", "c"], power=2.0, prior=4.0)
    for _ in range(100):
        league.record(2, "a", 0.0)  # a beats the learner at 2 players
        league.record(4, "a", 1.0)  # and loses to it at 4
    league.record(3, "a", 0.0)  # a count the run doesn't play: ignored
    assert league.win_rate(2, "a") < 0.05 and league.win_rate(4, "a") > 0.95
    assert league.win_rate(2, "b") == league.win_rate(4, "b") == 0.5
    w2 = dict(zip(league.members, league.league(2).weights()))
    w4 = dict(zip(league.members, league.league(4).weights()))
    assert w2["a"] > w2["b"] and w4["a"] < w4["b"]
    rng = np.random.default_rng(0)
    draws = [league.sample(2, rng) for _ in range(600)]
    assert draws.count("a") > 300
    league.add("d")
    assert all(league.win_rate(n, "d") == 0.5 for n in (2, 4))
    league.remove("b")
    assert league.members == ["a", "c", "d"] and all("b" not in league.league(n).members for n in (2, 4))
    league.min_shares = {"c": 0.2}
    assert league.league(2).min_shares == league.league(4).min_shares == {"c": 0.2}
    restored = pg.CountLeague.from_state(json.loads(json.dumps(league.state())), [2, 4], power=2.0)
    for n in (2, 4):
        assert restored.league(n).weights() == pytest.approx(league.league(n).weights())
    widened = pg.CountLeague.from_state(league.state(), [2, 4, 6], power=2.0)  # a count the state lacks
    assert widened.members == league.members and widened.win_rate(6, "a") == 0.5


def test_a_pooled_league_state_seeds_each_count_of_a_mix_lightly():
    pooled = pg.League(["a", "b"], power=2.0, prior=4.0, min_shares={"b": 0.1})
    for _ in range(300):
        pooled.record("a", 1.0)
    one = pg.CountLeague.from_state(pooled.state(), [4], power=2.0)
    assert one.league(4).stats == pooled.stats  # one count: the run's own league, as it was
    mixed = pg.CountLeague.from_state(pooled.state(), [2, 4], power=2.0, prior=4.0)
    for n in (2, 4):
        assert mixed.win_rate(n, "a") == pytest.approx(pooled.win_rate("a"))
        assert mixed.league(n).stats["a"][1] == pytest.approx(4.0)  # worth `prior` games, not ~200
        assert mixed.league(n).min_shares == {"b": 0.1}
    for _ in range(12):
        mixed.record(2, "a", 0.0)  # a few 2-player games outweigh the pooled seed
    assert mixed.win_rate(2, "a") < 0.3 and mixed.win_rate(4, "a") > 0.95


def test_league_slots_per_player_count():
    cfg = pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", player_mix={2: 1, 4: 3}, opponent_slots=2)
    assert pg.league_slots(cfg) == {2: ["opp2p0", "opp2p1"], 4: ["opp4p0", "opp4p1"]}
    one = pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", opponent_slots=3)
    assert pg.league_slots(one) == {4: ["opp0", "opp1", "opp2"]}


def test_league_for_a_mixed_run(tmp_path):
    pop = tmp_path / "pop.json"
    pop.write_text(json.dumps({"members": ["a", {"path": "b", "min_share": 0.1}]}))
    cfg = pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", population=str(pop), player_mix={2: 1, 5: 1})
    league = pg.league_for_run(cfg, None)
    assert league.counts == [2, 5] and league.members == ["a", "b"] and league.min_shares == {"b": 0.1}
    league.record(5, "a", 1.0)
    resumed = pg.league_for_run(cfg, league.state())
    assert resumed.win_rate(5, "a") == pytest.approx(league.win_rate(5, "a")) and resumed.win_rate(2, "a") == 0.5


def test_a_pin_limited_to_some_player_counts(tmp_path):
    pop = tmp_path / "pop.json"
    pop.write_text(
        json.dumps(
            {
                "members": [
                    "a",
                    {"path": "b", "min_share": 0.1, "min_share_players": [4]},
                    {"path": "c", "min_share": 0.05},
                ]
            }
        )
    )
    assert pg.load_min_shares(pop) == {"b": 0.1, "c": 0.05}
    assert pg.load_min_shares(pop, 4) == {"b": 0.1, "c": 0.05}
    assert pg.load_min_shares(pop, 2) == {"c": 0.05}
    cfg = pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", population=str(pop), player_mix={2: 1, 4: 1})
    league = pg.league_for_run(cfg, None)
    assert league.league(4).min_shares == {"b": 0.1, "c": 0.05} and league.league(2).min_shares == {"c": 0.05}
    league.record(2, "a", 1.0)
    # A resume takes the file's pins per count (a state saved with every pin at every count included).
    state = league.state()
    state["by_count"]["2"]["min_shares"] = {"b": 0.1, "c": 0.05}
    resumed = pg.league_for_run(cfg, state)
    assert resumed.league(2).min_shares == {"c": 0.05} and resumed.league(4).min_shares == {"b": 0.1, "c": 0.05}
    assert resumed.win_rate(2, "a") == pytest.approx(league.win_rate(2, "a"))
    w2 = dict(zip(resumed.league(2).members, resumed.league(2).weights()))
    w4 = dict(zip(resumed.league(4).members, resumed.league(4).weights()))
    assert w4["b"] >= 0.1 and w2["b"] < w4["b"]
    two_only = pg.PGConfig(policy_checkpoint="p", value_checkpoint="v", population=str(pop), num_players=2)
    assert pg.league_for_run(two_only, None).min_shares == {"c": 0.05}
