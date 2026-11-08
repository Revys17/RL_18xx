"""A run from nothing (experiments az0_2p_20261108): a randomly initialized network
(scripts/make_fresh_network.py) a one-network search-learning run can start from, games that
all start at the waterfall auction under the real rules with no start file read, stalled
auctions and capped games scored on net worth, the real rule saved with each game, and the
evals' auction starts and uniform-random player."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_data as sd  # noqa: E402
from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero.lockstep import ModelEvaluator  # noqa: E402

from .search_test_utils import HashNet  # noqa: E402
from .test_search_data import _config, _save, assert_same_row  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _script(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_from_scratch_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_fresh_network_is_a_run_start(tmp_path):
    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value
    from rl18xx.agent.alphazero.start_positions import new_game

    fresh = _script("make_fresh_network")
    path = fresh.write_fresh(tmp_path / "a", seed=3)
    assert path == tmp_path / "a" / "learner" / "0.pth"
    meta = json.loads((tmp_path / "a" / "config.json").read_text())
    assert meta["learner_temperature"] == 1.0 and not meta["value_entity_encoder"]
    assert 7_500_000 < meta["parameters"] < 8_300_000  # the supervised checkpoints' architecture, plain trunk
    assert sl.run_temperature(str(path)) == 1.0
    with pytest.raises(SystemExit):
        fresh.write_fresh(tmp_path / "a", seed=3)  # never overwrites
    same = fresh.write_fresh(tmp_path / "b", seed=3)
    other = fresh.write_fresh(tmp_path / "c", seed=4)
    a, b, c = (load_policy_value(str(p)) for p in (path, same, other))
    pa, pb, pc = (dict(m.named_parameters()) for m in (a, b, c))
    assert all(torch.equal(pa[k], pb[k]) for k in pa)  # the seed decides the weights
    assert any(not torch.equal(pa[k], pc[k]) for k in pa)
    # Its own value heads read a 2-player position: win probabilities over the 2 seats.
    state, nodes = new_game(2)._game.encode_canonical_py()[:2]
    gs = torch.as_tensor(np.asarray(state, dtype=np.float32))[None]
    nf = torch.as_tensor(np.asarray(nodes, dtype=np.float32))[None]
    a.eval()
    with torch.no_grad():
        _, win, _, _ = a(gs, nf, value_only=True)
    p = torch.softmax(win.float(), dim=1)[0]
    assert float(p[:2].sum()) == pytest.approx(1.0, abs=1e-4)


def _real_rules_games(tmp_path, **overrides):
    missing = str(tmp_path / "no_such_start_file.jsonl")  # never read: every game starts at the auction
    settings = dict(
        num_players=2, auction_fraction=1.0, auction_unlock=False, midgame_fraction=0.0, target="visits",
        visit_target="raw", dirichlet_weight=0.25, full_move_temperature=1.0, explore_decisions=10, n_fast=2,
        critic_positions_per_game=0,
    )  # fmt: skip
    cfg = _config(missing, **{**settings, **overrides})
    settings = sl.worker_settings(cfg, version=3)
    assert settings["auction_unlock"] is cfg.auction_unlock
    gen = sl.Generator(settings, {"learner": ModelEvaluator(HashNet())})
    games = gen.run(6.0)["games"]
    assert games
    return cfg, games


@pytest.mark.parametrize(
    "overrides, cap",
    [(dict(auction_stall_decisions=4, max_decisions=200), "auction_stall"), (dict(max_decisions=12), "max_length")],
    ids=["stalled_auction", "decision_cap"],
)
def test_capped_games_from_the_auction_are_scored_on_net_worth(tmp_path, overrides, cap):
    cfg, games = _real_rules_games(tmp_path, **overrides)
    assert {g["start_kind"] for g in games} == {"auction"}
    assert cap in {g["termination"] for g in games}
    for g in games:
        # Placement on net worth: the leader wins (a tie splits), net-worth shares sum to 1.
        win = np.asarray(g["win_share"])
        fractions = np.asarray(g["fractions"])
        assert sorted(win.tolist()) in ([0.0, 1.0], [0.5, 0.5]) and float(fractions.sum()) == pytest.approx(1.0)
        assert win[int(np.argmax(fractions))] > 0
        for row in g["rows"]:
            assert row.critic  # every searched row trains the value (critic_positions_per_game 0)
            assert np.array_equal(row.fractions, g["fractions"])
    saved = _save(games, tmp_path / "data", cfg)
    assert {s.meta["auction_unlock"] for s in saved} == {False}  # the real rule, kept with the game
    assert {s.meta["termination"] for s in saved} == {g["termination"] for g in games}
    for g, s in zip(games, saved):
        rebuilt = sd.rebuild_rows(s)  # replays under the game's own rule
        assert len(rebuilt) == len(g["rows"])
        for a, b in zip(g["rows"], rebuilt):
            assert_same_row(a, b)


def test_auction_starts_play_the_variant_unless_told_otherwise(tmp_path):
    cfg, games = _real_rules_games(tmp_path, max_decisions=8, auction_unlock=True)
    saved = _save(games, tmp_path / "data", cfg)
    assert {s.meta["auction_unlock"] for s in saved} == {True}


def test_auction_stall_setting_is_checked(tmp_path):
    cfg = _config(str(tmp_path / "x.jsonl"), auction_stall_decisions=-1)
    with pytest.raises(SystemExit, match="auction-stall"):
        sl._validate(cfg)


def test_policy_only_eval_plays_random_players_from_the_auction():
    epo = _script("eval_policy_only")
    assert epo.parse_player("random") == ("random", 1.0, ())
    settings = {
        "servers": {"A": "random", "B": "random"},
        "temperatures": {"A": 1.0, "B": 1.0},
        "rules": {"A": (), "B": ()},
        "seed": 0,
        "start_positions": "/nonexistent",  # not read
        "random_start_fraction": 1.0,
        "max_decisions": 60,
        "auction_start": True,
        "auction_stall_decisions": 25,
        "price_eps": 0.0,
        "arrangements": epo.arrangements(1, 2),
    }
    records = epo.play_games(list(range(6)), settings)
    assert len(records) == 6
    for r in records:
        assert r["termination"] in ("finished", "max_length", "auction_stall")
        assert sorted(r["win_share"]) in ([0.0, 1.0], [0.5, 0.5])
        assert 0 < r["decisions"] <= 60


def test_searched_eval_can_start_games_under_the_real_rules():
    h2h = _script("eval_head_to_head")
    settings = {"num_players": 2, "seed": 0}
    game, label, players = h2h._setup_game(0, (), dict(settings, auction_unlock=False))
    assert label is None and players == {} and game._game.auction_unlock is False
    game, _, _ = h2h._setup_game(0, (), settings)
    assert game._game.auction_unlock is True  # the default stays the self-play variant


def test_a_tiny_one_network_two_player_run_from_the_auction(tmp_path, monkeypatch):
    """The az0 run's shape on CPU with a tiny network: one network (--shared, the search on its own win head),
    2 players, every game from the auction under the real rules with no start file, the stall and decision
    caps, raw visit targets, root noise, the value on every searched row; it trains and saves learner
    checkpoints only."""
    import math

    from rl18xx.agent.alphazero.policy_gradient import _load, _save

    from .test_search_learning import _tiny_model

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    source = tmp_path / "fresh"
    (source / "learner").mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"learner_temperature": 1.0}))
    _save(_tiny_model(0), source / "learner" / "0.pth")
    cfg = sl.SearchLearningConfig(
        policy_checkpoint=str(source / "learner" / "0.pth"), value_checkpoint="", out_dir=str(tmp_path / "runs"),
        run_name="az0", shared=True, value_head="win", num_players=2, target="visits", visit_target="raw",
        dirichlet_weight=0.25, full_move_temperature=1.0, explore_decisions=10, prior_temperature=1.0,
        c_puct_init=0.3125, reuse_tree="additional", n_full=6, n_fast=3, p_full=0.5, parallel_readouts=2,
        workers=2, games_per_worker=2, task_seconds=2.0, max_decisions=40, auction_stall_decisions=20,
        start_positions=str(tmp_path / "none.jsonl"), auction_fraction=1.0, auction_unlock=False,
        inference_batch_size=64, window_rows=400, rows_per_update=10, min_window_rows=10, steps_per_update=2,
        minibatch=8, microbatch=4, max_updates=2, snapshot_every=1, kl_coef=0.0, kl_target=None, anchor_every=0,
        critic_positions_per_game=0, score_loss_weight=0.0, value_loss_weight=1.0, lr=3e-4, critic_lr=3e-4,
    )  # fmt: skip
    run_dir = sl.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    for h in history:
        assert math.isfinite(h["ce"]) and math.isfinite(h["critic_outcome_ce"]) and h["critic_rows_trained"] > 0
        assert set(h["starts"]) == {"auction"} and 0.5 < h["noised_share"] < 2.0
    assert not (run_dir / "critic").exists()  # one network
    first, last = _load(run_dir / "learner" / "0.pth"), _load(run_dir / "learner" / "2.pth")
    heads = [n for n, _ in first.named_parameters() if n.startswith("win_loss_head")]
    a, b = dict(first.named_parameters()), dict(last.named_parameters())
    assert heads and any(not torch_equal(a[n], b[n]) for n in heads)  # the value heads trained


def torch_equal(x, y) -> bool:
    import torch

    return bool(torch.equal(x.detach(), y.detach()))
