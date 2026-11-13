"""Policy gradient for the title-generic model (RL bridge milestone C2): the sparse-legal inference server
(``generic_inference``), the title play / start module (``generic_play``) and the trainer (``generic_pg``).

Positions are random walks of 1830 at 2-6 players (the layout's legal rows); games are played in-process on
small fresh networks (``LocalGenericClient`` stands in for a server process), one test spawns a real server
process and one runs ``generic_pg.run`` end to end on the CPU (two updates and a resume)."""

from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_inference as gi  # noqa: E402
from rl18xx.agent.alphazero import generic_pg as gpg  # noqa: E402
from rl18xx.agent.alphazero import generic_play as gp  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero import policy_gradient as pg  # noqa: E402
from rl18xx.agent.alphazero.generic_tokens import encode_batch, raw_game  # noqa: E402
from rl18xx.agent.alphazero.titles import new_game  # noqa: E402

SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip


def walk_1830(num_players: int, seed: int, steps: int = 400, every: int = 7) -> list:
    rng = random.Random(seed)
    game = raw_game(new_game("1830", num_players, auction_unlock=True))
    positions = []
    for step in range(steps):
        if game.finished:
            break
        legal = [int(i) for i in game.factored_legal_indices()]
        if step % every == 0 and len(legal) > 1:
            positions.append(game.pickle_clone())
        index = rng.choice(legal)
        price_range = game.price_range_for_index(index)
        game.apply_action_index(index, rng.randint(*price_range) if price_range else None)
    return positions


@pytest.fixture(scope="module")
def positions():
    out = []
    for n, seed in ((2, 1), (3, 2), (4, 3), (5, 4), (6, 5)):
        out += walk_1830(n, seed)
    return out


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = mg.GenericPolicyValueModel(mg.GenericModelConfig(**SMALL)).eval()
    with torch.no_grad():  # non-trivial heads (zero-initialized outputs give uniform price cells / equal odds)
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
    return m


@pytest.fixture
def checkpoint(model, tmp_path):
    path = tmp_path / "start.pth"
    mg.save_checkpoint(model, path, titles=["1830"])
    return str(path)


def _inputs(games):
    legal = [mg.legal_actions(g) for g in games]
    return mg.build_inputs(games, legal), legal


# ---------------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------------


def test_slice_positions_then_legal_lists_equal_build_inputs(positions):
    """A worker encodes every pending position once and splits them by server: the split arrays are
    ``build_inputs`` of the subset, bit for bit."""
    games = positions[:40]
    enc = encode_batch(games)
    subset = [3, 7, 8, 21, 30, 39]
    legal = [mg.legal_actions(games[i]) for i in subset]
    split = mg.add_legal_inputs(gi.slice_positions(enc, subset), legal)
    direct = mg.build_inputs([games[i] for i in subset], legal)
    assert set(split) == set(direct)
    for key, value in direct.items():
        if value is None:
            assert split[key] is None
        else:
            np.testing.assert_array_equal(split[key], value, err_msg=key)


def test_merged_requests_equal_separate_forwards(positions, model):
    """Requests from several workers (different player counts, legal-list lengths, relation counts) merge
    into one forward that gives each request what its own forward gives."""
    server = gi.LocalGenericServer(model, seed=0)
    chunks = [positions[0:9], positions[30:37], positions[60:75], positions[-6:]]
    requests = [(_inputs(c)[0], 0.0) for c in chunks]
    merged = server.run_many(requests, want_price_logp=True)
    for (arrays, _), reply in zip(requests, merged):
        alone = server.run(arrays, want_price_logp=True)
        np.testing.assert_allclose(reply.logp, alone.logp, atol=2e-5)
        np.testing.assert_allclose(reply.win, alone.win, atol=2e-5)
        np.testing.assert_allclose(reply.score, alone.score, atol=2e-5)
        np.testing.assert_allclose(reply.price_logp, alone.price_logp, atol=2e-5)
        assert reply.logp.shape == (len(arrays["legal_rows"]),)
        assert reply.win.shape == (len(arrays["meta"]), 6)
        # per-position distributions; win shares over the present seats
        offsets = arrays["legal_offsets"]
        for b in range(len(arrays["meta"])):
            assert math.isclose(np.exp(reply.logp[offsets[b] : offsets[b + 1]]).sum(), 1.0, rel_tol=1e-4)
        n = arrays["meta"][:, 4]
        np.testing.assert_allclose(reply.win.sum(1), 1.0, atol=1e-5)
        assert all(reply.win[b, n[b] :].max(initial=0) < 1e-6 for b in range(len(n)))
    # uncompressed requests give the same replies
    plain = server.run_many(requests, want_price_logp=True, compress=False)
    for a, b in zip(plain, merged):
        np.testing.assert_array_equal(a.logp, b.logp)


def test_sampled_price_cells_follow_the_mixed_head(positions, model):
    """Every priced legal row gets a cell from a non-empty cell of its range, drawn with the eps-mixed
    distribution whose log-probability comes back."""
    server = gi.LocalGenericServer(model, seed=1)
    priced = [g for g in positions if (mg.legal_actions(g).rows[:, mg.C_PRICE_KIND] >= 0).any()]
    assert len(priced) >= 5
    arrays, _ = _inputs(priced[:20])
    for eps in (0.0, 0.3):
        reply = server.run(arrays, price_eps=eps, want_price_logp=True)
        mask = arrays["price_cell_mask"]
        assert len(reply.price_cell) == len(arrays["price_rows"]) > 0
        for j, cell in enumerate(reply.price_cell):
            assert mask[j, cell]
            probs = np.where(mask[j], np.exp(reply.price_logp[j].astype(np.float64)), 0.0)
            probs /= probs.sum()
            mixed = np.where(mask[j], (1 - eps) * probs + eps / mask[j].sum(), 0.0)
            mixed /= mixed.sum()
            assert math.isclose(reply.price_cell_logp[j], math.log(mixed[cell]), abs_tol=1e-4)
    # the empirical distribution of one row's draws
    logp = torch.log_softmax(torch.tensor([[0.0, 1.0, 2.0, -50.0]]).repeat(20000, 1), dim=-1)
    mask = torch.tensor([[True, True, True, False]]).repeat(20000, 1)
    cells, cell_logp = gi.sample_price_cells(logp, mask, torch.full((20000,), 0.2), torch.Generator().manual_seed(0))
    p = np.exp(logp[0, :3].numpy()) / np.exp(logp[0, :3].numpy()).sum()
    expected = 0.8 * p + 0.2 / 3
    counts = np.bincount(cells.numpy(), minlength=4) / 20000
    np.testing.assert_allclose(counts[:3], expected, atol=0.015)
    assert counts[3] == 0
    np.testing.assert_allclose(cell_logp.exp().numpy(), expected[cells.numpy()], atol=1e-5)


def test_spawned_server_round_trip(positions, checkpoint):
    """A spawned generic server answers a worker client, reloads and shuts down."""
    handle = gi.start_generic_inference_server(
        num_workers=1, checkpoint_path=checkpoint, model_factory=gi.load_generic_server_model_cpu, batch_size=64,
        threads=1, seed=0,
    )
    try:
        deadline = time.monotonic() + 120
        while handle.health(timeout_s=60).state != "ACCEPTING":
            assert time.monotonic() < deadline, "spawned server never reached ACCEPTING"
            time.sleep(0.1)
        client = gi.GenericInferenceClient(request_q=handle.request_q, reply_q=handle.reply_qs[0], worker_id=0)
        arrays, _ = _inputs(positions[:12])
        reply = client.receive_outputs(client.send_inputs(arrays))
        model, _ = mg.load_checkpoint(checkpoint)
        local = gi.LocalGenericServer(model.eval()).run(arrays)
        np.testing.assert_allclose(reply.logp, local.logp, atol=1e-5)
        np.testing.assert_allclose(reply.win, local.win, atol=1e-5)
        assert handle.reload(checkpoint, timeout_s=60) == {"ok": True}
        reply = client.receive_outputs(client.send_inputs(arrays, compress=False))
        np.testing.assert_allclose(reply.logp, local.logp, atol=1e-5)
    finally:
        handle.shutdown(timeout_s=30)
    assert not handle.process.is_alive()


# ---------------------------------------------------------------------------------------------------
# Games and updates
# ---------------------------------------------------------------------------------------------------


@pytest.fixture
def start_file(tmp_path):
    """An empty start file: every game takes a random auction ending."""
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


def _settings(start_file, **overrides):
    settings = {
        "title": "1830",
        "num_players": 4,
        "positions_per_seat": 6,
        "learner_seats": 2,
        "sl_opponent_fraction": 0.5,
        "pool_ready": False,
        "pool_label": None,
        "learner_temperature": 0.5,
        "opponent_temperature": 1.0,
        "opponent_price_eps": 0.05,
        "max_decisions": 60,
        "start_positions": start_file,
        "random_start_fraction": 1.0,
        "reward": "placement",
        "league": None,
    }
    settings.update(overrides)
    return settings


@pytest.fixture
def clients(model, monkeypatch):
    server = gi.LocalGenericServer(model, seed=3)
    local = {name: gi.LocalGenericClient(server) for name in ("learner", "sl", "pool", "opp0", "opp1")}
    monkeypatch.setattr(gpg, "_CLIENTS", local)
    return local


def test_games_keep_only_learner_decisions_with_their_sampling_probabilities(start_file, clients, model):
    result = gpg.play_generic_pg_games(3, _settings(start_file))
    assert len(result["games"]) == 3
    rows = [r for g in result["games"] for r in g["rows"]]
    assert rows
    for g in result["games"]:
        assert g["termination"] in ("finished", "max_length")
        assert len(g["learner_seats"]) == 2 and g["opponent"] == "sl"
        assert math.isclose(sum(g["win_share"]), 1.0, rel_tol=1e-5)
        assert sorted(g["reward"], reverse=True)[0] == 1.0  # placement points: 1st gets 1
        for r in g["rows"]:
            assert r.seat in g["learner_seats"] and r.num_players == 4
            assert r.advantage == pytest.approx(g["reward"][r.seat])
            assert r.win[0] == pytest.approx(g["win_share"][r.seat])  # seat frame: the mover first
        assert sum(1 for r in g["rows"] if r.seat == g["learner_seats"][0]) <= 6
    # the trainer recomputes each kept move's probability exactly (same network: ratio 1)
    batch = gpg.batch_to_device(gpg.collate_rows(rows), "cpu")
    with torch.no_grad():
        out = model(batch["inputs"])
        losses = gpg.generic_pg_losses(out, out, batch, clip=0.2, kl_coef=0.02, entropy_coef=0.0, temperature=0.5)
    assert float(losses["log_ratio_abs_mean"]) < 1e-4
    assert float(losses["kl_anchor"]) == pytest.approx(0.0, abs=1e-6)
    assert float(losses["clip_frac"]) == 0.0
    assert len(batch["price_sel"]) == sum(1 for r in rows if r.price_cell >= 0)


def test_league_games_record_pairwise_results(start_file, clients):
    league = {
        "slots": ["opp0", "opp1"],
        "slot_labels": {"opp0": "member_a", "opp1": "member_b"},
        "slot_versions": {},
        "learner_seat_weights": {1: 0.6, 2: 0.4},
    }
    result = gpg.play_generic_pg_games(2, _settings(start_file, num_players=3, league=league, max_decisions=40))
    for g in result["games"]:
        assert g["opponent"] == "league" and g["num_players"] == 3
        assert set(g["seat_members"].values()) <= {"member_a", "member_b"}
        assert len(g["pairwise"]) == 3 - len(g["learner_seats"])
        assert all(0.0 <= share <= 1.0 for _, share in g["pairwise"])


def test_update_trains_learner_and_critic(start_file, clients, model):
    rows = []
    while len(rows) < 48:
        result = gpg.play_generic_pg_games(2, _settings(start_file, num_players=random.choice((2, 4, 5))))
        rows += [r for g in result["games"] for r in g["rows"]]
    cfg = gpg.GenericPGConfig(policy_checkpoint="unused", minibatch=16, microbatch=8, lr=1e-3, critic_lr=1e-3,
                              reward="placement", learner_temperature=0.5)
    learner = mg.GenericPolicyValueModel(model.config)
    learner.load_state_dict(model.state_dict())
    critic = mg.GenericPolicyValueModel(model.config)
    critic.load_state_dict(model.state_dict())
    anchor = model
    before = {k: v.clone() for k, v in learner.state_dict().items()}
    stats = gpg.subtract_baselines(critic, rows, cfg, "cpu")
    assert np.isfinite(stats["critic_row_brier"])
    lo = torch.optim.AdamW(learner.parameters(), lr=cfg.lr, weight_decay=0.0)
    co = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
    first = gpg._update(learner, critic, anchor, lo, co, rows, cfg, 0.02, "cpu", np.random.default_rng(0))
    for key in ("pg_loss", "kl_anchor", "entropy", "critic_ce", "critic_mse", "grad_norm", "approx_kl_first"):
        assert np.isfinite(first[key]), key
    assert first["approx_kl_first"] == pytest.approx(0.0, abs=1e-4)  # first minibatch: the sampling policy
    assert first["optimizer_steps"] == math.ceil(len(rows) / 16)
    assert any(not torch.equal(before[k], v) for k, v in learner.state_dict().items())
    for _ in range(3):
        later = gpg._update(learner, critic, anchor, lo, co, rows, cfg, 0.02, "cpu", np.random.default_rng(1))
    assert later["kl_anchor"] > 0
    assert later["critic_ce"] < first["critic_ce"]


def test_rewards_and_outcomes():
    worth = np.array([500.0, 900.0, 900.0, 100.0])
    win = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    reward, score = gpg.seat_rewards("placement", worth, win)
    np.testing.assert_allclose(reward, [0.2, 0.75, 0.75, 0.0])
    np.testing.assert_allclose(score, reward)
    reward, score = gpg.seat_rewards("win", worth, win)
    np.testing.assert_allclose(reward, win)
    np.testing.assert_allclose(score, [0.2, 0.75, 0.75, 0.0])
    with pytest.raises(ValueError):
        gpg.parse_reward("president:PRR")
    np.testing.assert_allclose(np.exp(gpg.tempered_logp(np.log([0.7, 0.2, 0.1]), 0.5)), np.array([49, 4, 1]) / 54)


# ---------------------------------------------------------------------------------------------------
# Starts
# ---------------------------------------------------------------------------------------------------


def test_1830_starts_are_the_legacy_ones(start_file):
    from rl18xx.agent.alphazero import start_positions as sp

    start, raw, skipped = gp.replayed_start("1830", 4, start_file, 1.0, rng=random.Random(5))
    legacy_start, legacy_game, _ = sp.replayed_start(4, start_file, 1.0, rng=random.Random(5))
    assert start.label == legacy_start.label == "random" and skipped == 0
    assert list(start.actions) == list(legacy_start.actions)
    assert raw.round.round_type == "Stock" and raw.no_buyback_rule
    auction, raw, _ = gp.replayed_start("1830", 3, start_file, 1.0, rng=random.Random(0), auction_fraction=1.0,
                                        auction_unlock=False)
    assert auction.label == "auction" and raw.round.round_type == "Auction" and not raw.auction_unlock


def test_random_openings_reach_the_first_stock_round():
    for title, n in (("1867", 3), ("1867", 6)):
        actions = gp.random_opening_actions(title, n, random.Random(n), seed=7)
        raw = gp.apply_actions(gp.new_title_game(title, n, seed=7), actions)
        assert raw.round.round_type == "Stock" and not raw.finished
        assert actions
    assert gp.random_opening_actions("1822", 4, random.Random(0), seed=1) == []  # opens with a Stock Round


def test_start_records_from_a_replayed_game(tmp_path):
    """Cut a played 1830 game at its Stock Rounds and sample starts back from the files."""
    rng = random.Random(3)
    game = new_game("1830", 4)
    raw = raw_game(game)
    stock_rounds = 0
    previous = None
    while not raw.finished and stock_rounds < 4:
        legal = [int(i) for i in raw.factored_legal_indices()]
        index = rng.choice(legal)
        price_range = raw.price_range_for_index(index)
        raw.apply_action_index(index, price_range[0] if price_range else None)
        rt = raw.round.round_type
        stock_rounds += rt == "Stock" and previous != "Stock"
        previous = rt
    replayed = gp.ReplayedGame("g1", 4, None, [dict(a) for a in raw.raw_actions])
    start, midgame, reason = gp.start_records("1830", replayed, first_round=2, last_round=6)
    assert reason is None and start is not None and midgame is not None
    assert set(midgame["cuts"]) >= {"2", "3"}
    cuts = gp.stock_round_cuts("1830", 4, replayed.actions)
    assert len(start["actions"]) == cuts[1]
    sfile, mfile = tmp_path / "s.jsonl", tmp_path / "m.jsonl"
    sfile.write_text(json.dumps(start) + "\n")
    mfile.write_text(json.dumps(midgame) + "\n")
    s = gp.sample_start("1830", 4, str(sfile), 0.0, rng=random.Random(0))
    assert s.label == "human:g1" and len(s.actions) == cuts[1]
    m = gp.sample_start("1830", 4, str(sfile), 0.0, rng=random.Random(0), midgame_path=str(mfile), midgame_fraction=1.0)
    assert m.label.startswith("midgame:g1:sr")
    k = int(m.label.rsplit("sr", 1)[1])
    assert len(m.actions) == cuts[k]
    with pytest.raises(ValueError):
        gp.load_start_file(str(sfile), "1867")


# ---------------------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------------------


def test_run_two_updates_snapshot_and_resume(tmp_path, checkpoint, start_file):
    """``generic_pg.run`` on the CPU: spawned servers and workers, mixed player counts, a league of two
    generic checkpoints, two updates (snapshots every update), then a resume for one more."""
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    league = tmp_path / "league.json"
    league.write_text(json.dumps({"members": [checkpoint, {"path": checkpoint, "min_share": 0.1}]}))
    template = str(tmp_path / "starts_{n}p.jsonl")
    for n in (3, 4):
        (tmp_path / f"starts_{n}p.jsonl").write_text("")
    cfg = gpg.GenericPGConfig(
        policy_checkpoint=checkpoint, out_dir=str(tmp_path / "runs"), run_name="t", player_mix={3: 1, 4: 1},
        workers=2, games_per_task=2, tasks_in_flight_per_worker=1, positions_per_seat=4, max_decisions=30,
        start_positions=template, random_start_fraction=1.0, population=str(league), opponent_slots=1,
        rows_per_update=24, minibatch=16, microbatch=8, max_updates=2, snapshot_every=1, slot_refresh_every=1,
        league_snapshot_every=1, reward="placement", learner_temperature=0.5, opponent_temperature=0.5,
        server_cpu=True, server_threads=1, inference_batch_size=64, save_game_every=1, threads=2,
    )
    run_dir = gpg.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    for h in history:
        assert np.isfinite(h["pg_loss"]) and np.isfinite(h["critic_ce"]) and h["games"] > 0
    assert (run_dir / "learner" / "2.pth").exists() and (run_dir / "critic" / "2.pth").exists()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["snapshot_update"] == 2 and set(state["league"]["by_count"]) == {"3", "4"}
    model, meta = mg.load_checkpoint(run_dir / "learner" / "2.pth")
    assert meta["titles"] == ["1830"] and meta["extra"]["role"] == "learner"
    saved = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
    assert saved and json.loads((run_dir / saved[0]["game_file"]).read_text())["title"] == "1830"
    resumed = gpg.GenericPGConfig(policy_checkpoint=checkpoint, max_updates=3)
    gpg.run(resumed, resume=str(run_dir), explicit={"max_updates"})
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2, 3]
