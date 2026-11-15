"""The robustness mechanisms of the generic policy-gradient stage (``docs/robustness_test_plan.md`` arms A1-A3;
``docs/rl_model.md`` "Robustness mechanisms"): tempered league members and per-count pins, the reference KL,
prioritized starts, the main-exploiter mode and the population reload -- and that with every one of them off the
stage is C2's (losses, starts, league).

Games are played in-process on small fresh networks (``LocalGenericClient``); two tests run ``generic_pg.run``
end to end on the CPU (spawned servers and workers)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_inference as gi  # noqa: E402
from rl18xx.agent.alphazero import generic_pg as gpg  # noqa: E402
from rl18xx.agent.alphazero import generic_play as gp  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero import policy_gradient as pg  # noqa: E402
from rl18xx.agent.alphazero import start_positions as sp  # noqa: E402
from rl18xx.agent.alphazero.generic_tokens import raw_game  # noqa: E402
from rl18xx.agent.alphazero.titles import new_game  # noqa: E402

SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip


def _model(seed: int):
    torch.manual_seed(seed)
    m = mg.GenericPolicyValueModel(mg.GenericModelConfig(**SMALL)).eval()
    with torch.no_grad():  # non-trivial heads (zero-initialized outputs give uniform price cells / equal odds)
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
    return m


@pytest.fixture(scope="module")
def model():
    return _model(0)


@pytest.fixture
def checkpoint(model, tmp_path):
    path = tmp_path / "start.pth"
    mg.save_checkpoint(model, path, titles=["1830"])
    return str(path)


@pytest.fixture
def start_file(tmp_path):
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
        "max_decisions": 50,
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
    local = {name: gi.LocalGenericClient(server) for name in ("learner", "sl", "pool", "opp0", "opp1", "main")}
    monkeypatch.setattr(gpg, "_CLIENTS", local)
    monkeypatch.setattr(gpg, "_SLOT_TEMPERATURES", {})
    return local


def _rows(start_file, n_games=3, **overrides):
    result = gpg.play_generic_pg_games(n_games, _settings(start_file, **overrides))
    return [r for g in result["games"] for r in g["rows"]], result


# ---------------------------------------------------------------------------------------------------
# Losses: unchanged without a reference; the reference KL
# ---------------------------------------------------------------------------------------------------


def _c2_losses(out, anchor_out, batch, clip, kl_coef, entropy_coef, temperature):
    """``generic_pg_losses`` as milestone C2 wrote it (before the reference KL), for the no-change check."""
    from rl18xx.agent.alphazero.model_generic import segment_log_softmax

    inputs = batch["inputs"]
    pos, offsets = inputs["legal_pos"], inputs["legal_offsets"]
    b = offsets.shape[0] - 1
    log_p = segment_log_softmax(out["logits"].float() / temperature, pos, b)
    sl_log_p = segment_log_softmax(anchor_out["logits"].float() / temperature, pos, b)
    p = log_p.exp()
    kl = log_p.new_zeros(b).index_add(0, pos, p * (log_p - sl_log_p))
    entropy = log_p.new_zeros(b).index_add(0, pos, -(p * log_p))
    new_logp = log_p[offsets[:-1] + batch["choice"]]
    price_kl = log_p.new_zeros(())
    if batch["price_sel"].numel():
        sel, ppos = batch["price_sel"], batch["price_pos"]
        mask = inputs["price_cell_mask"][sel]
        cell_lp = out["price_logp"][sel].float().masked_fill(~mask, 0.0)
        sl_cell_lp = anchor_out["price_logp"][sel].float().masked_fill(~mask, 0.0)
        new_logp = new_logp.index_add(0, ppos, cell_lp.gather(1, batch["price_cell"][:, None]).squeeze(1))
        cell_kl = (cell_lp.exp() * mask * (cell_lp - sl_cell_lp)).sum(dim=1)
        kl = kl.index_add(0, ppos, cell_kl)
        price_kl = cell_kl.mean()
    advantage = batch["advantage"]
    log_ratio = new_logp - batch["old_logp"]
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip, 1.0 + clip) * advantage)
    pg_loss = -surrogate.mean()
    total = pg_loss + kl_coef * kl.mean() - entropy_coef * entropy.mean()
    return {"total": total, "kl_anchor": kl.mean(), "price_kl_anchor": price_kl, "entropy": entropy.mean()}


def test_losses_without_reference_are_c2s_and_the_reference_kl_adds_on(start_file, clients, model):
    rows, _ = _rows(start_file, auction_fraction=1.0)  # auction starts: bids with open prices
    assert any(r.price_cell >= 0 for r in rows)  # the price cells' KL is exercised too
    for r in rows:
        r.advantage -= 0.4
    batch = gpg.batch_to_device(gpg.collate_rows(rows), "cpu")
    other = _model(7)
    with torch.no_grad():
        out, anchor_out, ref_out = model(batch["inputs"]), other(batch["inputs"]), other(batch["inputs"])
    new = gpg.generic_pg_losses(out, anchor_out, batch, 0.2, 0.03, 0.01, temperature=0.5)
    old = _c2_losses(out, anchor_out, batch, 0.2, 0.03, 0.01, 0.5)
    for key in ("total", "kl_anchor", "price_kl_anchor", "entropy"):
        assert torch.equal(new[key], old[key]), key
    assert "kl_ref" not in new
    # the reference at the learner's temperature: the anchor's KL again, weighted by ref_kl_coef
    with_ref = gpg.generic_pg_losses(
        out, anchor_out, batch, 0.2, 0.03, 0.01, temperature=0.5, ref_out=ref_out, ref_kl_coef=0.5
    )
    assert float(with_ref["kl_ref"]) == pytest.approx(float(new["kl_anchor"]), rel=1e-6)
    assert float(with_ref["total"]) == pytest.approx(float(new["total"]) + 0.5 * float(new["kl_anchor"]), rel=1e-5)
    # at the reference's own temperature (piKL's form) the KL differs, and is 0 only against itself
    t1 = gpg.generic_pg_losses(out, out, batch, 0.2, 0.0, 0.0, temperature=0.5, ref_out=out, ref_kl_coef=1.0,
                               ref_temperature=1.0)  # fmt: skip
    t_same = gpg.generic_pg_losses(out, out, batch, 0.2, 0.0, 0.0, temperature=0.5, ref_out=out, ref_kl_coef=1.0)
    assert float(t_same["kl_ref"]) == pytest.approx(0.0, abs=1e-6)
    assert float(t1["kl_ref"]) > 1e-4


def test_update_with_a_reference_pulls_toward_it(start_file, clients, model):
    rows, _ = _rows(start_file, n_games=4)
    reference = _model(11)
    cfg = gpg.GenericPGConfig(policy_checkpoint="unused", minibatch=16, microbatch=8, lr=3e-3, critic_lr=1e-3,
                              reward="placement", learner_temperature=0.5, ref_kl_coef=5.0)  # fmt: skip
    for r in rows:
        r.advantage = 0.0  # no policy-gradient signal: only the KL terms move the learner
    learner = _model(0)
    critic = _model(0)
    lo = torch.optim.AdamW(learner.parameters(), lr=cfg.lr, weight_decay=0.0)
    co = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
    kls = []
    for i in range(4):
        stats = gpg._update(learner, critic, model, lo, co, rows, cfg, 0.0, "cpu", np.random.default_rng(i), reference)
        kls.append(stats["kl_ref"])
    assert all(np.isfinite(kls)) and kls[-1] < kls[0]


# ---------------------------------------------------------------------------------------------------
# A1: tempered members, per-count pins
# ---------------------------------------------------------------------------------------------------


def test_member_keys_and_population_fields(tmp_path):
    assert gpg.member_key("a/b.pth") == "a/b.pth" and gpg.member_key("a/b.pth", 1.0) == "a/b.pth@1"
    assert gpg.member_path("a/b.pth@0.5") == "a/b.pth" and gpg.member_path("a/b.pth") == "a/b.pth"
    assert gpg.member_path("x@y/b.pth") == "x@y/b.pth"  # an @ that isn't a temperature stays in the path
    assert gpg.member_temperature("a/b.pth@0.5", 1.0) == 0.5 and gpg.member_temperature("a/b.pth", 0.7) == 0.7
    pop = tmp_path / "pop.json"
    pop.write_text(json.dumps({"members": [
        "plain.pth",
        {"path": "sl.pth", "temperature": 1.0, "min_share": 0.125},
        {"path": "sl.pth", "temperature": 0.5, "min_share_by_players": {"2": 0.1, "4": 0.2}},
        {"path": "exp.pth", "min_share": 0.3, "min_share_players": [4]},
        {"path": "plain.pth"},
    ]}))  # fmt: skip
    assert gpg.load_members(pop) == ["plain.pth", "sl.pth@1", "sl.pth@0.5", "exp.pth"]
    assert gpg.load_member_min_shares(pop, 4) == {"sl.pth@1": 0.125, "sl.pth@0.5": 0.2, "exp.pth": 0.3}
    assert gpg.load_member_min_shares(pop, 2) == {"sl.pth@1": 0.125, "sl.pth@0.5": 0.1}
    assert gpg.load_member_min_shares(pop, 3) == {"sl.pth@1": 0.125}
    # without the new fields the pins are the legacy loader's
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"members": ["a.pth", {"path": "b.pth", "min_share": 0.2, "min_share_players": [4]}]}))
    for n in (None, 3, 4):
        assert gpg.load_member_min_shares(legacy, n) == pg.load_min_shares(legacy, n)
    assert gpg.load_members(legacy) == pg.load_population(legacy)


def test_tempered_slots_sample_at_their_members_temperature(start_file, clients, monkeypatch):
    seen = []
    original = gpg.tempered_logp

    def spy(logp, temperature):
        seen.append(float(temperature))
        return original(logp, temperature)

    monkeypatch.setattr(gpg, "tempered_logp", spy)
    league = {
        "slots": ["opp0", "opp1"],
        "slot_labels": {"opp0": "sl.pth@0.7", "opp1": "sl.pth@1.3"},
        "slot_versions": {},
        "slot_temperatures": {"opp0": 0.7, "opp1": 1.3},
        "learner_seat_weights": {1: 1.0},
    }
    result = gpg.play_generic_pg_games(3, _settings(start_file, league=league, max_decisions=60))
    used = {s for g in result["games"] for s in g["seat_members"].values()}
    expected = {0.5} | {0.7 for m in used if m.endswith("@0.7")} | {1.3 for m in used if m.endswith("@1.3")}
    assert set(seen) == expected and len(expected) >= 2
    # the shared value (the trainer's, updated as a slot changes members) wins over the task's settings
    monkeypatch.setattr(gpg, "_SLOT_TEMPERATURES", {"opp0": type("V", (), {"value": 2.0})()})
    assert gpg.seat_temperature("opp0", _settings(start_file, league=league)) == 2.0
    assert gpg.seat_temperature("opp1", _settings(start_file, league=league)) == 1.3
    assert gpg.seat_temperature("sl", _settings(start_file)) == 1.0
    assert gpg.seat_temperature("learner", _settings(start_file)) == 0.5


def _league_cfg(pop, **kw):
    return gpg.GenericPGConfig(policy_checkpoint="unused", population=str(pop), player_mix={2: 1, 4: 1}, **kw)


def test_league_reload_and_resume_semantics(tmp_path):
    run_dir = tmp_path / "run"
    own = str(run_dir / "learner" / "50.pth")
    pop = tmp_path / "pop.json"
    pop.write_text(json.dumps({"members": [
        {"path": "sl.pth", "temperature": 1.0, "min_share": 0.125},
        {"path": "sl.pth", "temperature": 0.5, "min_share": 0.125},
    ]}))  # fmt: skip
    league = gpg.league_for_run(_league_cfg(pop), None, run_dir)
    assert league.members == ["sl.pth@1", "sl.pth@0.5"]
    assert league.league(2).min_shares == league.league(4).min_shares == {"sl.pth@1": 0.125, "sl.pth@0.5": 0.125}
    league.add(own)
    known = set(gpg.load_members(pop))
    # the loop adds an exploiter (pinned at 4 players) and the file drops the T 0.5 member
    exp = tmp_path / "exp.pth"
    mg.save_checkpoint(_model(1), exp, titles=["1830"])
    pop.write_text(json.dumps({"members": [
        {"path": "sl.pth", "temperature": 1.0, "min_share": 0.125},
        {"path": str(exp), "min_share": 0.1, "min_share_players": [4]},
    ]}))  # fmt: skip
    changes = gpg.reload_population(league, _league_cfg(pop), run_dir, known)
    assert changes == {"added": [str(exp)], "removed": ["sl.pth@0.5"]}
    assert set(league.members) == {"sl.pth@1", own, str(exp)}  # own snapshots stay
    assert league.league(4).min_shares == {"sl.pth@1": 0.125, str(exp): 0.1}
    assert league.league(2).min_shares == {"sl.pth@1": 0.125}
    assert gpg.reload_population(league, _league_cfg(pop), run_dir, known) == {"added": [], "removed": []}
    # a resume: without the reload the restored league keeps what the file dropped (legacy), with it the file rules
    state = league.state()
    pop.write_text(json.dumps({"members": [{"path": "sl.pth", "temperature": 1.0}]}))
    kept = gpg.league_for_run(_league_cfg(pop), state, run_dir)
    assert set(kept.members) == {"sl.pth@1", own, str(exp)}
    pruned = gpg.league_for_run(_league_cfg(pop, population_reload_every=2), state, run_dir)
    assert set(pruned.members) == {"sl.pth@1", own}
    assert pruned.league(4).min_shares == {}


# ---------------------------------------------------------------------------------------------------
# A2: prioritized starts
# ---------------------------------------------------------------------------------------------------


def _write_lines(path: Path, records) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


@pytest.fixture
def library_files(tmp_path):
    """Synthetic 4-player start files (the sampler never replays them): three SR1 starts, two mid-game games."""
    starts = [{"id": f"g{i}", "num_players": 4, "actions": [{"i": i}]} for i in range(3)]
    mids = [
        {"id": "m0", "num_players": 4, "actions": [{"k": k} for k in range(30)], "cuts": {"2": 10, "3": 20}},
        {"id": "m1", "num_players": 4, "actions": [{"k": k} for k in range(30)], "cuts": {"2": 12, "4": 25}},
        {"id": "x", "num_players": 3, "actions": [], "cuts": {"2": 1}},
    ]
    return _write_lines(tmp_path / "s_4p.jsonl", starts), _write_lines(tmp_path / "m_4p.jsonl", mids)


def test_start_picker_chooses_within_a_kind_and_falls_back_to_uniform(library_files):
    sfile, mfile = library_files
    # no picker / an empty one: the legacy draws exactly
    for seed in range(20):
        kwargs = dict(midgame_path=mfile, midgame_fraction=0.5, rng=random.Random(seed))
        base = sp.sample_start_position(4, sfile, 0.2, **kwargs)
        kwargs["rng"] = random.Random(seed)
        assert sp.sample_start_position(4, sfile, 0.2, picker=gp.StartPicker({}), **kwargs) == base
    picker = gp.StartPicker({"human": ["human:g2", None, "human:nope"], "midgame": ["midgame:m1:sr4", "midgame:m0:sr9"]})
    starts = [sp.sample_start_position(4, sfile, 0.0, rng=random.Random(i), picker=picker) for i in range(4)]
    assert starts[0].label == "human:g2" and starts[0].actions == ({"i": 2},)
    assert all(s.label.startswith("human:") for s in starts[1:])
    mid = sp.sample_start_position(4, sfile, 0.0, rng=random.Random(0), midgame_path=mfile, midgame_fraction=1.0,
                                   picker=picker)  # fmt: skip
    assert mid.label == "midgame:m1:sr4" and len(mid.actions) == 25
    mid = sp.sample_start_position(4, sfile, 0.0, rng=random.Random(0), midgame_path=mfile, midgame_fraction=1.0,
                                   picker=picker)  # fmt: skip
    assert mid.label.startswith("midgame:m")  # sr9 does not exist: uniform
    assert picker.counts == {"human_prioritized": 2, "human_uniform": 1, "human_missing": 1, "human_exhausted": 1,
                             "midgame_prioritized": 2, "midgame_missing": 1}  # fmt: skip
    # the generic sampler takes the same picks (1867-style tuples carry seed and rules)
    g = gp.StartPicker({"midgame": ["midgame:m0:sr3"]})
    games = [("m0", tuple(range(30)), {2: 10, 3: 20}, None, ()), ("m1", tuple(range(30)), {2: 12}, None, ())]
    assert g.midgame(games) == (games[0], 3)


def test_start_priorities_score_rank_staleness_and_state(library_files):
    sfile, mfile = library_files
    prio = gp.StartPriorities.for_run([3, 4], sfile.replace("4p", "{n}p"), mfile.replace("4p", "{n}p"),
                                      uniform=0.0, staleness=0.0, temperature=1.0, ema=0.5)  # fmt: skip
    lib = prio.libraries[4]
    assert lib["human"] == ["human:g0", "human:g1", "human:g2"]
    assert lib["midgame"] == ["midgame:m0:sr2", "midgame:m0:sr3", "midgame:m1:sr2", "midgame:m1:sr4"]
    assert prio.libraries[3] == {"human": [], "midgame": []}  # (3p files don't exist; the 4p one has no 3p lines)
    rng = np.random.default_rng(0)
    assert prio.picks(4, 5, rng) == {"human": [None] * 5, "midgame": [None] * 5}  # nothing scored yet: uniform

    def row(label, adv):
        return gpg.PGRow(tokens=b"", rows=None, price_range=None, choice=0, old_logp=0.0, price_cell=-1, seat=0,
                         num_players=4, advantage=adv, start=label)  # fmt: skip

    for label in ("human:g0", "human:g1", "auction", "midgame:m1:sr4"):
        prio.observe_game(4, label)
    scored = prio.observe_rows([row("human:g0", 0.1), row("human:g0", -0.3), row("human:g1", 0.05),
                                row("midgame:m1:sr4", -0.4), row("auction", 0.9), row("random", 0.9)])  # fmt: skip
    assert scored == 3
    assert prio.scores[4] == {"human:g0": pytest.approx(0.2), "human:g1": pytest.approx(0.05),
                              "midgame:m1:sr4": pytest.approx(0.4)}  # fmt: skip
    labels, p = prio.distribution(4, "human")
    assert dict(zip(labels, p)) == {"human:g0": pytest.approx(2 / 3), "human:g1": pytest.approx(1 / 3)}
    prio.observe_rows([row("human:g1", 0.55)])  # ema 0.5: 0.05 -> 0.3, now first
    labels, p = prio.distribution(4, "human")
    assert dict(zip(labels, p))["human:g1"] == pytest.approx(2 / 3)
    draws = prio.picks(4, 2000, rng)["human"]
    assert abs(draws.count("human:g1") / 2000 - 2 / 3) < 0.04 and None not in draws
    # pos_adv scores only the positive part
    pos = gp.StartPriorities(prio.libraries, score="pos_adv")
    pos.observe_rows([row("human:g0", -0.5), row("human:g0", 0.3)])
    assert pos.scores[4]["human:g0"] == pytest.approx(0.15)
    # staleness: a start not played for long gains replay mass
    stale = gp.StartPriorities(prio.libraries, uniform=0.0, staleness=1.0)
    stale.observe_game(4, "human:g0")
    stale.observe_game(4, "human:g1")
    for _ in range(8):
        stale.observe_game(4, "human:g1")
    stale.observe_rows([row("human:g0", 0.1), row("human:g1", 0.9)])
    labels, p = stale.distribution(4, "human")
    assert dict(zip(labels, p)) == {"human:g0": pytest.approx(9 / 9), "human:g1": pytest.approx(0.0)}
    # a uniform share of the draws stays the run's own choice
    half = gp.StartPriorities(prio.libraries, uniform=0.5)
    half.load_state(prio.state())
    draws = half.picks(4, 4000, np.random.default_rng(1))["human"]
    assert abs(draws.count(None) / 4000 - 0.5) < 0.03
    # state round trip; a start the files no longer hold is dropped
    again = gp.StartPriorities(prio.libraries)
    state = prio.state()
    state["scores"]["4"]["human:gone"] = 9.0
    again.load_state(state)
    assert again.scores[4] == prio.scores[4] and again.games[4] == prio.games[4] and again.plays == prio.plays
    top = prio.summary()["p4"]["human"]
    assert top["library"] == 3 and top["scored"] == 2 and top["top"][0][0] == "human:g1"


def test_games_report_start_labels_and_picks(clients, library_files, tmp_path):
    """A pool task with picks: the games start from the picked starts (here the records' actions don't replay, so
    the walk-built files of the end-to-end test are used there; this checks the plumbing with a random start)."""
    empty = str(tmp_path / "e.jsonl")
    Path(empty).write_text("")
    result = gpg.play_generic_pg_games(2, _settings(empty, start_picks={"human": ["human:none"]}, max_decisions=30))
    for g in result["games"]:
        assert g["start_label"] == "random"
        assert all(r.start == "random" for r in g["rows"])
    assert result["stats"].get("start_human_prioritized", 0) == 0  # no human starts at all: nothing taken


# ---------------------------------------------------------------------------------------------------
# End to end: A1 + A2 + reload + seed (with a resume), and the main-exploiter mode
# ---------------------------------------------------------------------------------------------------


def _walk_records(num_players: int, seed: int):
    """A random 1830 game's start / mid-game records (as the start-file builder writes them)."""
    rng = random.Random(seed)
    raw = raw_game(new_game("1830", num_players))
    stock_rounds, previous = 0, None
    while not raw.finished and stock_rounds < 4:
        legal = [int(i) for i in raw.factored_legal_indices()]
        index = rng.choice(legal)
        price_range = raw.price_range_for_index(index)
        raw.apply_action_index(index, price_range[0] if price_range else None)
        rt = raw.round.round_type
        stock_rounds += rt == "Stock" and previous != "Stock"
        previous = rt
    replayed = gp.ReplayedGame(f"w{seed}", num_players, None, [dict(a) for a in raw.raw_actions])
    start, midgame, _ = gp.start_records("1830", replayed, first_round=2, last_round=6)
    return start, midgame


@pytest.fixture(scope="module")
def walk_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("walks")
    for n in (3, 4):
        records = [_walk_records(n, seed) for seed in range(2)]
        _write_lines(root / f"starts_{n}p.jsonl", [s for s, _ in records if s])
        _write_lines(root / f"mid_{n}p.jsonl", [m for _, m in records if m])
    return str(root / "starts_{n}p.jsonl"), str(root / "mid_{n}p.jsonl")


def _spawn():
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)


def test_run_with_tempered_members_reference_kl_prioritized_starts_and_resume(tmp_path, checkpoint, walk_files):
    _spawn()
    starts, mids = walk_files
    pop = tmp_path / "league.json"
    pop.write_text(json.dumps({"members": [
        {"path": checkpoint, "temperature": 1.0, "min_share": 0.25},
        {"path": checkpoint, "temperature": 0.5, "min_share_by_players": {"4": 0.25}},
    ]}))  # fmt: skip
    cfg = gpg.GenericPGConfig(
        policy_checkpoint=checkpoint, out_dir=str(tmp_path / "runs"), run_name="a4", player_mix={3: 1, 4: 1},
        workers=2, games_per_task=3, tasks_in_flight_per_worker=1, positions_per_seat=4, max_decisions=40,
        start_positions=starts, midgame_positions=mids, random_start_fraction=0.0, midgame_fraction=0.5,
        population=str(pop), opponent_slots=1, rows_per_update=24, minibatch=16, microbatch=8, max_updates=2,
        snapshot_every=1, slot_refresh_every=1, league_snapshot_every=1, reward="placement",
        learner_temperature=0.5, opponent_temperature=0.5, server_cpu=True, server_threads=1,
        inference_batch_size=64, threads=2, ref_kl_coef=0.05, ref_temperature=1.0, start_priority=True,
        start_priority_uniform=0.2, population_reload_every=1, seed=5,
    )  # fmt: skip
    run_dir = gpg.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    for h in history:
        assert np.isfinite(h["kl_ref"]) and h["kl_ref"] > 0 and h["ref_kl_coef"] == 0.05
        assert set(h["start_priority"]) == {"p3", "p4"}
        assert h["starts"].get("human", 0) + h["starts"].get("midgame", 0) == h["games"]
        assert any(k.startswith("human_") or k.startswith("midgame_") for k in h["start_picks"])
    assert sum(kind["scored"] for count in history[-1]["start_priority"].values() for kind in count.values()) >= 1
    assert set(history[-1]["league_slots"].values()) <= {f"{checkpoint}@1", f"{checkpoint}@0.5"} | {
        str(run_dir / "learner" / f"{u}.pth") for u in (1, 2)
    }
    state = json.loads((run_dir / "state.json").read_text())
    assert state["start_priorities"] == gpg.START_PRIORITIES_FILE
    saved = json.loads((run_dir / gpg.START_PRIORITIES_FILE).read_text())
    assert sum(len(v) for v in saved["scores"].values()) >= 1
    assert {f"{checkpoint}@1", f"{checkpoint}@0.5"} <= set(state["league"]["by_count"]["4"]["stats"])
    assert state["league"]["by_count"]["4"]["min_shares"] == {f"{checkpoint}@1": 0.25, f"{checkpoint}@0.5": 0.25}
    assert state["league"]["by_count"]["3"]["min_shares"] == {f"{checkpoint}@1": 0.25}
    # the file drops the T 0.5 member while stopped: the resume (reload on) drops it too, scores carry over
    pop.write_text(json.dumps({"members": [{"path": checkpoint, "temperature": 1.0, "min_share": 0.25}]}))
    gpg.run(gpg.GenericPGConfig(policy_checkpoint=checkpoint, max_updates=3), resume=str(run_dir),
            explicit={"max_updates"})  # fmt: skip
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2, 3]
    state = json.loads((run_dir / "state.json").read_text())
    assert f"{checkpoint}@0.5" not in state["league"]["by_count"]["4"]["stats"]
    resumed = json.loads((run_dir / gpg.START_PRIORITIES_FILE).read_text())
    assert sum(resumed["games"].values()) > sum(saved["games"].values())


def test_exploiter_mode_plays_only_the_frozen_main(tmp_path, checkpoint, start_file):
    _spawn()
    main = tmp_path / "main.pth"
    mg.save_checkpoint(_model(3), main, titles=["1830"])
    template = str(tmp_path / "starts_{n}p.jsonl")
    for n in (2, 4):
        (tmp_path / f"starts_{n}p.jsonl").write_text("")
    common = dict(
        policy_checkpoint=checkpoint, out_dir=str(tmp_path / "runs"), player_mix={2: 1, 4: 1}, workers=2,
        games_per_task=2, tasks_in_flight_per_worker=1, positions_per_seat=4, max_decisions=30,
        start_positions=template, random_start_fraction=1.0, rows_per_update=16, minibatch=16, microbatch=8,
        max_updates=2, snapshot_every=1, slot_refresh_every=1, league_snapshot_every=1, reward="placement",
        learner_temperature=1.0, entropy_coef=0.01, kl_coef=0.0, server_cpu=True, server_threads=1,
        inference_batch_size=64, threads=2, save_game_every=1,
    )  # fmt: skip
    with pytest.raises(SystemExit):
        gpg.run(gpg.GenericPGConfig(run_name="bad", exploiter_of=str(main), population=str(main), **common))
    run_dir = gpg.run(gpg.GenericPGConfig(run_name="mx", exploiter_of=str(main), **common))
    config = json.loads((run_dir / "config.json").read_text())
    assert config["population"] == str(run_dir / gpg.EXPLOITER_POPULATION) and not config["league_add_snapshots"]
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    for h in history:
        assert h["league_slots"] == {"main": str(main)} and h["league_size"] == 1
        assert 0.0 <= h["p2_vs_main_rate"] <= 1.0 and 0.0 <= h["p4_vs_main_rate"] <= 1.0
        assert h["exploiter_of"] == str(main)
    saved = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
    assert saved
    for entry in saved:
        labels = [lab for lab, side in zip(entry["seat_labels"], entry["sides"]) if side == "B"]
        assert labels and set(labels) == {str(main)}
    state = json.loads((run_dir / "state.json").read_text())
    assert list(state["league"]["by_count"]["4"]["stats"]) == [str(main)]  # no own snapshot joined


# ---------------------------------------------------------------------------------------------------
# A5p: practice starts (a mid-game-format file; drawn before the rest of the mix; off by default)
# ---------------------------------------------------------------------------------------------------


def test_practice_starts_off_draw_exactly_as_before_and_on_draw_practice_cuts(library_files):
    sfile, mfile = library_files
    for seed in range(30):  # fraction 0 (or no file): not one draw more -- the legacy starts exactly
        kwargs = dict(midgame_path=mfile, midgame_fraction=0.3, auction_fraction=0.2)
        base = sp.sample_start_position(4, sfile, 0.2, rng=random.Random(seed), **kwargs)
        for practice in (dict(practice_path=mfile, practice_fraction=0.0),
                         dict(practice_path=None, practice_fraction=0.5)):  # fmt: skip
            assert sp.sample_start_position(4, sfile, 0.2, rng=random.Random(seed), **kwargs, **practice) == base
        g = gp.sample_start("1830", 4, sfile, 0.2, rng=random.Random(seed), **kwargs)
        assert gp.sample_start("1830", 4, sfile, 0.2, rng=random.Random(seed), practice_path=mfile,
                               practice_fraction=0.0, **kwargs) == g  # fmt: skip
    # fraction 1: always a practice cut of the count (labels practice:<id>:sr<k>, the prefix up to the cut)
    seen = set()
    cuts = {"m0": {2: 10, 3: 20}, "m1": {2: 12, 4: 25}}
    for seed in range(40):
        s = sp.sample_start_position(4, sfile, 0.2, rng=random.Random(seed), practice_path=mfile, practice_fraction=1.0)
        game_id, _, k = s.label[len(sp.PRACTICE_PREFIX):].rpartition(":sr")
        assert s.label.startswith("practice:") and len(s.actions) == cuts[game_id][int(k)]
        seen.add(s.label)
        g = gp.sample_start("1830", 4, sfile, 0.2, rng=random.Random(seed), practice_path=mfile, practice_fraction=1.0)
        assert g.label.startswith("practice:")
    assert seen == {"practice:m0:sr2", "practice:m0:sr3", "practice:m1:sr2", "practice:m1:sr4"}
    # a count without practice games falls through to the rest of the mix
    s = sp.sample_start_position(2, sfile, 1.0, rng=random.Random(0), practice_path=mfile, practice_fraction=1.0)
    assert s.label == "random"
    # the picker chooses practice cuts as it does mid-game cuts
    picker = gp.StartPicker({"practice": ["practice:m1:sr4", "practice:m0:sr9", None]})
    s = sp.sample_start_position(4, sfile, 0.2, rng=random.Random(0), practice_path=mfile, practice_fraction=1.0,
                                 picker=picker)  # fmt: skip
    assert s.label == "practice:m1:sr4" and len(s.actions) == 25
    for _ in range(2):
        assert sp.sample_start_position(4, sfile, 0.2, rng=random.Random(1), practice_path=mfile,
                                        practice_fraction=1.0, picker=picker).label.startswith("practice:m")
    assert picker.counts == {"practice_prioritized": 2, "practice_missing": 1, "practice_uniform": 1}
    assert gp.start_kind("practice:m1:sr4") == "practice" and gp.start_kind("midgame:m1:sr4") == "midgame"


def test_practice_library_in_the_start_priorities(library_files):
    sfile, mfile = library_files
    without = gp.StartPriorities.for_run([4], sfile, mfile)
    assert set(without.libraries[4]) == {"human", "midgame"}
    assert set(without.picks(4, 3, np.random.default_rng(0))) == {"human", "midgame"}
    prio = gp.StartPriorities.for_run([3, 4], sfile.replace("4p", "{n}p"), mfile.replace("4p", "{n}p"),
                                      mfile.replace("4p", "{n}p"), uniform=0.0, staleness=0.0)  # fmt: skip
    assert prio.libraries[4]["practice"] == ["practice:m0:sr2", "practice:m0:sr3", "practice:m1:sr2",
                                             "practice:m1:sr4"]  # fmt: skip
    assert prio.picks(4, 2, np.random.default_rng(0))["practice"] == [None, None]  # nothing scored: uniform

    def row(label, adv):
        return gpg.PGRow(tokens=b"", rows=None, price_range=None, choice=0, old_logp=0.0, price_cell=-1, seat=0,
                         num_players=4, advantage=adv, start=label)  # fmt: skip

    prio.observe_game(4, "practice:m1:sr4")
    assert prio.observe_rows([row("practice:m1:sr4", -0.6), row("midgame:m1:sr4", 0.1)]) == 2
    assert prio.picks(4, 3, np.random.default_rng(0))["practice"] == ["practice:m1:sr4"] * 3
    assert prio.summary()["p4"]["practice"]["scored"] == 1


def test_practice_config_checks(tmp_path, checkpoint, walk_files):
    starts, mids = walk_files
    common = dict(policy_checkpoint=checkpoint, out_dir=str(tmp_path), player_mix={3: 1, 4: 1}, start_positions=starts)
    gpg.check_config(gpg.GenericPGConfig(**common))  # off: no file needed
    for bad in (dict(practice_fraction=1.0, practice_positions=mids), dict(practice_fraction=-0.1),
                dict(practice_fraction=0.2, practice_positions=mids.replace("{n}", "4")),
                dict(practice_fraction=0.2, practice_positions=str(tmp_path / "none_{n}p.jsonl")),
                dict(practice_fraction=0.2)):  # fmt: skip
        with pytest.raises(SystemExit):
            gpg.check_config(gpg.GenericPGConfig(**common, **bad))
    gpg.check_config(gpg.GenericPGConfig(**common, practice_fraction=0.2, practice_positions=mids))
    cfg = gpg.GenericPGConfig(**common, practice_fraction=0.2)
    gpg.default_start_templates(cfg)
    assert cfg.practice_positions == sp.PRACTICE_TEMPLATE == "human_games/practice_positions_1830_{n}p_v1.jsonl"
    cfg = gpg.GenericPGConfig(**common)
    gpg.default_start_templates(cfg)
    assert cfg.practice_positions is None  # off: no default either


def test_run_with_practice_starts_through_the_priorities(tmp_path, checkpoint, walk_files):
    """A run with practice starts (the walks' mid-game file as the practice library) and A2's priorities: games
    start at practice cuts, their rows score them, and the history counts them."""
    _spawn()
    starts, mids = walk_files
    cfg = gpg.GenericPGConfig(
        policy_checkpoint=checkpoint, out_dir=str(tmp_path / "runs"), run_name="a5p", player_mix={3: 1, 4: 1},
        workers=2, games_per_task=3, tasks_in_flight_per_worker=1, positions_per_seat=4, max_decisions=40,
        start_positions=starts, random_start_fraction=0.0, practice_positions=mids, practice_fraction=0.6,
        rows_per_update=24, minibatch=16, microbatch=8, max_updates=2, snapshot_every=1, reward="placement",
        learner_temperature=0.5, opponent_temperature=0.5, server_cpu=True, server_threads=1,
        inference_batch_size=64, threads=2, start_priority=True, start_priority_uniform=0.2, seed=7,
        save_game_every=1,
    )  # fmt: skip
    run_dir = gpg.run(cfg)
    config = json.loads((run_dir / "config.json").read_text())
    assert config["practice_positions"] == mids and config["practice_fraction"] == 0.6
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    assert sum(h["starts"].get("practice", 0) for h in history) >= 1
    assert all(h["starts"].get("practice", 0) + h["starts"].get("human", 0) == h["games"] for h in history)
    assert any(k.startswith("practice_") for h in history for k in h["start_picks"])
    assert all("practice" in h["start_priority"]["p4"] for h in history)
    saved = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
    assert any(e["start"].startswith("practice:") for e in saved)
