"""The value trainer (rl18xx/agent/alphazero/value_trainer.py, ``main.py value-train``): a saved
game's positions and targets (decisions past the auction, canonical frame, reproducible per uid
and pass), the held-out split, the gate (paired, game-clustered SE) and its escalation, the
cycle trigger, and tiny CPU cycles end to end -- candidates, promotions, promoted.json,
cycles.jsonl, a resume after an interrupted cycle, atomic writes."""

import json
import os
import shutil
import zlib

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_data as sd  # noqa: E402
from rl18xx.agent.alphazero import value_trainer as vt  # noqa: E402
from rl18xx.agent.alphazero.policy_gradient import placement_points  # noqa: E402
from rl18xx.agent.alphazero.start_positions import new_game  # noqa: E402

from .test_search_data import _config, _generate, _save  # noqa: E402


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Real saved games from a tiny CPU search-learning generator (HashNet), in two batches of shards."""
    root = tmp_path_factory.mktemp("value_trainer")
    start = root / "starts.jsonl"
    start.write_text("")
    cfg = _config(str(start), n_fast=2, auction_fraction=0.3)
    # At least 24 games, not a fixed 8 s: on a loaded machine (the pre-push gate beside other jobs) 8 s plays
    # too few, and cycles then run out of positions (2 steps) or fail their gate.
    games, _ = _generate(cfg, seconds=8.0, min_games=24)
    assert len(games) >= 12
    half = len(games) // 2
    data = root / "data"
    first = _save(games[:half], data, cfg, shard_games=2)
    second = _save(games[half:], root / "later", cfg, shard_games=2)
    return {"root": root, "cfg": cfg, "data": data, "later": root / "later", "first": first, "second": second}


def _per_mille(saved: list) -> int:
    """A held-out threshold putting about a third of these games out (uids are random per run)."""
    values = sorted(zlib.crc32(g.uid.encode()) % 1000 for g in saved)
    return values[len(values) // 3] + 1


@pytest.fixture(scope="module")
def init_checkpoint(generated):
    from rl18xx.agent.alphazero.policy_gradient import _save as save_model

    from .test_search_learning import _tiny_model

    path = generated["root"] / "init" / "learner" / "0.pth"
    path.parent.mkdir(parents=True)
    (path.parent.parent / "config.json").write_text(json.dumps({"learner_temperature": 1.0}))
    save_model(_tiny_model(0), path)
    return path


@pytest.fixture(scope="module")
def fixed_dir(generated):
    """A value-gate data directory (manifest.json + shards/*.npz) made of the second batch's games:
    some "test" (the fixed set), some "train" (never loaded)."""
    out = generated["root"] / "fixed"
    (out / "shards").mkdir(parents=True)
    games, parts = [], []
    for i, g in enumerate(generated["second"]):
        pos = vt.game_positions(g.meta, g.actions, 4, 0)
        if pos["game_states"] is None:
            continue
        n = len(pos["index"])
        win, placement = vt.canonical_targets(
            np.array([g.meta["win_share"]], np.float32), np.array([g.meta["net_worth"]]), np.zeros(n, np.int32),
            pos["rotation"],
        )
        parts.append({"game_states": pos["game_states"], "nodes": pos["nodes"], "win": win, "placement": placement,
                      "rotation": pos["rotation"], "game": np.full(n, len(games), np.int32)})
        games.append({"id": g.uid, "positions": n, "split": "test" if i % 3 else "train"})
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(out / "shards" / "shard_00000.npz", **arrays)
    (out / "manifest.json").write_text(json.dumps({"shards": ["shard_00000.npz"], "games": games}))
    return out


def _trainer_config(tmp_path, generated, init_checkpoint, fixed_dir=None, **overrides) -> vt.ValueTrainerConfig:
    cfg = vt.ValueTrainerConfig(
        run_dir=str(tmp_path / "run"), data=[str(generated["data"])], init=str(init_checkpoint),
        fixed_data=None if fixed_dir is None else str(fixed_dir), positions_per_game=4, rolling_games=1000,
        held_out_per_mille=_per_mille(generated["first"] + generated["second"]), min_new_games=1, poll_seconds=0.2,
        lr=1e-3, batch_size=8, eval_every=2, patience=2, max_steps=30, workers=0, task_games=3, buffer_positions=48,
        eval_batch_size=32, device="cpu",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ----------------------------------------------------------------------------- positions


def test_positions_are_decisions_past_the_auction_in_the_canonical_frame(generated):
    path = sd.shard_paths(generated["data"])[0]
    header, logs = vt.read_shard_actions(path)
    games = sd.read_shard(path)
    assert logs == [g.actions_json for g in games]  # the head-only read gives read_shard's bytes
    assert [{k: v for k, v in m.items() if k != "counts"} for m in header["games"]] == [g.meta for g in games]

    differs, with_positions = 0, 0
    for g in generated["first"] + generated["second"]:
        pos = vt.game_positions(g.meta, g.actions, 5, 3)
        assert len(pos["index"]) <= 5 and np.all(np.diff(pos["index"]) > 0)
        if not len(pos["index"]):  # an auction start that ended at max_decisions in (or just past) the auction
            continue
        with_positions += 1
        rs = new_game(4, g.meta["auction_unlock"])._game
        wanted = {int(i): k for k, i in enumerate(pos["index"])}
        for i, action in enumerate(g.actions):
            if i in wanted:
                k = wanted[i]
                assert rs.round.round_type != "Auction" and len(rs.factored_legal_indices()) > 1
                state, nodes, _, rotation, _ = rs.encode_canonical_py()
                assert np.array_equal(pos["game_states"][k], state)
                assert pos["nodes"].dtype == np.float16 and np.array_equal(pos["nodes"][k], nodes.astype(np.float16))
                assert pos["rotation"][k] == rotation
            rs.process_action(action)
        # Reproducible per (uid, pass); another pass draws again.
        again = vt.game_positions(g.meta, g.actions, 5, 3)
        assert np.array_equal(again["index"], pos["index"]) and np.array_equal(again["game_states"], pos["game_states"])
        differs += not np.array_equal(vt.game_positions(g.meta, g.actions, 5, 4)["index"], pos["index"])
        # Targets: the outcome rolled into the mover-first frame.
        n = len(pos["index"])
        win, placement = vt.canonical_targets(
            np.array([g.meta["win_share"]], np.float32), np.array([g.meta["net_worth"]]), np.zeros(n, np.int32),
            pos["rotation"],
        )
        points = placement_points(np.array(g.meta["net_worth"]))
        for k, r in enumerate(pos["rotation"]):
            assert np.allclose(win[k], np.roll(np.array(g.meta["win_share"], np.float32), -int(r)))
            assert np.allclose(placement[k], np.roll(points, -int(r)))
    assert differs > 0 and with_positions >= 0.75 * len(generated["first"] + generated["second"])
    assert any(r != 0 for g in generated["first"] for r in vt.game_positions(g.meta, g.actions, 5, 3)["rotation"])


def test_an_encode_task_returns_positions_with_their_games_outcomes(generated):
    path = sd.shard_paths(generated["data"])[0]
    games = sd.read_shard(path)
    slots = list(range(len(games)))
    task = {"k": 3, "pass_id": 7, "pieces": [(str(path), slots, [100 + i for i in slots])]}
    out = vt.result_positions(vt.encode_task(task))
    assert set(out["game"].tolist()) <= {100 + i for i in range(len(games))}
    for i, g in enumerate(games):
        rows = out["game"] == 100 + i
        pos = vt.game_positions(g.meta, g.actions, 3, 7)
        if pos["game_states"] is None:  # a game with no position to keep: the task returns none of its rows either
            assert not rows.any()
            continue
        assert np.array_equal(out["game_states"][rows], pos["game_states"])
        assert np.allclose(out["win"][rows].sum(1), 1.0)


# ----------------------------------------------------------------------------- sampling


def _synthetic_index(n=5000, seed=0) -> vt.GameIndex:
    rng = np.random.default_rng(seed)
    players = np.full(n, 4, np.int8)
    players[rng.choice(n, 50, replace=False)] = 3
    return vt.GameIndex(paths=[f"s{i}" for i in range(n // 10)], shard=np.arange(n) // 10, slot=np.arange(n) % 10,
                        crc=rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32), players=players)


def test_held_out_games_are_never_sampled_and_form_the_rolling_set():
    index = _synthetic_index()
    held = index.crc % 1000 < 30
    trainable = np.flatnonzero(~held & (index.players == 4))
    rng = np.random.default_rng(0)
    recent, old = vt.choose_sample(index, 30, 1000, 1.0, rng)
    assert np.array_equal(recent, trainable[-1000:])  # every trainable game of the newest
    assert len(old) == 1000 and len(np.unique(old)) == 1000 and old.max() < recent.min()
    assert not held[np.concatenate([recent, old])].any() and (index.players[np.concatenate([recent, old])] == 4).all()
    recent2, old2 = vt.choose_sample(index, 30, 1000, 2.0, np.random.default_rng(0))  # twice the sample
    assert len(recent2) == 2000 and len(old2) == 2000
    recent3, old3 = vt.choose_sample(index, 30, 3000, 1.0, np.random.default_rng(0))  # fewer older games than recent
    assert len(recent3) == 3000 and len(old3) == len(trainable) - 3000

    cfg = vt.ValueTrainerConfig(run_dir="x", rolling_games=2000, rolling_positions_per_game=4,
                                rolling_max_positions=10_000)
    rolling = vt.rolling_games(index, cfg)
    newest = np.arange(len(index) - 2000, len(index))
    assert np.array_equal(rolling, newest[held[newest] & (index.players[newest] == 4)])
    cfg.rolling_max_positions = 4 * 10
    capped = vt.rolling_games(index, cfg)
    assert np.array_equal(capped, rolling[-10:])  # the newest ten


def test_tasks_group_games_by_shard():
    index = _synthetic_index(200)
    games = np.random.default_rng(1).permutation(200)[:60]
    tasks = vt.make_tasks(index, games, 4, 9, task_games=7)
    seen = []
    for task in tasks:
        assert task["k"] == 4 and task["pass_id"] == 9 and sum(len(s) for _, s, _ in task["pieces"]) <= 7
        for path, slots, ids in task["pieces"]:
            assert all(index.paths[index.shard[g]] == path and index.slot[g] == s for s, g in zip(slots, ids))
            seen.extend(ids)
    assert sorted(seen) == sorted(games.tolist())
    pieces = [p for t in tasks for p, _, _ in t["pieces"]]
    assert len(set(pieces)) == len({index.shard[g] for g in games})  # each shard once (none split here)


def test_the_shuffle_buffer_draws_every_position_once():
    rng = np.random.default_rng(0)
    example = {"game_states": np.zeros((1, 3), np.float32), "nodes": np.zeros((1, 2, 2), np.float16),
               "win": np.zeros((1, 4), np.float32), "placement": np.zeros((1, 4), np.float32)}
    buf = vt.ShuffleBuffer(10, example)
    ids = np.arange(37)

    def positions(lo, hi):
        return {"game_states": np.repeat(ids[lo:hi, None], 3, 1).astype(np.float32),
                "nodes": np.zeros((hi - lo, 2, 2), np.float16), "win": np.repeat(ids[lo:hi, None], 4, 1).astype(
                    np.float32), "placement": np.zeros((hi - lo, 4), np.float32)}

    drawn, at = [], 0
    while at < len(ids) or buf.n:
        if at < len(ids):
            at += buf.add(positions(at, min(len(ids), at + 6)))
        batch = buf.take(min(4, buf.n), rng)
        assert np.array_equal(batch["win"][:, 0], batch["game_states"][:, 0])  # rows stay whole
        drawn.extend(batch["win"][:, 0].astype(int).tolist())
    assert sorted(drawn) == ids.tolist()


def test_a_lost_encoding_task_fails_the_cycle_instead_of_hanging():
    class Lost:  # a task whose worker was killed: Pool never completes it
        def ready(self):
            return False

        def wait(self, timeout=None):
            pass

    class Pool:
        def apply_async(self, fn, args):
            return Lost()

    stream = vt.Stream(Pool(), iter([{"k": 1}]), 2, timeout=0.0)
    with pytest.raises(RuntimeError, match="encoding task"):
        stream.collect(block=True)


# ----------------------------------------------------------------------------- the gate


def test_paired_stats_cluster_the_se_by_game():
    rng = np.random.default_rng(0)
    games = np.repeat(np.arange(50), 4)
    old = rng.random(200) + 0.5
    new = old - 0.01 + np.repeat(rng.normal(0, 0.02, 50), 4) + rng.normal(0, 0.005, 200)
    s = vt.paired_stats(new, old, games)
    per_game = (new - old).reshape(50, 4).mean(1)
    assert s["n"] == 200 and s["games"] == 50 and np.isclose(s["diff"], per_game.mean())
    assert np.isclose(s["se"], per_game.std(ddof=1) / np.sqrt(50))
    assert s["se"] > (new - old).std(ddof=1) / np.sqrt(200)  # clustering widens it: positions share a game
    # Unequal positions per game: the cluster-robust SE of the position mean.
    games = np.repeat(np.arange(5), [1, 2, 3, 4, 5])
    d = np.arange(15, dtype=float) / 10
    s = vt.paired_stats(d, np.zeros(15), games)
    sums, counts = np.bincount(games, weights=d), np.bincount(games)
    expected = np.sqrt(5 / 4 * ((sums - counts * d.mean()) ** 2).sum()) / 15
    assert np.isclose(s["se"], expected) and np.isclose(s["diff"], d.mean())
    assert vt.paired_stats([], [], [])["n"] == 0 and vt.paired_stats([1.0], [0.5], [3])["se"] is None


def _stats(diff, se, ce_current=0.9, n=1000):
    return {"n": n, "games": n // 4, "ce": ce_current + diff, "ce_current": ce_current, "diff": diff, "se": se}


def test_the_gate_promotes_rejects_and_flags_divergence():
    gate = vt.gate_decision
    assert gate(_stats(-0.003, 0.001), _stats(0.001, 0.001)) == (True, False, "promoted")
    assert gate(_stats(-0.003, 0.001), None)[0]  # no fixed set
    promoted, diverged, reason = gate(_stats(-0.0015, 0.001), _stats(0.0, 0.001))
    assert not promoted and not diverged and "rolling gain" in reason
    promoted, diverged, reason = gate(_stats(-0.003, 0.001), _stats(0.0025, 0.001))  # fixed worse by > 2 SE
    assert not promoted and not diverged and "fixed set worse" in reason
    assert gate(_stats(-0.003, 0.001), _stats(0.0019, 0.001))[0]  # fixed worse, within 2 SE
    promoted, diverged, reason = gate(_stats(0.011, 0.001), _stats(0.0, 0.001))
    assert not promoted and diverged and "diverged" in reason
    promoted, diverged, _ = gate(_stats(-0.003, 0.001), _stats(0.0, 0.001), start_ce=0.90, final_ce=0.92)
    assert not promoted and diverged  # training ran away from its start, whatever the best state
    assert gate(_stats(0.0, 0.0), None)[0] is False  # an unchanged candidate: no gain
    assert gate(_stats(-0.01, None), None)[0] is False  # one game: no SE
    assert gate(None, None) == (False, False, "no rolling held-out positions")
    assert gate(_stats(-0.003, 0.0012), None, z=3.0)[0] is False


def test_with_a_gate_set_the_gain_is_required_there_and_the_rolling_set_only_checked_for_divergence():
    gate = vt.gate_decision
    # The rolling set no longer decides: a rolling gain is not enough, a rolling loss within diverge_ce doesn't block.
    promoted, _, reason = gate(_stats(-0.02, 0.005), _stats(0.0, 0.001), gate=_stats(-0.001, 0.001))
    assert not promoted and "gate-set gain too small" in reason
    assert gate(_stats(0.005, 0.005), _stats(0.0, 0.001), gate=_stats(-0.003, 0.001)) == (True, False, "promoted")
    assert gate(None, _stats(0.0, 0.001), gate=_stats(-0.003, 0.001))[0]  # no rolling positions yet
    promoted, _, reason = gate(_stats(0.0, 0.005), _stats(0.0025, 0.001), gate=_stats(-0.003, 0.001))
    assert not promoted and "fixed set worse" in reason  # the fixed guard stays
    promoted, diverged, reason = gate(_stats(0.011, 0.005), None, gate=_stats(-0.003, 0.001))
    assert not promoted and diverged and "rolling CE" in reason
    promoted, diverged, reason = gate(_stats(0.0, 0.005), None, gate=_stats(0.011, 0.001))
    assert not promoted and diverged and "gate CE" in reason
    assert gate(_stats(-0.003, 0.001), None, gate=_stats(0.0, 0.0, n=0)) == (False, False, "no gate-set positions")


def test_gate_sets_from_several_directories_keep_their_games_apart(fixed_dir):
    one = vt.load_fixed(fixed_dir)
    both = vt.load_sets([fixed_dir, fixed_dir])
    assert len(both["win"]) == 2 * len(one["win"])
    games = len(np.unique(one["game"]))
    assert len(np.unique(both["game"])) == 2 * games
    s = vt.paired_stats(np.ones(len(both["win"])), np.zeros(len(both["win"])), both["game"])
    assert s["games"] == 2 * games
    assert vt.load_sets([]) is None


def test_failures_escalate_the_schedule_and_then_ask_for_a_review():
    cfg = vt.ValueTrainerConfig(run_dir="x", cycle_hours=12, cycle_games=40_000, sample_scale=1.0)
    schedule = [vt.escalation(cfg, f) for f in range(8)]
    assert [s["cycle_hours"] for s in schedule] == [12, 12, 12, 24, 24, 24, 24, 24]
    assert [s["cycle_games"] for s in schedule] == [40_000] * 3 + [80_000] * 5
    assert [s["sample_scale"] for s in schedule] == [1.0] * 3 + [2.0] * 5
    assert [s["review"] for s in schedule] == [False] * 6 + [True] * 2


# ----------------------------------------------------------------------------- the trainer


def test_writes_are_atomic(tmp_path, monkeypatch):
    path = tmp_path / "promoted.json"
    vt.write_json_atomic(path, {"current": 1})

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(vt.os, "replace", fail)
    with pytest.raises(OSError):
        vt.write_json_atomic(path, {"current": 2})
    assert json.loads(path.read_text()) == {"current": 1} and os.listdir(tmp_path) == ["promoted.json"]
    monkeypatch.undo()
    log = tmp_path / "cycles.jsonl"
    vt.append_jsonl(log, {"cycle": 1})
    with open(log, "a") as f:
        f.write('{"cycle": 2, "tor')  # a crash mid-append
    assert vt.read_jsonl(log) == [{"cycle": 1}]
    vt.repair_jsonl(log)
    vt.append_jsonl(log, {"cycle": 2})
    assert vt.read_jsonl(log) == [{"cycle": 1}, {"cycle": 2}]


def test_a_tiny_cycle_end_to_end(tmp_path, generated, init_checkpoint, fixed_dir):
    data = tmp_path / "data"
    shutil.copytree(generated["data"], data)  # new shards arrive below
    cfg = _trainer_config(tmp_path, generated, init_checkpoint, fixed_dir, data=[str(data)], workers=1, epochs=2.0,
                          cycle_games=1, patience=100, max_steps=0,  # both passes in full, whatever the sample's size
                          buffer_positions=16,  # two passes through a small buffer
                          diverge_ce=10.0)  # a tiny net at lr 1e-3 on a few dozen held-out positions: noise, not a test
    trainer = vt.ValueTrainer(cfg)
    run = tmp_path / "run"
    fixed = vt.load_fixed(fixed_dir)
    manifest = json.loads((fixed_dir / "manifest.json").read_text())
    assert len(fixed["win"]) == sum(g["positions"] for g in manifest["games"] if g["split"] == "test")
    promoted = json.loads((run / "promoted.json").read_text())
    v0 = promoted["versions"][0]
    assert promoted["current"] == 0 and v0["path"] == "promoted/0.pth" and v0["cycle"] == 0
    assert v0["rolling_ce"] is None and v0["fixed_ce"] > 0 and v0["games_total"] == len(generated["first"])
    assert (run / "promoted" / "0.pth").read_bytes() == init_checkpoint.read_bytes()
    assert trainer.due()[0]

    assert trainer.serve(max_cycles=1) == 1
    lines = vt.read_jsonl(run / "cycles.jsonl")
    assert len(lines) == 1
    line = lines[0]
    assert set(vt.CYCLE_FIELDS) <= set(line)
    assert line["cycle"] == 1 and line["start_version"] == 0 and line["candidate"] == "candidates/1.pth"
    assert line["games_total"] == len(generated["first"]) and line["games_trained"] >= line["games_encoded"] > 0
    assert line["games_encoded"] == len(trainer.last_cycle["trained_crc"])
    assert line["positions_trained"] > cfg.buffer_positions and line["steps"] == -(-line["positions_trained"] // 8)
    assert line["epochs"] == 2.0 and line["best_step"] in [s for s, _ in line["history"]]  # both passes trained
    assert line["fixed_n"] == len(fixed["win"]) and line["rolling_n"] > 0 and line["rolling_games"] > 0
    assert np.isclose(line["fixed_ce_current"], v0["fixed_ce"])
    assert line["promoted"] == (line["version"] == 1) and not line["diverged"]
    assert (run / "candidates" / "1.pth").exists()
    from rl18xx.agent.alphazero.policy_gradient import _load

    _load(run / "candidates" / "1.pth")
    state = json.loads((run / "state.json").read_text())
    assert state["cycle"] == 1 and state["data_cut"] == line["data_cut"]
    assert json.loads((run / "status.json").read_text())["stage"] == "stopped"
    assert json.loads((run / "config.json").read_text())["data"] == [str(data)]
    indexes = list((run / "index").rglob("*.sld.json"))
    assert len(indexes) == len(sd.shard_paths(data))
    # Held-out games: never trained, and the rolling set is made of them.
    per_mille = cfg.held_out_per_mille
    assert len(trainer.last_cycle["trained_crc"]) > 0 and len(trainer.last_cycle["rolling_crc"]) > 0
    assert (trainer.last_cycle["trained_crc"] % 1000 >= per_mille).all()
    assert (trainer.last_cycle["rolling_crc"] % 1000 < per_mille).all()
    # Nothing new: not due. New shards arrive: a cycle is due again (rescan).
    assert not trainer.due()[0]
    with sd.SearchDataWriter(data, {"run_name": "later"}, 2) as writer:
        for g in generated["second"]:
            writer.add(g.packed())
    trainer.scan()
    due, why = trainer.due()
    assert due and "new trainable" in why


def test_a_promotion_links_the_candidate_into_promoted(tmp_path, generated, init_checkpoint, monkeypatch):
    # min_improvement -1: every evaluation is "the best" (best_step > 0); the gate passes it.
    cfg = _trainer_config(tmp_path, generated, init_checkpoint, min_improvement=-1.0)
    trainer = vt.ValueTrainer(cfg)
    monkeypatch.setattr(vt, "gate_decision", lambda *a, **k: (True, False, "promoted"))
    line = trainer.run_cycle()
    run = tmp_path / "run"
    assert line["promoted"] and line["best_step"] > 0 and line["reason"] == "promoted"
    promoted = json.loads((run / "promoted.json").read_text())
    assert promoted["current"] == 1 and [v["version"] for v in promoted["versions"]] == [0, 1]
    v1 = promoted["versions"][1]
    assert v1["path"] == "promoted/1.pth" and v1["cycle"] == 1 and v1["candidate"] == "candidates/1.pth"
    assert v1["rolling_ce"] == line["rolling_ce"] and v1["fixed_ce"] is None
    assert v1["games_total"] == line["games_total"]
    assert os.path.samefile(run / "promoted" / "1.pth", run / "candidates" / "1.pth")
    assert line["version"] == 1 and line["consecutive_failures"] == 0
    assert json.loads((run / "state.json").read_text())["consecutive_failures"] == 0
    # The next cycle warm-starts from the promoted version and promotes version 2.
    second = trainer.run_cycle()
    assert second["start_version"] == 1 and second["version"] == 2 and second["candidate"] == "candidates/2.pth"
    assert json.loads((run / "promoted.json").read_text())["current"] == 2


def test_failures_count_up_and_double_the_next_sample(tmp_path, generated, init_checkpoint, monkeypatch):
    cfg = _trainer_config(tmp_path, generated, init_checkpoint, escalate_after=2, max_steps=4)
    trainer = vt.ValueTrainer(cfg)
    monkeypatch.setattr(vt, "gate_decision", lambda *a, **k: (False, False, "rejected"))
    lines = [trainer.run_cycle() for _ in range(5)]
    assert [x["consecutive_failures"] for x in lines] == [1, 2, 3, 4, 5]
    assert [x["sample_scale_used"] for x in lines] == [1.0, 1.0, 2.0, 2.0, 2.0]
    assert [x["cycle_hours"] for x in lines] == [12.0, 24.0, 24.0, 24.0, 24.0]
    assert [x["review"] for x in lines] == [False, False, False, True, True]
    assert "review the data mix" in lines[-1]["reason"] and all(x["version"] == 0 for x in lines)


def test_a_cycle_with_an_anchor_and_a_gate_set(tmp_path, generated, init_checkpoint, fixed_dir):
    # Gate set = the fixture's test games, anchor = its train games. patience 1 would stop a rolling-gated cycle
    # at its first evaluation without a gain; with a gate set there is no early stopping.
    cfg = _trainer_config(tmp_path, generated, init_checkpoint, fixed_dir, gate_data=[str(fixed_dir)],
                          anchor_data=str(fixed_dir), anchor_share=0.5, patience=1, max_steps=0, epochs=1.0,
                          diverge_ce=10.0)  # (a tiny net at lr 1e-3: noise, not a test)
    trainer = vt.ValueTrainer(cfg)
    gate_set = vt.load_fixed(fixed_dir, "test")
    anchor = vt.load_fixed(fixed_dir, "train")
    assert len(trainer.gate["win"]) == len(gate_set["win"])
    line = trainer.run_cycle()
    assert line["gate_n"] == len(gate_set["win"]) and line["gate_games"] == len(np.unique(gate_set["game"]))
    assert line["gate_diff"] is not None and np.isclose(line["gate_diff"], line["gate_ce"] - line["gate_ce_current"])
    # Half of each batch from the anchor (4 + 4 at batch 8), the passes counting the stream's positions only.
    assert line["steps"] == -(-line["positions_trained"] // 4)
    assert abs(line["anchor_positions"] - line["positions_trained"]) <= line["steps"]
    assert np.isclose(line["anchor_passes"], line["anchor_positions"] / len(anchor["win"]), atol=1e-4)
    assert line["epochs"] == 1.0
    # No early stopping: the candidate is the last state; the gate set was tracked through training.
    assert line["best_step"] == line["steps"] > cfg.eval_every
    assert [s for s, _ in line["gate_history"]][0] == 0 and line["gate_history"][-1][0] == line["steps"]
    assert np.isclose(line["gate_history"][0][1], line["gate_ce_current"])
    assert np.isclose(line["gate_history"][-1][1], line["gate_ce"])
    assert ("gate-set" in line["reason"]) or line["promoted"]


def test_changed_gate_settings_forget_earlier_failures(tmp_path, generated, init_checkpoint, fixed_dir, monkeypatch):
    cfg = _trainer_config(tmp_path, generated, init_checkpoint, max_steps=4, escalate_after=2)
    trainer = vt.ValueTrainer(cfg)
    monkeypatch.setattr(vt, "gate_decision", lambda *a, **k: (False, False, "rejected"))
    for _ in range(3):
        trainer.run_cycle()
    assert vt.ValueTrainer(cfg).state["consecutive_failures"] == 3  # the same settings: kept
    cfg.lr = 2e-3
    assert vt.ValueTrainer(cfg).state["consecutive_failures"] == 3  # another setting: kept
    cfg.gate_data = [str(fixed_dir)]
    again = vt.ValueTrainer(cfg)
    assert again.state["consecutive_failures"] == 0 and not again.state.get("review")
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["consecutive_failures"] == 0 and state["cycle_hours"] == cfg.cycle_hours and state["cycle"] == 3
    assert vt.ValueTrainer(cfg).state["consecutive_failures"] == 0
    for _ in range(2):
        again.run_cycle()
    assert vt.ValueTrainer(cfg).state["consecutive_failures"] == 2
    cfg.ema_decay = 0.999  # candidates made another way: their failures start afresh too
    assert vt.ValueTrainer(cfg).state["consecutive_failures"] == 0


def test_with_a_weight_average_the_candidate_and_the_evaluations_are_the_average(
    tmp_path, generated, init_checkpoint, fixed_dir, monkeypatch
):
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _load

    decay = 0.7
    # eval_every 1: the generated fixture plays for a fixed 8 s, so on a loaded machine a cycle may get only
    # 2 steps of data; evaluating every step still checks several evaluations of the average.
    cfg = _trainer_config(tmp_path, generated, init_checkpoint, fixed_dir, gate_data=[str(fixed_dir)], max_steps=12,
                          ema_decay=decay, diverge_ce=10.0, eval_every=1)  # (a tiny net at lr 1e-3: noise, not a test)
    trainer = vt.ValueTrainer(cfg)
    expected = {}  # the average, computed here from the raw weights after every step
    step = vt.train_step

    def recording_step(model, *args, **kwargs):
        if not expected:
            expected.update({k: p.detach().double().clone() for k, p in model.named_parameters()})
        out = step(model, *args, **kwargs)
        for k, p in model.named_parameters():
            expected[k] = decay * expected[k] + (1 - decay) * p.detach().double()
        recording_step.raw = {k: p.detach().clone() for k, p in model.named_parameters()}
        return out

    monkeypatch.setattr(vt, "train_step", recording_step)
    line = trainer.run_cycle()
    assert line["ema_decay"] == decay and line["best_step"] == line["steps"] > cfg.eval_every
    candidate = _load(tmp_path / "run" / line["candidate"])
    params = dict(candidate.named_parameters())
    assert set(params) == set(expected)
    for k, p in params.items():
        assert torch.allclose(p.detach().double(), expected[k], atol=1e-6), k
    assert any(not torch.allclose(params[k].detach(), recording_step.raw[k]) for k in params)  # not the raw weights
    # The gate set was tracked on the average: its last point is the candidate's gate CE.
    assert np.isclose(line["gate_history"][-1][1], line["gate_ce"])
    assert np.isclose(line["gate_history"][0][1], line["gate_ce_current"])


def test_an_interrupted_cycle_is_redone_after_a_restart(tmp_path, generated, init_checkpoint, monkeypatch):
    cfg = _trainer_config(tmp_path, generated, init_checkpoint)
    run = tmp_path / "run"
    trainer = vt.ValueTrainer(cfg)
    step = vt.train_step
    calls = []

    def sigterm_after_three(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            trainer._on_sigterm(15, None)  # as SIGTERM would
        return step(*args, **kwargs)

    monkeypatch.setattr(vt, "train_step", sigterm_after_three)
    assert trainer.serve(max_cycles=1) == 0
    assert not vt.read_jsonl(run / "cycles.jsonl") and not (run / "state.json").exists()
    monkeypatch.setattr(vt, "train_step", step)

    again = vt.ValueTrainer(cfg)  # systemd's restart, same arguments
    assert again.state["cycle"] == 0 and again.due()[0]
    assert again.serve(max_cycles=1) == 1
    lines = vt.read_jsonl(run / "cycles.jsonl")
    assert [x["cycle"] for x in lines] == [1]
    # A crash after the cycle's line but before state.json: the restart takes the line's state.
    vt.write_json_atomic(run / "state.json", {"cycle": 0, "data_cut": None, "consecutive_failures": 0})
    third = vt.ValueTrainer(cfg)
    assert third.state["cycle"] == 1 and third.state["data_cut"] == lines[0]["data_cut"]
    assert not third.due()[0]
    # A STOP file ends the trainer before any cycle.
    (run / "STOP").touch()
    assert third.serve() == 0
    assert json.loads((run / "status.json").read_text())["stage"] == "stopped"


def test_the_trigger_counts_new_games_since_the_cut(tmp_path, generated, init_checkpoint):
    cfg = _trainer_config(tmp_path, generated, init_checkpoint, data=[str(generated["data"]), str(generated["later"])],
                          min_new_games=10_000)
    trainer = vt.ValueTrainer(cfg)
    trainable = trainer.new_games()
    assert trainable == sum(s.trainable(cfg.held_out_per_mille) for s in trainer.shards.values()) > 0
    assert not trainer.due()[0]  # the first cycle needs min_new_games
    cfg.min_new_games = trainable
    assert trainer.due()[0]
    now = 1e9
    later = [s for s in trainer.shards.values() if str(generated["later"]) in str(s.path)]
    for s in trainer.shards.values():
        s.first_seen = now - 100 if s in later else now - 1000
    new = sum(s.trainable(cfg.held_out_per_mille) for s in later)
    trainer.state.update(data_cut=now - 500, last_cycle_started=now - 3600, consecutive_failures=0)
    cfg.cycle_games, cfg.min_new_games = new + 1, 1
    assert trainer.new_games() == new and not trainer.due(now)[0]  # neither enough games nor hours
    assert trainer.due(now + 12 * 3600)[0]  # hours passed, min_new_games arrived
    cfg.cycle_games = new
    assert trainer.due(now)[0]
    trainer.state["consecutive_failures"] = cfg.escalate_after  # escalated: twice the games and hours
    assert not trainer.due(now)[0] and not trainer.due(now + 12 * 3600)[0] and trainer.due(now + 24 * 3600)[0]
