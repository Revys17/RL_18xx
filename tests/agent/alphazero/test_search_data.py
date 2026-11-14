"""Search learning's saved games (rl18xx/agent/alphazero/search_data.py, ``--save-data``):
the rows rebuilt from a saved game -- its positions replayed from the action log and
encoded, its targets and value inputs read back -- are the generator's in-memory rows
bit for bit, value targets follow the reader's settings, a crash mid-shard leaves the
earlier shards alone and recovers the journal, and a tiny CPU run keeps every game."""

import copy
import dataclasses
import hashlib
import json

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_data as sd  # noqa: E402
from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero.lockstep import ModelEvaluator  # noqa: E402

from .search_test_utils import HashNet  # noqa: E402


@pytest.fixture
def start_file(tmp_path):
    """An empty start file: every game takes a random auction ending (no human data needed)."""
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


def _config(start_file, **overrides) -> sl.SearchLearningConfig:
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="v", learner_temperature=0.5, prior_temperature=0.5,
        start_positions=start_file, random_start_fraction=1.0, n_full=6, p_full=0.5, games_per_worker=3,
        max_decisions=40, parallel_readouts=4, save_data="unused",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _generate(cfg, seconds=6.0, min_games=1):
    """Generator games (HashNet), with each finished game's search values and search records
    as the generator held them: runs of ``seconds`` until at least ``min_games`` games (at most 8 runs) --
    a fixed time alone plays too few games on a loaded machine (the pre-push gate beside other jobs)."""
    games, kept = [], {}
    for _ in range(8):
        gen = sl.Generator(sl.worker_settings(cfg, version=7), {"learner": ModelEvaluator(HashNet())})
        finish = gen._finish

        def keep(g, finish=finish):
            kept[g.uid] = (list(g.search_values), list(g.searches))
            return finish(g)

        gen._finish = keep
        games += gen.run(seconds)["games"]
        if len(games) >= min_games:
            break
    assert games and all("saved" in g for g in games)
    return games, kept


def _save(games, directory, cfg, shard_games=2):
    with sd.SearchDataWriter(directory, dataclasses.asdict(cfg), shard_games) as writer:
        for g in games:
            writer.add(g["saved"])
    return list(sd.iter_games(directory))


def assert_same_row(a: sl.SearchRow, b: sl.SearchRow) -> None:
    for name in sl.SearchRow.__dataclass_fields__:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray):
            assert isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape, name
            assert np.array_equal(x.view(np.uint8), y.view(np.uint8)), name  # bit for bit
        elif name == "price" and x is not None:
            assert y is not None and x[:3] == y[:3] and all(type(u) is int for u in y[:3])
            assert x[3].dtype == y[3].dtype and np.array_equal(x[3].view(np.uint8), y[3].view(np.uint8))
        else:
            assert type(x) is type(y) and x == y, (name, x, y)


SETTINGS = {
    # PUCT visits, fast searches with rows, lambda returns, exploration, auction starts (price rows).
    "visits": dict(
        target="visits", n_fast=2, fast_row_weight=0.3, value_lambda=0.9, full_move_temperature=1.0,
        explore_decisions=10, auction_fraction=0.5, critic_positions_per_game=3,
    ),
    # Gumbel root, n-step targets, the win reward (no score target), fast searches without rows.
    "gumbel": dict(target="gumbel", n_fast=2, value_nstep=3, reward="win", auction_fraction=0.3),
}


@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_saved_games_rebuild_the_generators_rows_bit_for_bit(tmp_path, start_file, name):
    cfg = _config(start_file, **SETTINGS[name])
    games, kept = _generate(cfg)
    saved = _save(games, tmp_path / "data", cfg)
    assert [s.uid for s in saved] == [g["uid"] for g in games]
    rows = 0
    for g, s in zip(games, saved):
        values, searches = kept[g["uid"]]
        c = s.columns
        # Every search (full and fast, with a row or not), as recorded.
        assert np.array_equal(c["search_value"], np.stack(values))
        assert c["search_decision"].tolist() == [x[0] for x in searches]
        assert c["search_action"].tolist() == [x[1] for x in searches]
        assert (c["search_flags"] & sd.FAST).astype(bool).tolist() == [x[2] for x in searches]
        assert set(c["search_version"].tolist()) == {7}
        assert s.meta["decisions"] == g["decisions"] and s.meta["termination"] == g["termination"]
        assert s.meta["win_share"] == g["win_share"] and s.meta["start_kind"] == g["start_kind"]
        assert len(s.actions) == s.meta["actions"]
        rebuilt = sd.rebuild_rows(s)
        assert len(rebuilt) == len(g["rows"])
        for a, b in zip(g["rows"], rebuilt):
            assert_same_row(a, b)
        rows += len(rebuilt)
    assert rows > 20
    if name == "visits":
        all_rows = [r for g in games for r in g["rows"]]
        assert any(r.fast for r in all_rows) and any(r.price is not None for r in all_rows)
        assert any(c for s in saved for c in (s.columns["search_flags"] & sd.EXPLORED))


def test_value_targets_follow_the_readers_settings(tmp_path, start_file):
    cfg = _config(start_file, n_fast=2)  # generated with the outcome as the win target
    games, kept = _generate(cfg)
    saved = _save(games, tmp_path / "data", cfg)
    for g, s in zip(games, saved):
        values, _ = kept[g["uid"]]
        win = sl.share_of_winners(g["rows"][0].fractions) if g["rows"] else None
        for row in sd.rebuild_rows(s, encode=False):
            assert np.array_equal(row.win_target, win)
            assert row.game_state is None and row.node_features is None
        returns = sl.lambda_returns(win, values, 0.8) if g["rows"] else None
        for row in sd.rebuild_rows(s, encode=False, value_lambda=0.8, value_bootstrap_weight=0.6):
            assert np.array_equal(row.win_target, sl.lambda_win_target(win, returns, row.search_index, 0.6))
        for row in sd.rebuild_rows(s, encode=False, value_nstep=4):
            assert np.array_equal(row.win_target, sl.nstep_win_target(win, values, row.search_index, 4, 0.5))
        points = sl.score_target(np.array(s.meta["net_worth"]), "placement:1,0.4,0.1,0")
        for row in sd.rebuild_rows(s, encode=False, reward="placement:1,0.4,0.1,0"):
            assert np.array_equal(row.score_target, points)
    with pytest.raises(TypeError):
        sd.rebuild_rows(saved[0], value_lamda=0.8)


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_crash_mid_shard_keeps_the_earlier_shards_and_recovers_the_journal(tmp_path, start_file):
    cfg = _config(start_file)
    games, _ = _generate(cfg, seconds=4.0, min_games=7)
    assert len(games) >= 7
    directory = tmp_path / "data"
    writer = sd.SearchDataWriter(directory, dataclasses.asdict(cfg), shard_games=2)
    for g in games[:5]:
        writer.add(g["saved"])
    writer._thread.join()
    shards = sd.shard_paths(directory)
    assert len(shards) == 2 and len(list(directory.glob("*.journal"))) == 1
    digests = {p.name: _digest(p) for p in shards}
    # A live writer's journal is locked: nobody else turns it into a shard.
    assert sd.recover_journals(directory) == 0
    # The crash: a torn record at the journal's end, and the process (its lock) gone.
    path, f = writer._journal
    f.write(b"\x40\x00\x00\x00\x00\x00\x00\x00partial")
    f.close()
    (directory / "ignored.sld.tmp").write_bytes(b"half a shard")
    assert [s.uid for s in sd.iter_games(directory)] == [g["uid"] for g in games[:4]]  # complete shards only
    again = sd.SearchDataWriter(directory, dataclasses.asdict(cfg), shard_games=2)  # recovers it
    assert not list(directory.glob("*.journal"))
    assert {p.name: _digest(p) for p in shards} == digests
    for g in games[5:7]:
        again.add(g["saved"])
    again.close()
    saved = list(sd.iter_games(directory))
    assert [s.uid for s in saved] == [g["uid"] for g in games[:7]]
    for g, s in zip(games, saved):
        for a, b in zip(g["rows"], sd.rebuild_rows(s)):
            assert_same_row(a, b)


def test_a_saved_game_is_a_game_record_the_viewer_reads(tmp_path, start_file):
    from rl18xx.agent.alphazero.game_records import list_games, load_game_file

    cfg = _config(start_file, auction_fraction=1.0)
    games, _ = _generate(cfg, seconds=3.0)
    _save(games, tmp_path / "data", cfg)
    assert sd.export_collection(tmp_path / "data", tmp_path / "collection", limit=2) == 2
    entries = list_games(tmp_path / "collection")
    assert len(entries) == 2 and all(e["auction_unlock"] for e in entries)
    loaded = load_game_file(tmp_path / "collection" / entries[0]["game_file"])
    assert loaded.auction_unlock is True and loaded.num_players == 4
    assert [a["id"] for a in loaded.actions] == list(range(1, len(loaded.actions) + 1))
    stats = sd.stats(tmp_path / "data")
    assert stats["games"] == len(games) and stats["rows"] == sum(len(g["rows"]) for g in games)
    assert 0 < stats["bytes_per_game"]


def test_a_tiny_cpu_run_saves_every_game_and_its_rows(tmp_path, start_file, monkeypatch):
    from .test_search_learning import _save_checkpoints

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    policy, value = _save_checkpoints(tmp_path)
    added = []
    window_add = sl.ReplayWindow.add

    def add(self, row):
        added.append(copy.copy(row))  # as it enters the window (training clears its critic flag)
        window_add(self, row)

    monkeypatch.setattr(sl.ReplayWindow, "add", add)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint=str(policy), value_checkpoint=str(value), out_dir=str(tmp_path / "runs"), run_name="tiny",
        p_full=0.5, n_full=4, n_fast=2, parallel_readouts=2, workers=2, games_per_worker=2, task_seconds=2.0,
        max_decisions=16, start_positions=start_file, random_start_fraction=1.0, inference_batch_size=64,
        window_rows=400, rows_per_update=10, min_window_rows=10, steps_per_update=2, minibatch=8, microbatch=4,
        max_updates=1, snapshot_every=1, anchor_every=0, kl_target=0.05, critic_positions_per_game=4,
        value_lambda=0.9, save_data=str(tmp_path / "data"), save_data_shard_games=3,
    )
    run_dir = sl.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    saved = list(sd.iter_games(tmp_path / "data"))
    assert not list((tmp_path / "data").glob("*.journal"))  # the last, partial shard was written at the end
    assert len(saved) >= history[-1]["games_total"] and history[-1]["saved_games"] >= history[-1]["games_total"]
    assert saved[0].config["run_name"] == "tiny" and saved[0].config["value_lambda"] == 0.9
    rebuilt = [row for game in saved for row in sd.rebuild_rows(game)]
    assert len(rebuilt) == len(added) > 10
    for a, b in zip(added, rebuilt):
        assert_same_row(a, b)

    # A resume keeps saving into the run's directory (its config says where).
    again = copy.copy(cfg)
    again.save_data, again.max_updates = None, 2
    sl.run(again, resume=str(run_dir))
    assert len(list(sd.iter_games(tmp_path / "data"))) > len(saved)
