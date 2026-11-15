"""Start files read lazily (``start_positions.index_start_file`` / ``LazyActions``): the loaded lists keep their order
and length, so a seed draws exactly the starts the parsed (eager) files gave -- human SR1 starts, mid-game cuts and
practice cuts, with and without A2's picks -- and the starts a sampler returns hold tuples, as before."""

from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_play as gp  # noqa: E402
from rl18xx.agent.alphazero import start_positions as sp  # noqa: E402


def _actions(seed: int, n: int) -> list:
    rng = random.Random(seed)
    return [{"type": "bid", "entity": rng.randint(1, 4), "entity_type": "player", "company": f"C{i}",
             "price": rng.randrange(20, 400, 5)} for i in range(n)]  # fmt: skip


def _write(path: Path, records) -> str:
    path.write_text("".join(json.dumps(r) + "\n" + ("\n" if i == 1 else "") for i, r in enumerate(records)))
    return str(path)  # (a blank line after the second record: skipped, as before)


@pytest.fixture
def files(tmp_path):
    """Synthetic 1830 files (the samplers never replay them): SR1 starts, mid-game and practice cuts for 3 and 4
    players, and a 1867-style set (title, seed) for the generic sampler."""
    starts, mids, practice, t1867_starts, t1867_mids = [], [], [], [], []
    for i in range(12):
        n = 4 if i % 3 else 3
        starts.append({"id": f"s{i}", "num_players": n, "actions": _actions(i, 5 + i)})
        acts = _actions(100 + i, 60 + 3 * i)
        mids.append({"id": f"m{i}", "num_players": n, "actions": acts,
                     "cuts": {str(k): 10 * (k - 1) + i for k in range(2, 2 + 2 + i % 4)}})  # fmt: skip
        practice.append({"id": f"p{i}", "num_players": n, "actions": acts[:40], "cuts": {"4": 20 + i, "5": 30 + i},
                         "source": "human", "dumps": [{"sr": 6}]})  # fmt: skip
        t1867_starts.append(
            {"id": f"t{i}", "title": "1867", "num_players": n, "seed": 7 * i, "actions": _actions(i, 4)}
        )
        t1867_mids.append({"id": f"u{i}", "title": "1867", "num_players": n, "seed": 3 * i, "actions": acts,
                           "cuts": {"2": 11, "3": 22 + i}})  # fmt: skip
    return {
        "starts": _write(tmp_path / "starts_4p.jsonl", starts),
        "mids": _write(tmp_path / "mids_4p.jsonl", mids),
        "practice": _write(tmp_path / "practice_4p.jsonl", practice),
        "t_starts": _write(tmp_path / "t_starts.jsonl", t1867_starts),
        "t_mids": _write(tmp_path / "t_mids.jsonl", t1867_mids),
    }


# the parsed (eager) loaders, as before 2026-11-15
def _eager_start_positions(path) -> dict:
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out.setdefault(int(r["num_players"]), []).append(sp.StartPosition(f"human:{r['id']}", tuple(r["actions"])))
    return out


def _eager_midgame_positions(path) -> dict:
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            cuts = {int(k): int(v) for k, v in r["cuts"].items()}
            out.setdefault(int(r["num_players"]), []).append((str(r["id"]), tuple(r["actions"]), cuts))
    return out


def _eager_generic(kind):
    def load(path, title):
        out = {}
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rules = gp._record_rules(r, title)
                if kind == "start":
                    item = gp.GenericStart(f"human:{r['id']}", tuple(r["actions"]), r.get("seed"), rules)
                else:
                    cuts = {int(k): int(v) for k, v in r["cuts"].items()}
                    item = (str(r["id"]), tuple(r["actions"]), cuts, r.get("seed"), rules)
                out.setdefault(int(r["num_players"]), []).append(item)
        return out

    return load


def _picks(starts, mids, practice, n: int, seed: int) -> dict:
    """A2-style picks per kind (a label or None), drawn from the files' libraries."""
    lib = gp.start_labels(starts, mids, n, practice)
    rng = np.random.default_rng(seed)
    return {kind: [None if rng.random() < 0.3 else labels[int(rng.integers(len(labels)))] for _ in range(40)]
            for kind, labels in lib.items() if labels}  # fmt: skip


def _legacy_draws(files, seed: int, with_picks: bool) -> tuple:
    out, counts = [], []
    for n in (3, 4):
        picks = _picks(files["starts"], files["mids"], files["practice"], n, seed)
        picker = gp.StartPicker(picks) if with_picks else None
        rng = random.Random(seed)
        for _ in range(30):
            out.append(sp.sample_start_position(
                n, files["starts"], 0.1, rng=rng, midgame_path=files["mids"], midgame_fraction=0.4,
                auction_fraction=0.1, picker=picker, practice_path=files["practice"],
                practice_fraction=0.25))  # fmt: skip
        counts.append(dict(picker.counts) if picker else None)
    return out, counts


def _generic_draws(files, seed: int, with_picks: bool) -> list:
    out = []
    for n in (3, 4):
        picks = _picks(files["t_starts"], files["t_mids"], files["t_mids"], n, seed)
        picker = gp.StartPicker(picks) if with_picks else None
        rng = random.Random(seed)
        for _ in range(30):
            out.append(gp.sample_start("1867", n, files["t_starts"], 0.0, rng=rng, midgame_path=files["t_mids"],
                                       midgame_fraction=0.4, auction_fraction=0.1, picker=picker,
                                       practice_path=files["t_mids"], practice_fraction=0.2))  # fmt: skip
    return out


@pytest.mark.parametrize("with_picks", [False, True])
def test_lazy_files_draw_exactly_the_eager_starts(files, monkeypatch, with_picks):
    sp._LOADED.clear(), sp._LOADED_MIDGAME.clear(), gp._LOADED.clear()
    for seed in range(4):
        lazy, lazy_counts = _legacy_draws(files, seed, with_picks)
        lazy_generic = _generic_draws(files, seed, with_picks)
        with monkeypatch.context() as m:
            m.setattr(sp, "load_start_positions", _eager_start_positions)
            m.setattr(sp, "load_midgame_positions", _eager_midgame_positions)
            m.setattr(gp, "load_start_file", _eager_generic("start"))
            m.setattr(gp, "load_midgame_file", _eager_generic("midgame"))
            eager, eager_counts = _legacy_draws(files, seed, with_picks)
            eager_generic = _generic_draws(files, seed, with_picks)
        assert [s.label for s in lazy] == [s.label for s in eager]
        assert lazy == eager and lazy_generic == eager_generic and lazy_counts == eager_counts
        assert all(type(s.actions) is tuple for s in lazy + lazy_generic)  # the returned starts hold tuples
        kinds = {s.label.split(":")[0] for s in lazy}
        assert {"human", "midgame", "practice", "auction"} <= kinds
        assert {s.label.split(":")[0] for s in lazy_generic} >= {"human", "midgame", "practice"}
    if with_picks:
        assert any(v for c in lazy_counts for k, v in c.items() if k.endswith("_prioritized"))


def test_loaded_lists_hold_lazy_actions(files):
    sp._LOADED_MIDGAME.clear()
    games = sp.load_midgame_positions(files["mids"])
    assert sp.load_midgame_positions(files["mids"]) is games  # indexed once per process
    eager = _eager_midgame_positions(files["mids"])
    assert sorted(games) == [3, 4] and [len(v) for _, v in sorted(games.items())] == [4, 8]
    for n in games:
        for (gid, lazy, cuts), (gid2, actions, cuts2) in zip(games[n], eager[n]):
            assert isinstance(lazy, sp.LazyActions) and (gid, cuts) == (gid2, cuts2)
            assert len(lazy) == len(actions) and lazy == actions and actions == lazy
            k = max(cuts)
            assert lazy[: cuts[k]] == actions[: cuts[k]] and type(lazy[: cuts[k]]) is tuple
            assert lazy[3] == actions[3] and list(lazy) == list(actions)
            assert pickle.loads(pickle.dumps(lazy)) == actions  # a worker reopens the file by path
    starts = sp.load_start_positions(files["starts"])
    assert sp.materialized(starts[4][0]) == _eager_start_positions(files["starts"])[4][0]


def test_a_replaced_file_keeps_its_contents_and_a_rewritten_one_is_refused(files, tmp_path):
    sp._LOADED.clear()
    starts = sp.load_start_positions(files["starts"])
    first = starts[4][0]
    before = tuple(first.actions)
    other = tmp_path / "new.jsonl"
    other.write_text(json.dumps({"id": "zz", "num_players": 4, "actions": _actions(9, 50)}) + "\n")
    os.replace(other, files["starts"])  # the builders' atomic rename: the process reads the file it indexed
    assert tuple(first.actions) == before
    sp._LOADED.clear()
    fresh = sp.load_start_positions(files["starts"])
    assert [s.label for s in fresh[4]] == ["human:zz"]
    Path(files["starts"]).write_text(json.dumps({"id": "yy", "num_players": 4, "actions": []}) + "\n")  # in place
    with pytest.raises(ValueError, match="changed under this process"):
        tuple(fresh[4][0].actions)


def test_real_start_files_draw_the_same_with_lazy_loading_if_present(monkeypatch):
    """With the repository's human_games files (skipped without them): 4-player draws from the v2 SR1 starts and
    mid-game cuts, eager and lazy, give the same labels and actions."""
    starts, mids = sp.default_start_positions(4), sp.default_midgame_positions(4)
    if not (Path(starts).is_file() and Path(mids).is_file()):
        pytest.skip("no human_games start files here")
    sp._LOADED.clear(), sp._LOADED_MIDGAME.clear()

    def draws():
        rng = random.Random(11)
        return [sp.sample_start_position(4, starts, 0.0, rng=rng, midgame_path=mids, midgame_fraction=0.5)
                for _ in range(40)]  # fmt: skip

    lazy = draws()
    with monkeypatch.context() as m:
        m.setattr(sp, "load_start_positions", _eager_start_positions)
        m.setattr(sp, "load_midgame_positions", _eager_midgame_positions)
        eager = draws()
    assert lazy == eager
