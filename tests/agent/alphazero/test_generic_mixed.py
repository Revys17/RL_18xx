"""Mixed-title supervised training of the generic model (``generic_data.MixedPositionBatches``,
``generic_pretrain`` with several datasets, ``--init`` fine-tuning; ``docs/rl_model.md`` "Mixed-title
training"), on CPU with the small config.

Data: the 1830 fixture games (``convert_dataset`` with the 1830 importer) and synthetic 1867 games --
random walks over 1867's layout, converted by a test importer that labels each logged action with the
index the walk played (the real 1867 importer reads human exports).

Checked: nested subsets and split overrides (a ``dev`` split, refusals); the mixed stream draws each
dataset at its share, carries each item's source, mixes titles inside a batch, cycles a small dataset and
never draws a zero-share one; a mixed run end to end (per-dataset evaluations, selection on the chosen
dataset, the value refit, the final per-dataset evaluation, checkpoints naming both titles); fine-tuning
from a checkpoint (``load_init``) and its refusals; the ``pretrain-generic`` command line.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_data as gd  # noqa: E402
from rl18xx.agent.alphazero import generic_pretrain as gp  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero.titles import new_game  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "1830"
GAMES_1830 = ("router_d_trains.json", "train_rotation_game.json", "buy_train_at_limit.json")
TITLE_ID = {t: i + 1 for i, t in enumerate(mg.vocab()["titles"])}


# ---------------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------------


def walk_1867(num_players: int, seed: int, steps: int) -> dict:
    """A random 1867 game over its layout, as a 'cleaned game' with the index played at each action."""
    rng = random.Random(seed)
    game = new_game("1867", num_players, seed=seed)
    indices = []
    for _ in range(steps):
        if game.finished:
            break
        moves = json.loads(game.rl_legal_moves_json())
        if not moves:
            break
        m = rng.choice(moves)
        price = None
        if m["range"]:
            lo, hi, step = m["range"]
            price = lo + step * rng.randint(0, min(3, (hi - lo) // step))
        game.rl_apply_index(m["index"], price)
        indices.append(int(m["index"]))
    return {
        "title": "1867",
        "seed": seed,
        "players": [{"id": i + 1, "name": f"Player {i + 1}"} for i in range(num_players)],
        "actions": [dict(a) for a in game.raw_actions],
        "indices": indices,
    }


class WalkImporter1867(gd.TitleImporter):
    """Labels a random walk's actions with the indices it played (test only)."""

    title = "1867"

    def load(self, game):
        self._indices = list(game["indices"])
        self._k = 0
        return game["actions"], None

    def seed(self, game):
        return game["seed"]

    def match(self, state, action, legal_indices):
        index = self._indices[self._k]
        self._k += 1
        return int(index), "ok"


@pytest.fixture(scope="module")
def data_1830(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed_1830")
    (root / "games").mkdir()
    paths = []
    for name in GAMES_1830:
        game = json.loads((FIXTURES / name).read_text())
        game.setdefault("title", "1830")
        paths.append(root / "games" / name)
        paths[-1].write_text(json.dumps(game))
    out = root / "ds"
    gd.convert_dataset([str(p) for p in paths], str(out), workers=1, validation_percentage=0.5)
    return out


@pytest.fixture(scope="module")
def data_1867(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed_1867")
    (root / "games").mkdir()
    paths = []
    for k, n in enumerate((3, 4, 5, 6, 4)):
        path = root / "games" / f"{900 + k}.json"
        path.write_text(json.dumps(walk_1867(n, seed=k + 1, steps=160)))
        paths.append(path)
    previous = gd._IMPORTERS.get("1867")  # the real 1867 importer (A3) reads human exports, not walks
    gd.register_importer("1867", WalkImporter1867)
    gd._WORKER_IMPORTER = None  # an in-process conversion caches its importer per title
    try:
        out = root / "ds"
        # finished_only=False: these are random-walk prefixes, not finished games (1867 converts finished games
        # only by default since A3's follow-up).
        gd.convert_dataset([str(p) for p in paths], str(out), title="1867", workers=1, validation_percentage=0.4,
                           finished_only=False)
    finally:
        if previous is None:
            gd._IMPORTERS.pop("1867", None)
        else:
            gd._IMPORTERS["1867"] = previous
        gd._WORKER_IMPORTER = None
    ds = gd.GenericDataset(out)
    assert ds.titles == ["1867"] and ds.num_positions() > 400
    assert ds.meta["stats"].get("unmatched") is None and ds.meta["stats"].get("not_legal") is None
    return out


def splits_1867(data_1867, tmp_path, dev: int = 1) -> Path:
    """train / dev / validation over the 1867 games: dev carved out of the converted training games."""
    ds = gd.GenericDataset(data_1867)
    train = ds.game_ids("train")
    assert len(train) > dev
    path = tmp_path / "splits_1867.json"
    path.write_text(json.dumps({"train": train[dev:], "dev": train[:dev], "validation": ds.game_ids("validation")}))
    return path


SMALL_RUN = dict(
    model=dict(gp.SMALL_CONFIG),
    batch_size=32,
    lr=2e-3,
    epochs=1,
    patience=4,
    refit_eval_steps=4,
    refit_max_steps=8,
    refit_patience=2,
    device="cpu",
    bf16="off",
    workers=0,
    buffer_positions=256,
    log_every=4,
    threads=2,
)


# ---------------------------------------------------------------------------------------------------
# Subsets and split overrides
# ---------------------------------------------------------------------------------------------------


def test_nested_subset():
    ids = [str(i) for i in range(50)]
    a, b = gd.nested_subset(ids, 10, seed=3), gd.nested_subset(ids, 25, seed=3)
    assert len(a) == 10 and b[:10] == a and len(set(b)) == 25
    assert gd.nested_subset(ids, 10, seed=3) == a  # stable
    assert gd.nested_subset(ids, 10, seed=4) != a
    assert sorted(gd.nested_subset(ids, None)) == sorted(ids)


def test_split_override(data_1867, tmp_path):
    path = splits_1867(data_1867, tmp_path)
    plain = gd.GenericDataset(data_1867)
    ds = gd.GenericDataset(data_1867, splits=path)
    spec = json.loads(path.read_text())
    assert ds.game_ids("dev") == spec["dev"] and ds.game_ids("train") == spec["train"]
    assert ds.game_ids("validation") == plain.game_ids("validation")
    dec = ds.decisions()
    assert ds.num_positions("dev") == sum(dec[g] for g in spec["dev"])
    assert ds.num_positions() == plain.num_positions()
    # a game in no split is in none
    partial = gd.GenericDataset(data_1867, splits={"train": spec["train"][:1]})
    assert partial.game_ids() == spec["train"][:1] and partial.game_ids("validation") == []
    with pytest.raises(gd.DatasetMismatch):
        gd.GenericDataset(data_1867, splits={"train": ["no-such-game"]})
    with pytest.raises(ValueError):
        gd.GenericDataset(data_1867, splits={"train": spec["dev"], "dev": spec["dev"]})
    # the loader reads exactly the override's games
    stream = gd.PositionBatches(data_1867, "dev", batch_size=64, shuffle=False, splits=path)
    assert sum(len(b["target_index"]) for b in stream) == ds.num_positions("dev")
    with pytest.raises(ValueError):
        gd.PositionBatches(data_1867, "test", batch_size=8, shuffle=False, splits=path)


# ---------------------------------------------------------------------------------------------------
# The mixed stream
# ---------------------------------------------------------------------------------------------------


def test_mixed_stream_follows_the_mix(data_1830, data_1867):
    sources = [
        gd.DataSource(str(data_1830), "1830", weight=0.75),
        gd.DataSource(str(data_1867), "1867", weight=0.25),
        gd.DataSource(str(data_1867), "unused", weight=0.0),
    ]
    stream = gd.MixedPositionBatches(sources, batch_size=48, buffer_positions=128, seed=5)
    picks, titles_per_batch = [], []
    n_1867 = gd.GenericDataset(data_1867).num_positions("train")
    it = iter(stream)
    for _ in range(40):  # 1,920 items: the 1867 share (~480) exceeds its training positions -> it cycles
        b = next(it)
        src = b["source"]
        title = b["inputs"]["meta"][:, 0]
        assert np.array_equal(title == TITLE_ID["1830"], src == 0)
        assert np.array_equal(title == TITLE_ID["1867"], src == 1)
        assert len(b["inputs"]["legal_offsets"]) == 49
        picks.append(src)
        titles_per_batch.append(len(set(title.tolist())))
    picks = np.concatenate(picks)
    share = float((picks == 1).mean())
    assert abs(share - 0.25) < 4 * np.sqrt(0.25 * 0.75 / len(picks))
    assert not (picks == 2).any()
    assert (picks == 1).sum() > 0.5 * n_1867 and max(titles_per_batch) == 2
    # the same seed gives the same stream
    again = next(iter(gd.MixedPositionBatches(sources, batch_size=48, buffer_positions=128, seed=5)))
    first = next(iter(gd.MixedPositionBatches(sources, batch_size=48, buffer_positions=128, seed=5)))
    assert np.array_equal(again["source"], first["source"])
    assert np.array_equal(again["target_index"], first["target_index"])
    with pytest.raises(ValueError):
        gd.MixedPositionBatches([gd.DataSource(str(data_1830), "a", weight=0.0)], batch_size=8)


def test_mixed_batch_forward_backward(data_1830, data_1867):
    torch.manual_seed(0)
    model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**gp.SMALL_CONFIG))
    sources = [gd.DataSource(str(data_1830), "1830", 0.5), gd.DataSource(str(data_1867), "1867", 0.5)]
    batch = next(iter(gd.MixedPositionBatches(sources, batch_size=40, buffer_positions=64, seed=1, config=model.config)))
    b = gd.batch_to_device(batch, "cpu")
    out = model(b["inputs"])
    off = b["inputs"]["legal_offsets"]
    probs = torch.zeros(len(off) - 1).index_add(0, b["inputs"]["legal_pos"], out["logp"].exp())
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)
    losses = gp.compute_losses(out, b, gp.GenericPretrainConfig(data="", out=""))
    losses["total"].backward()
    assert all(torch.isfinite(v) for v in losses.values())


# ---------------------------------------------------------------------------------------------------
# Sources and a mixed run
# ---------------------------------------------------------------------------------------------------


def test_resolve_sources(data_1830, data_1867, tmp_path):
    cfg = gp.GenericPretrainConfig(data=[str(data_1830), f"target={data_1867}"], out="", train_max_games={"target": 2})
    s = gp.resolve_sources(cfg)
    assert [x.label for x in s] == ["1830", "target"] and s[1].max_games == 2 and s[0].max_games is None
    ds = gd.GenericDataset(data_1867)
    dec = ds.decisions()
    assert s[1].weight == sum(dec[g] for g in ds.game_ids("train")[:2])  # default share: training positions
    cfg = gp.GenericPretrainConfig(data=[str(data_1830), str(data_1867)], out="", mix={"1830": 1, "1867": 3})
    assert [x.weight for x in gp.resolve_sources(cfg)] == [1.0, 3.0]
    bad = [
        dict(data=[str(data_1830), str(data_1830)]),  # duplicate labels
        dict(data=[str(data_1830), str(data_1867)], mix={"1830": 1.0}),  # a dataset without a share
        dict(data=[str(data_1830), str(data_1867)], mix={"1830": 1.0, "1867": 1.0, "1822": 1.0}),
        dict(data=[str(data_1830)], select_title="1867"),
        dict(data=[str(data_1830)], splits={"1867": "x.json"}),
        dict(data=[str(data_1830)], train_max_games={"1867": 3}),
        dict(data=[str(data_1830), str(data_1867)], mix={"1830": 0.0, "1867": 0.0}),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            gp.resolve_sources(gp.GenericPretrainConfig(out="", **kwargs))


@pytest.fixture(scope="module")
def mixed_run(data_1830, data_1867, tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed_run")
    splits = splits_1867(data_1867, root)
    cfg = gp.GenericPretrainConfig(
        data=[str(data_1830), str(data_1867)],
        out=str(root / "run"),
        mix={"1830": 0.5, "1867": 0.5},
        splits={"1867": str(splits)},
        select_title="1867",
        max_steps=16,
        eval_every=8,
        val_max_games={"1830": 1},
        final_eval=True,
        seed=3,
        **SMALL_RUN,
    )
    # 1830 has no dev split: evaluated on validation during training only through eval_titles
    with pytest.raises(ValueError, match="dev"):
        gp.pretrain(gp.GenericPretrainConfig(**{**cfg.__dict__, "select_split": "dev"}))
    torch.manual_seed(0)
    summary = gp.pretrain(cfg)
    return cfg, summary


def test_mixed_run(mixed_run):
    cfg, summary = mixed_run
    out = Path(cfg.out)
    for name in ("best_joint.pth", "last_joint.pth", "refit.pth", "final.pth", "summary.json", "config.json"):
        assert (out / name).exists(), name
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    train = [h for h in history if h["kind"] == "train"]
    assert train and all(set(h["by_source"]) == {"1830", "1867"} for h in train)
    assert abs(np.mean([h["by_source"]["1867"]["share"] for h in train]) - 0.5) < 0.2
    evals = [h for h in history if h["kind"] == "eval"]
    assert {h["source"] for h in evals} == {"1830", "1867"} and {h["step"] for h in evals} == {8, 16}
    # the best checkpoint is the minimum of 1867's validation loss without the value terms (the value refit is on)
    e67 = {h["step"]: h["policy_total"] for h in evals if h["source"] == "1867"}
    assert summary["joint"]["select_key"] == "policy_total"
    assert summary["joint"]["best_step"] == min(e67, key=e67.get)
    assert summary["joint"]["best_val"] == pytest.approx(min(e67.values()))
    assert any(h["kind"] == "select" for h in history)
    final = summary["final_eval"]
    assert set(final) == {"1830", "1867"}
    for label, m in final.items():
        assert m["positions"] > 0 and m["value_baseline"] > 0 and 0 <= m["top1"] <= 1 and m["top1_by_family"]
    n_val = gd.GenericDataset(cfg.data[1], splits=cfg.splits["1867"]).num_positions("validation")
    assert final["1867"]["positions"] == n_val
    _, meta = mg.load_checkpoint(out / "final.pth")
    assert meta["titles"] == ["1830", "1867"] and meta["extra"]["stage"] == "refit"
    assert meta["layouts"]["1867"]["fingerprint"] == mg.action_descriptors("1867").fingerprint
    config = json.loads((out / "config.json").read_text())
    assert config["loader"] == "stream" and config["selection"] == "1867" and len(config["sources"]) == 2


def test_fine_tune_from_a_checkpoint(mixed_run, data_1867, tmp_path):
    cfg, _ = mixed_run
    init = Path(cfg.out) / "best_joint.pth"
    splits = splits_1867(data_1867, tmp_path)
    ft = gp.GenericPretrainConfig(
        data=[str(data_1867)],
        out=str(tmp_path / "ft"),
        init=str(init),
        splits={"1867": str(splits)},
        select_split="dev",
        loader="stream",
        max_steps=8,
        eval_every=4,
        final_eval=True,
        **{**SMALL_RUN, "model": {}},
    )
    summary = gp.pretrain(ft)
    start, _ = mg.load_checkpoint(init)
    tuned, meta = mg.load_checkpoint(tmp_path / "ft" / "best_joint.pth")
    assert tuned.config == start.config
    assert meta["titles"] == ["1830", "1867"]  # the checkpoint's titles and the run's
    assert meta["extra"]["init"]["path"] == str(init) and meta["extra"]["init"]["stage"] == "joint"
    moved = [k for k, v in tuned.state_dict().items() if not torch.equal(v, start.state_dict()[k])]
    assert moved
    history = [json.loads(line) for line in (tmp_path / "ft" / "history.jsonl").read_text().splitlines()]
    assert {h["split"] for h in history if h["kind"] == "eval"} == {"dev"}
    assert set(summary["final_eval"]) == {"1867"} and summary["final_eval"]["1867"]["split"] == "validation"
    # the same architecture given explicitly is accepted
    same = gp.GenericPretrainConfig(**{**ft.__dict__, "out": str(tmp_path / "ft2"), "model": dict(gp.SMALL_CONFIG),
                                       "max_steps": 1, "eval_every": 1, "value_refit": False, "final_eval": False})
    gp.pretrain(same)


def test_init_refusals(mixed_run, data_1867, tmp_path):
    cfg, _ = mixed_run
    init = Path(cfg.out) / "best_joint.pth"
    datasets = {"1867": gd.GenericDataset(data_1867)}
    with pytest.raises(mg.CheckpointMismatch, match="architecture"):
        gp.load_init(init, {"d_model": 64}, datasets, "cpu")
    # a checkpoint without an embedding row for the dataset's title
    torch.manual_seed(0)
    narrow = mg.GenericPolicyValueModel(mg.GenericModelConfig(**gp.SMALL_CONFIG, num_titles=TITLE_ID["1867"]))
    mg.save_checkpoint(narrow, tmp_path / "narrow.pth", titles=["1830"])
    with pytest.raises(mg.CheckpointMismatch, match="beyond"):
        gp.load_init(tmp_path / "narrow.pth", None, datasets, "cpu")
    # a dataset converted on another layout than the checkpoint was trained on
    payload = torch.load(init, weights_only=True)
    model, _ = mg.load_checkpoint(init)
    other = gd.GenericDataset(data_1867)
    other.meta = json.loads(json.dumps(other.meta))
    other.meta["layouts"]["1867"]["fingerprint"] = "0000000000000000"
    with pytest.raises(mg.CheckpointMismatch, match="layout"):
        gp.load_init(init, None, {"1867": other}, "cpu")
    # B2's refusals: another encoder version
    payload["generic_meta"]["encoder_version"] = -1
    torch.save(payload, tmp_path / "old_encoder.pth")
    with pytest.raises(mg.CheckpointMismatch, match="encoder"):
        gp.load_init(tmp_path / "old_encoder.pth", None, datasets, "cpu")
    del model


def test_command_line(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    try:
        import main
    finally:
        sys.path.remove(str(root))
    seen = []
    monkeypatch.setattr(gp, "pretrain", lambda cfg: seen.append(cfg))
    args = main.build_parser().parse_args(
        [
            "pretrain-generic", "--data", "/d/1830", "--data", "target=/d/1867", "--out", str(tmp_path),
            "--mix", "1830:0.5,target:0.5", "--splits", "target:/s/t.json", "--select-title", "target",
            "--select-split", "dev", "--eval-titles", "target", "--train-max-games", "target:100",
            "--val-max-games", "40", "--final-eval", "--report-max-games", "1830:20", "--init", "/c/best.pth",
            "--model-preset", "small",
        ]
    )
    main.cmd_pretrain_generic(args)
    cfg = seen[-1]
    assert cfg.data == ["/d/1830", "target=/d/1867"] and cfg.mix == {"1830": 0.5, "target": 0.5}
    assert cfg.splits == {"target": "/s/t.json"} and cfg.select_title == "target" and cfg.select_split == "dev"
    assert cfg.eval_titles == ["target"] and cfg.train_max_games == {"target": 100} and cfg.val_max_games == 40
    assert cfg.final_eval and cfg.report_max_games == {"1830": 20} and cfg.init == "/c/best.pth"
    assert cfg.model == gp.SMALL_CONFIG
    args = main.build_parser().parse_args(["pretrain-generic", "--data", "/d/1830", "--out", str(tmp_path)])
    main.cmd_pretrain_generic(args)
    assert seen[-1].data == "/d/1830" and seen[-1].model == {} and seen[-1].loader == "auto"
    args = main.build_parser().parse_args(
        ["pretrain-generic", "--data", "/d/1830", "--out", str(tmp_path), "--init", "/c/x.pth"]
    )
    main.cmd_pretrain_generic(args)
    assert seen[-1].model == {}  # from a checkpoint: its architecture
