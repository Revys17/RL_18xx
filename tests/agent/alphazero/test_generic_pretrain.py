"""Supervised pretraining of the generic model (``generic_pretrain``) and its policy-only player
(``generic_policy``), on CPU with the small config and 1830 fixture games.

Checked: the rank / top-1 helper; a two-stage run end to end (stage 1 trains and keeps the best joint
checkpoint, stage 2 changes only the value heads, every checkpoint loads with its metadata, history and
summary written); the evaluation's per-family and per-player-count metrics; the policy-only player
plays legal moves with prices inside their legal ranges; ``eval_policy_only.parse_player`` accepts a
``generic:`` player.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_data as gd  # noqa: E402
from rl18xx.agent.alphazero import generic_pretrain as gp  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "1830"
GAMES = ("router_d_trains.json", "train_rotation_game.json", "buy_train_at_limit.json")


def test_segment_rank():
    # two positions: rows 0-2 and 3-4
    logp = torch.tensor([-1.0, -0.5, -2.0, -0.7, -0.7])
    pos = torch.tensor([0, 0, 0, 1, 1])
    offsets = torch.tensor([0, 3, 5])
    assert gp.segment_rank(logp, pos, offsets, torch.tensor([1, 0])).tolist() == [0, 0]
    assert gp.segment_rank(logp, pos, offsets, torch.tensor([0, 1])).tolist() == [1, 1]  # tie: first wins
    assert gp.segment_rank(logp, pos, offsets, torch.tensor([2, 0])).tolist() == [2, 0]


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("generic_pretrain")
    (root / "games").mkdir()
    paths = []
    for name in GAMES:  # as cleaned-game files (with their title)
        game = json.loads((FIXTURES / name).read_text())
        game.setdefault("title", "1830")
        paths.append(root / "games" / name)
        paths[-1].write_text(json.dumps(game))
    out = root / "ds"
    gd.convert_dataset([str(p) for p in paths], str(out), workers=1, validation_percentage=0.5)
    ds = gd.GenericDataset(out)
    assert ds.game_ids("train") and ds.game_ids("validation")
    return out


@pytest.fixture(scope="module")
def run(dataset_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("generic_pretrain") / "run"
    torch.manual_seed(0)
    cfg = gp.GenericPretrainConfig(
        data=str(dataset_dir),
        out=str(out),
        model=dict(gp.SMALL_CONFIG),
        batch_size=32,
        lr=2e-3,
        epochs=2,
        max_steps=24,
        eval_every=12,
        patience=4,
        refit_eval_steps=5,
        refit_max_steps=10,
        refit_patience=2,
        device="cpu",
        bf16="off",
        workers=0,
        buffer_positions=256,
        log_every=6,
        threads=2,
        val_max_games=1,
    )
    summary = gp.pretrain(cfg)
    return cfg, summary


def test_two_stage_run(run):
    cfg, summary = run
    out = Path(cfg.out)
    for name in ("best_joint.pth", "last_joint.pth", "refit.pth", "final.pth", "history.jsonl", "summary.json"):
        assert (out / name).exists(), name
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    train = [h for h in history if h["kind"] == "train"]
    evals = [h for h in history if h["kind"] == "eval"]
    assert train and evals and any(h["kind"] == "refit_eval" for h in history)
    assert all(np.isfinite(h["total"]) for h in train)
    e = evals[-1]
    assert 0.0 <= e["top1"] <= 1.0 and e["top5"] >= e["top1"]
    assert e["top1_by_family"] and all(0 <= v["top1"] <= 1 for v in e["top1_by_family"].values())
    assert e["by_players"] and e["value_baseline"] > 0
    joint, joint_meta = mg.load_checkpoint(out / "best_joint.pth")
    refit, refit_meta = mg.load_checkpoint(out / "refit.pth")
    assert joint_meta["extra"]["stage"] == "joint" and refit_meta["extra"]["stage"] == "refit"
    assert joint_meta["titles"] == ["1830"]
    js, rs = joint.state_dict(), refit.state_dict()
    for key in js:
        same = torch.equal(js[key], rs[key])
        if key.startswith(("win_head", "score_head")):
            continue
        assert same, f"stage 2 changed {key}"
    assert any(not torch.equal(js[k], rs[k]) for k in js if k.startswith("win_head"))


def test_evaluate_matches_a_manual_pass(run):
    cfg, _ = run
    model, _ = mg.load_checkpoint(Path(cfg.out) / "final.pth")
    m = gp.evaluate(model, cfg.data, "validation", batch_size=16, max_games=1, cfg=cfg)
    ds = gd.GenericDataset(cfg.data)
    gid = ds.game_ids("validation")[0]
    items = list(gd.iter_items(ds.record(gid)))
    assert m["positions"] == len(items)
    b = gd.batch_to_device(gd.collate(items), "cpu")
    with torch.no_grad():
        out = model(b["inputs"])
    rank = gp.segment_rank(out["logp"], b["inputs"]["legal_pos"], b["inputs"]["legal_offsets"], b["target_index"])
    assert m["top1"] == pytest.approx(float((rank == 0).float().mean()), abs=1e-6)
    fam = sum(v["n"] for v in m["top1_by_family"].values())
    assert fam == len(items)


def test_policy_plays_legal_moves_with_legal_prices(run):
    from rl18xx.agent.alphazero.generic_policy import GenericPolicy
    from rl18xx.agent.alphazero.titles import new_game

    cfg, _ = run
    policy = GenericPolicy(str(Path(cfg.out) / "final.pth"))
    rng = np.random.default_rng(0)
    games = [new_game("1830", n) for n in (2, 4)]
    priced = 0
    for _ in range(150):
        live = [g for g in games if not g.finished]
        if not live:
            break
        legal = [[int(i) for i in g._game.factored_legal_indices()] for g in live]
        outputs = policy.evaluate(live, legal)
        for g, idx, o in zip(live, legal, outputs):
            assert o.probs.sum() == pytest.approx(1.0, abs=1e-5)
            index, price = policy.choose(g, o, 0.5, rng)
            assert index in idx
            pr = g._game.price_range_for_index(index)
            if pr is not None:
                assert price is not None and pr[0] <= price <= pr[1]
                priced += pr[0] < pr[1]
            g._game.apply_action_index(index, price)
    assert priced > 0  # the auction's bids came from the price head


def test_eval_policy_only_parses_generic_players(run):
    import sys

    cfg, _ = run
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from eval_policy_only import parse_player
    finally:
        sys.path.remove(str(scripts))
    path = str(Path(cfg.out) / "final.pth")
    spec, temperature, rules = parse_player(f"generic:{path}@0.5")
    assert spec == f"generic:{path}" and temperature == 0.5 and rules == ()


def test_bf16_training_steps(dataset_dir, tmp_path):
    """Stage 1 under bf16 autocast (CPU here; CUDA in production): finite losses, an fp32 evaluation."""
    cfg = gp.GenericPretrainConfig(
        data=str(dataset_dir),
        out=str(tmp_path / "bf16"),
        model=dict(gp.SMALL_CONFIG),
        batch_size=32,
        epochs=1,
        max_steps=6,
        value_refit=False,
        device="cpu",
        bf16="on",
        workers=0,
        buffer_positions=128,
        log_every=3,
        threads=2,
        val_max_games=1,
    )
    summary = gp.pretrain(cfg)
    assert summary["joint"]["steps"] == 6 and np.isfinite(summary["joint"]["best_val_total"])
    history = [json.loads(line) for line in (tmp_path / "bf16" / "history.jsonl").read_text().splitlines()]
    assert all(np.isfinite(h["total"]) for h in history if h["kind"] == "train")
