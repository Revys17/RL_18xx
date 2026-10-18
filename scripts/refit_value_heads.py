"""Refit a policy network's own value heads to its critic (az2's shared network).

The trunk is frozen (the policy is unchanged); the win / score heads are fit on
positions of saved search games toward ``(1 - a) * critic + a * outcome``
(``rl18xx/agent/alphazero/value_refit.py``), ``a`` swept over
``--outcome-weights`` and chosen on a validation split of those games. Then
every value source is scored on the held-out collection by game stage (the az1
critic checks: every 25th action after the 20th) and the refit learner is saved
with a ``config.json`` carrying the source run's ``learner_temperature``.

Usage (from the repo root; the GPU makes the feature pass ~1 minute)::

    uv run python scripts/refit_value_heads.py \\
        --policy model_checkpoints_pg/rab_place_20261014/learner/1800.pth \\
        --critic model_checkpoints_pg/rab_place_20261014/critic/1800.pth \\
        --games logs/eval/az1_u25_sys64 logs/eval/search_c005_800_vs_200 ... \\
        --heldout logs/eval/az1_u50_sys64 \\
        --out model_checkpoints_pg/gen4_shared_20261018/learner/1800.pth
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl18xx.agent.alphazero import value_refit as vr  # noqa: E402


def concat(parts: list) -> vr.Extracted:
    preds = {}
    for name in parts[0].predictions:
        preds[name] = {k: np.concatenate([p.predictions[name][k] for p in parts]) for k in ("win", "score")}
    return vr.Extracted(
        seat_input=np.concatenate([p.seat_input for p in parts]),
        win=np.concatenate([p.win for p in parts]),
        placement=np.concatenate([p.placement for p in parts]),
        stage=np.concatenate([p.stage for p in parts]),
        games=sum((p.games for p in parts), []),
        predictions=preds,
    )


def subset(data: vr.Extracted, mask: np.ndarray) -> vr.Extracted:
    return vr.Extracted(
        seat_input=data.seat_input[mask],
        win=data.win[mask],
        placement=data.placement[mask],
        stage=data.stage[mask],
        games=[g for g, m in zip(data.games, mask) if m],
        predictions={n: {k: v[mask] for k, v in d.items()} for n, d in data.predictions.items()},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="The learner checkpoint whose value heads are refit")
    p.add_argument("--critic", required=True, help="The critic to distill")
    p.add_argument("--games", nargs="+", required=True, help="Eval collections (games/ + games.jsonl) to fit on")
    p.add_argument("--heldout", nargs="+", required=True, help="Eval collections to report on (never fit)")
    p.add_argument("--out", required=True, help="Where to save the refit learner (<run>/learner/<n>.pth)")
    p.add_argument("--positions-per-game", type=int, default=48)
    p.add_argument("--val-fraction", type=float, default=0.1, help="Games of --games held out to choose the fit")
    p.add_argument("--outcome-weights", type=str, default="0,0.25,0.5", help="Weights a of the outcome to try")
    p.add_argument("--inits", type=str, default="stale,reset", help="Head initializations to try")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256, help="Positions per feature forward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=str, default=None, help="JSON report path (default: next to --out)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch

    from rl18xx.agent.alphazero.policy_gradient import _load, use_tf32

    use_tf32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner, critic = _load(args.policy).to(device).eval(), _load(args.critic).to(device).eval()
    rng = np.random.default_rng(args.seed)
    started = time.time()

    def extract_collections(dirs, factory):
        parts = []
        for directory in dirs:
            positions = vr.collect([directory], factory, exclude=[])
            if positions:
                parts.append(vr.extract(positions, learner, {"critic": critic}, device, args.batch_size))
            logging.info(f"{directory}: {len(positions)} positions ({time.time() - started:.0f}s)")
        return concat(parts)

    heldout_names = {Path(h).name for h in args.heldout}
    fit_dirs = [d for d in args.games if Path(d).name not in heldout_names]
    data = extract_collections(fit_dirs, lambda gid, n: vr.sampled_selector(args.positions_per_game, n, rng))
    val_mask = np.array([vr.in_validation(g, args.val_fraction) for g in data.games])
    train, val = subset(data, ~val_mask), subset(data, val_mask)
    heldout = extract_collections(args.heldout, lambda gid, n: vr.heldout_selector)
    logging.info(f"fit {len(train.games)} / val {len(val.games)} / held-out {len(heldout.games)} positions")

    fits = []
    for init in [s for s in args.inits.split(",") if s]:
        for a in [float(x) for x in args.outcome_weights.split(",") if x]:
            cfg = vr.RefitConfig(outcome_weight=a, lr=args.lr, max_epochs=args.max_epochs, init=init, seed=args.seed)
            fit = vr.fit_heads(learner, train, val, "critic", cfg, device)
            val_pred = vr.head_predictions(fit["win_head"], fit["score_head"], val, device)
            val_report = vr.stage_report(val, val_pred)
            fits.append({"init": init, "outcome_weight": a, "fit": fit, "val": val_report,
                         "epochs": len(fit["history"]) - 1})
            logging.info(f"init {init} a {a}: val outcome CE {val_report['win_ce']:.4f} "
                         f"winner {val_report['winner_hit']:.3f} ({len(fit['history']) - 1} epochs)")
    best = min(fits, key=lambda f: f["val"]["win_ce"])

    reports = {
        "critic": vr.stage_report(heldout, heldout.predictions["critic"]),
        "learner (stale heads)": vr.stage_report(heldout, heldout.predictions["learner"]),
    }
    val_reports = {
        "critic": vr.stage_report(val, val.predictions["critic"]),
        "learner (stale heads)": vr.stage_report(val, val.predictions["learner"]),
    }
    for f in fits:
        name = f"refit {f['init']} a={f['outcome_weight']:g}"
        reports[name] = vr.stage_report(heldout, vr.head_predictions(f["fit"]["win_head"], f["fit"]["score_head"], heldout, device))
        val_reports[name] = f["val"]
    print(f"\nValidation ({len(val.games)} positions of {len(set(val.games))} games from --games):")
    for name, r in val_reports.items():
        print(vr.format_report(name, r))
    print(f"\nHeld out ({len(heldout.games)} positions of {len(set(heldout.games))} games: {', '.join(args.heldout)}):")
    for name, r in reports.items():
        print(vr.format_report(name, r))
    chosen = f"refit {best['init']} a={best['outcome_weight']:g}"
    print(f"\nChosen on validation: {chosen}")

    provenance = {
        "refit": {
            "critic": args.critic,
            "games": fit_dirs,
            "heldout": args.heldout,
            "positions_per_game": args.positions_per_game,
            "init": best["init"],
            "outcome_weight": best["outcome_weight"],
            "train_positions": len(train.games),
            "heldout_report": reports,
            "chosen": chosen,
        }
    }
    out = vr.write_refit_checkpoint(args.policy, best["fit"]["win_head"], best["fit"]["score_head"], args.out, provenance)
    report_path = Path(args.report) if args.report else Path(args.out).with_suffix(".refit.json")
    report_path.write_text(json.dumps({"heldout": reports, "validation": val_reports, "chosen": chosen,
                                       "histories": {f"{f['init']} a={f['outcome_weight']:g}": f["fit"]["history"] for f in fits}},
                                      indent=2))
    print(f"saved {out}; report {report_path} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
