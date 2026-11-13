"""The weight-sharing probe (RL bridge milestone C3; ``docs/rl_model.md`` "The sharing probe"): does training on
another title (the *source*, 1830) as well help a small generic model on a *target* title (1867), against the
target alone?

Arms, per target training-set size and seed (all with the same small model and recipe, early-stopped on a dev
split carved out of the target's training games, the value heads refit, scored on the target's validation split):

* ``a``: the target alone;
* ``b``: joint -- target and source mixed in every batch at ``--joint-share`` of the positions from the target;
* ``c``: the source alone first (``pre``: one run per seed, shared by every size), then the target alone from its
  best checkpoint (``--init``).

Every arm's target phase runs at most ``--steps`` steps (equal compute for a, b and c's fine-tuning; arm b sees
``joint-share`` of a's target positions per step), evaluates on the dev split every ``--eval-positions`` target
positions (one pass over the size's training games if that is fewer, at least ``--min-eval-positions``) and stops
after ``--patience`` evaluations without improvement; arm c's source phase runs ``--pretrain-steps``.

    # plan (writes <out>/plan.json and the split files), then run, then report
    uv run python scripts/sharing_probe.py plan --target 1867=<1867 dataset> --source 1830=<1830 dataset> --out <dir>
    CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python scripts/sharing_probe.py run --out <dir> --jobs 6
    uv run python scripts/sharing_probe.py report --out <dir>

With the same dataset as target and source (``--target 1830=<d> --source 1830=<d>``), the plan carves disjoint game
sets: a pseudo-target of ``--pseudo-target-games`` training games (+ dev) and half the validation games, the source
the rest -- a positive control (the source is the target's own distribution, so it must help at small sizes).

``run`` is resumable: a cell is done when its ``summary.json`` exists; a failed or interrupted cell is started
again from scratch. ``report`` applies the decision rule written into ``docs/rl_model.md`` before the probe ran.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: The probe's model (~0.54M parameters): one map layer short of 1830's smallest useful depth, CPU-trainable at
#: ~300 positions/s per thread (docs/rl_model.md "The sharing probe").
PROBE_MODEL = dict(
    d_model=64,
    num_heads=4,
    hex_layers=2,
    entity_layers=1,
    fusion_layers=1,
    d_trunk=128,
    trunk_blocks=1,
    d_head=64,
    d_edge=16,
    d_cond=32,
    d_id=16,
)
ARMS = ("a", "b", "c")
FAMILIES_SHOWN = ("Pass", "Bid.company", "Par", "BuyShares", "SellShares", "LayTile", "PlaceToken", "BuyTrain.depot",
                  "Dividend", "RunRoutes", "Merge", "Convert")


def _dataset(path):
    from rl18xx.agent.alphazero import generic_data as gd

    return gd.GenericDataset(path)


def _parse_named(text: str) -> tuple:
    name, sep, path = text.partition("=")
    if not sep:
        raise SystemExit(f"expected TITLE=PATH, got {text!r}")
    return name, str(Path(path).resolve())


# ---------------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------------


def make_plan(args) -> dict:
    from rl18xx.agent.alphazero import generic_data as gd

    out = Path(args.out)
    if (out / "plan.json").exists() and not args.force:
        raise SystemExit(f"{out}/plan.json exists (--force to overwrite: only before any cell has run)")
    (out / "splits").mkdir(parents=True, exist_ok=True)
    t_name, t_path = _parse_named(args.target)
    s_name, s_path = _parse_named(args.source)
    target, source = _dataset(t_path), _dataset(s_path)
    t_dec, s_dec = target.decisions(), source.decisions()
    pseudo = t_path == s_path
    seed = args.subset_seed
    t_train = gd.nested_subset(target.game_ids("train"), None, seed=seed)
    t_val = target.game_ids("validation")
    dev = t_train[: args.dev_games]
    pool = t_train[args.dev_games:]
    s_train = source.game_ids("train")
    s_val = source.game_ids("validation")
    if pseudo:
        if not args.pseudo_target_games:
            raise SystemExit("the same dataset as target and source needs --pseudo-target-games")
        pool = pool[: args.pseudo_target_games]
        taken = set(dev) | set(pool)
        s_train = [g for g in s_train if g not in taken]
        val_order = gd.nested_subset(t_val, None, seed=seed + 1)
        half = len(val_order) // 2
        t_val, s_val = sorted(val_order[:half], key=t_val.index), sorted(val_order[half:], key=t_val.index)
    if args.target_val_games:
        keep = set(gd.nested_subset(t_val, args.target_val_games, seed=seed + 2))
        t_val = [g for g in t_val if g in keep]
    sizes = []
    for text in args.sizes.split(","):
        n = len(pool) if text == "all" else int(text)
        if n > len(pool):
            raise SystemExit(f"size {n} > the {len(pool)} target training games")
        if n not in sizes:
            sizes.append(n)
    files = {}
    for n in sizes:
        spec = {"train": pool[:n], "dev": dev, "validation": t_val}
        files[f"target_{n}"] = out / "splits" / f"target_{n}.json"
        files[f"target_{n}"].write_text(json.dumps(spec))
    files["source"] = out / "splits" / "source.json"
    files["source"].write_text(json.dumps({"train": s_train, "validation": s_val}))

    batch = args.batch_size
    share = args.joint_share
    model_json = json.dumps(PROBE_MODEL if args.model == "probe" else json.loads(args.model))
    common = [
        "--batch-size", str(batch), "--lr", str(args.lr), "--patience", str(args.patience),
        "--buffer-positions", str(args.buffer_positions), "--log-every", "100", "--refit-eval-steps",
        str(args.refit_eval_steps), "--refit-patience", str(args.refit_patience), "--refit-max-steps",
        str(args.refit_max_steps), "--device", "cpu", "--bf16", "off", "--final-eval", "--loader", "stream",
    ]
    target_data = ["--data", f"target={t_path}"]
    source_data = ["--data", f"source={s_path}", "--splits", f"source:{files['source']}"]
    cells = []
    runs = out / "runs"

    def eval_steps(n: int, per_step: float) -> int:
        positions = sum(t_dec[g] for g in pool[:n])
        every = min(max(positions, args.min_eval_positions), args.eval_positions)
        return max(1, int(round(every / per_step)))

    for seed_k in [int(s) for s in args.seeds.split(",")]:
        pre = f"c_pre_s{seed_k}"
        pre_steps = int(args.pretrain_steps)
        cells.append(
            {
                "name": pre, "arm": "c_pre", "size": None, "seed": seed_k, "depends": [],
                "args": [
                    *source_data, "--model-config", model_json, "--select-title", "source", "--select-split",
                    "validation", "--val-max-games", f"source:{args.source_val_games}", "--report-max-games",
                    f"source:{args.source_val_games}", "--no-value-refit", "--max-steps", str(pre_steps),
                    "--eval-every", str(max(1, pre_steps // args.pretrain_evals)), "--seed", str(seed_k), *common,
                ],
            }
        )
        for n in sizes:
            split = ["--splits", f"target:{files[f'target_{n}']}"]
            target_eval = ["--select-title", "target", "--select-split", "dev", "--eval-titles", "target"]
            steps_a = steps_b = int(args.steps)
            cells.append(
                {
                    "name": f"a_{n}_s{seed_k}", "arm": "a", "size": n, "seed": seed_k, "depends": [],
                    "args": [
                        *target_data, *split, *target_eval, "--model-config", model_json, "--max-steps", str(steps_a),
                        "--eval-every", str(eval_steps(n, batch)), "--seed", str(seed_k), *common,
                    ],
                }
            )
            cells.append(
                {
                    "name": f"b_{n}_s{seed_k}", "arm": "b", "size": n, "seed": seed_k, "depends": [],
                    "args": [
                        *target_data, *split, *source_data, *target_eval, "--mix", f"target:{share},source:{1 - share}",
                        "--report-max-games", f"source:{args.source_val_games}", "--model-config", model_json,
                        "--max-steps", str(steps_b), "--eval-every", str(eval_steps(n, batch * share)),
                        "--seed", str(seed_k), *common,
                    ],
                }
            )
            cells.append(
                {
                    "name": f"c_{n}_s{seed_k}", "arm": "c", "size": n, "seed": seed_k, "depends": [pre],
                    "args": [
                        *target_data, *split, *source_data, *target_eval, "--mix", "target:1,source:0",
                        "--report-max-games", f"source:{args.source_val_games}",
                        "--init", str(runs / pre / "best_joint.pth"), "--max-steps", str(steps_a),
                        "--eval-every", str(eval_steps(n, batch)), "--seed", str(seed_k), *common,
                    ],
                }
            )
    plan = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": {"title": t_name, "path": t_path, "titles": target.titles},
        "source": {"title": s_name, "path": s_path, "titles": source.titles},
        "pseudo_target": pseudo,
        "sizes": sizes,
        "seeds": [int(s) for s in args.seeds.split(",")],
        "settings": {k: v for k, v in vars(args).items() if k not in ("func", "force")},
        "model": json.loads(model_json),
        "counts": {
            "target_pool_games": len(pool),
            "target_dev_games": len(dev),
            "target_dev_positions": sum(t_dec[g] for g in dev),
            "target_validation_games": len(t_val),
            "target_validation_positions": sum(t_dec[g] for g in t_val),
            "target_train_positions": {str(n): sum(t_dec[g] for g in pool[:n]) for n in sizes},
            "source_train_games": len(s_train),
            "source_train_positions": sum(s_dec[g] for g in s_train),
            "source_validation_games": len(s_val),
        },
        "cells": cells,
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: v for k, v in plan.items() if k != "cells"}, indent=2))
    print(f"{len(cells)} cells -> {out / 'plan.json'}")
    return plan


# ---------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------


def _cell_dir(out: Path, cell: dict) -> Path:
    return out / "runs" / cell["name"]


def _done(out: Path, cell: dict) -> bool:
    return (_cell_dir(out, cell) / "summary.json").exists()


def _order(cells: list) -> list:
    """Source pretraining first (arm c waits on it), then the cells expected to run longest: larger target sizes
    first (small ones stop early), arm b before a before c at a size, then by seed."""

    def steps(c):
        a = c["args"]
        return int(a[a.index("--max-steps") + 1])

    rank = {"c_pre": 0, "b": 1, "a": 2, "c": 3}
    return sorted(
        cells, key=lambda c: (c["arm"] != "c_pre", -(c["size"] or 0), -steps(c), rank.get(c["arm"], 9), c["seed"])
    )


def run_plan(args) -> None:
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    cells = [c for c in plan["cells"] if (not args.arms or c["arm"].split("_")[0] in args.arms.split(","))]
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c["name"] in wanted]
    pending = [c for c in _order(cells) if not _done(out, c)]
    names = {c["name"] for c in plan["cells"]}
    running: dict = {}
    failed: dict = {}
    status_path = out / "status.json"
    started = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED="1")
    print(f"{len(pending)} cells to run ({len(cells) - len(pending)} done), {args.jobs} at a time", flush=True)

    def write_status():
        status = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - started, 1),
            "done": sorted(c["name"] for c in plan["cells"] if _done(out, c)),
            "running": sorted(running),
            "pending": [c["name"] for c in pending],
            "failed": failed,
        }
        status_path.write_text(json.dumps(status, indent=2))

    while pending or running:
        for name, (proc, cell, t0, fh) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            fh.close()
            del running[name]
            seconds = round(time.time() - t0, 1)
            if code == 0 and _done(out, cell):
                print(f"[{time.strftime('%H:%M:%S')}] done {name} in {seconds} s", flush=True)
            else:
                failed[name] = {"code": code, "seconds": seconds, "log": str(_cell_dir(out, cell) / "log.txt")}
                print(f"[{time.strftime('%H:%M:%S')}] FAILED {name} (exit {code}) after {seconds} s", flush=True)
        blocked = {c["name"] for c in pending if any(d in failed for d in c["depends"])}
        for c in [c for c in pending if c["name"] in blocked]:
            failed.setdefault(c["name"], {"code": None, "reason": "dependency failed"})
        pending = [c for c in pending if c["name"] not in blocked]
        while len(running) < args.jobs:
            ready = [
                c for c in pending
                if all(d not in names or _done(out, next(x for x in plan["cells"] if x["name"] == d)) for d in c["depends"])
            ]
            if not ready:
                break
            cell = ready[0]
            pending.remove(cell)
            cdir = _cell_dir(out, cell)
            if cdir.exists():
                shutil.rmtree(cdir)  # an interrupted or failed attempt: start again
            cdir.mkdir(parents=True)
            cmd = [
                sys.executable, str(REPO_ROOT / "main.py"), "pretrain-generic", "--out", str(cdir),
                *cell["args"], "--threads", str(args.threads), "--workers", str(args.workers),
            ]
            (cdir / "command.json").write_text(json.dumps(cmd))
            fh = open(cdir / "log.txt", "w")
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[cell["name"]] = (proc, cell, time.time(), fh)
            print(f"[{time.strftime('%H:%M:%S')}] start {cell['name']} (pid {proc.pid})", flush=True)
        write_status()
        if pending and not running:
            print("cells left whose dependencies are not done:", [c["name"] for c in pending], flush=True)
            break
        time.sleep(5)
    write_status()
    print(f"finished in {round(time.time() - started)} s; failed: {sorted(failed) or 'none'}", flush=True)


# ---------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _spread(xs):
    return (max(xs) - min(xs)) if len(xs) > 1 else float("nan")


def collect(out: Path, plan: dict) -> list:
    rows = []
    for cell in plan["cells"]:
        path = _cell_dir(out, cell) / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        label = "source" if cell["arm"] == "c_pre" else "target"
        final = (summary.get("final_eval") or {}).get(label)
        if final is None:
            continue
        joint = summary["joint"]
        row = {
            "name": cell["name"], "arm": cell["arm"], "size": cell["size"], "seed": cell["seed"],
            "policy_ce": final.get("policy_nll"), "top1": final.get("top1"), "top5": final.get("top5"),
            "chance_top1": final.get("chance_top1"), "value_ce": final.get("value_ce"),
            "value_baseline": final.get("value_baseline"), "winner_acc": final.get("winner_acc"),
            "price_nll": final.get("price_nll"), "positions": final.get("positions"),
            "families": {k: v["top1"] for k, v in (final.get("top1_by_family") or {}).items()},
            "family_n": {k: v["n"] for k, v in (final.get("top1_by_family") or {}).items()},
            "best_step": joint.get("best_step"), "steps": joint.get("steps"), "planned_steps": joint.get("planned_steps"),
            "early_stopped": joint.get("early_stopped"), "seconds": summary.get("seconds"),
        }
        # budget-bound: the best checkpoint is the last evaluation and the run did not stop early
        curve = joint.get("curve") or []
        row["budget_bound"] = bool(curve) and joint.get("best_step") == curve[-1]["step"] and not joint.get("early_stopped")
        source = (summary.get("final_eval") or {}).get("source")
        if source is not None and label == "target":
            row["source_policy_ce"] = source.get("policy_nll")
            row["source_top1"] = source.get("top1")
        rows.append(row)
    return rows


def decide(rows: list, plan: dict, factor: float = 2.0) -> dict:
    """The pre-set rule (docs/rl_model.md "The sharing probe: decision rule")."""
    sizes = plan["sizes"]
    full = max(sizes)
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["size"]), []).append(r)
    result = {"factor": factor, "per_size": {}, "warnings": []}
    for n in sizes:
        base = by.get(("a", n), [])
        entry = {}
        for arm in ("b", "c"):
            other = by.get((arm, n), [])
            if len(base) < 2 or len(other) < 2:
                entry[arm] = {"status": "insufficient seeds", "n_a": len(base), "n_arm": len(other)}
                continue
            ce_a, ce_x = [r["policy_ce"] for r in base], [r["policy_ce"] for r in other]
            v_a, v_x = [r["value_ce"] for r in base], [r["value_ce"] for r in other]
            spread = max(_spread(ce_a), _spread(ce_x))
            vspread = max(_spread(v_a), _spread(v_x))
            gain = _mean(ce_a) - _mean(ce_x)
            vloss = _mean(v_x) - _mean(v_a)
            policy_pass = gain > factor * spread
            value_ok = vloss <= factor * vspread
            entry[arm] = {
                "policy_gain": gain, "policy_spread": spread, "policy_pass": policy_pass,
                "value_loss": vloss, "value_spread": vspread, "value_ok": value_ok,
                "pass": policy_pass and value_ok,
                "worse": -gain > factor * spread,
                "budget_bound": [r["name"] for r in base + other if r.get("budget_bound")],
            }
        result["per_size"][str(n)] = entry
    full_entry = result["per_size"].get(str(full), {})
    passing_full = [arm for arm, e in full_entry.items() if e.get("pass")]
    passing_small = sorted(
        {arm for n in sizes if n != full for arm, e in result["per_size"][str(n)].items() if e.get("pass")}
    )
    worse_full = [arm for arm, e in full_entry.items() if e.get("worse")]
    if any(e.get("status") for e in full_entry.values()) or not full_entry:
        verdict = "incomplete: the full-size cells need at least 2 seeds per arm"
    elif passing_full:
        verdict = (
            f"SHARING SIGNAL at full size (arm{'s' if len(passing_full) > 1 else ''} {', '.join(passing_full)}): "
            "a candidate for 1867 -- confirm with the full-size supervised comparison (E1) on the GPU before "
            "changing the per-title default"
        )
    elif passing_small:
        verdict = (
            f"helps only below full size (arm{'s' if len(passing_small) > 1 else ''} {', '.join(passing_small)}): "
            "keep 1867 per-title; evidence for warm-starting data-poor titles from another title's model"
        )
    else:
        verdict = "NO SHARING SIGNAL: keep per-title models (the default)"
    a_bound = [r["name"] for r in by.get(("a", full), []) if r.get("budget_bound")]
    if passing_full and a_bound:
        verdict += f" (provisional on the budget: arm a still improving at the cap in {a_bound})"
    if worse_full:
        verdict += f"; arm{'s' if len(worse_full) > 1 else ''} {', '.join(worse_full)} WORSE than target-only at full size (interference)"
    bound = sorted({name for e in full_entry.values() for name in e.get("budget_bound", [])})
    if bound:
        result["warnings"].append(f"budget-bound at full size (best = last evaluation): {bound}")
    result["verdict"] = verdict
    return result


def _fmt(xs, digits=4):
    if not xs:
        return "-"
    m = _mean(xs)
    if len(xs) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ±{_spread(xs) / 2:.{digits}f}"


def report(args) -> dict:
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    rows = collect(out, plan)
    decision = decide(rows, plan, factor=args.factor)
    lines = [
        f"# Sharing probe: {plan['target']['title']} target, {plan['source']['title']} source"
        + (" (pseudo-target: one dataset split in two)" if plan["pseudo_target"] else ""),
        "",
        f"Plan {out / 'plan.json'} ({plan['created']}); target pool {plan['counts']['target_pool_games']} games, "
        f"dev {plan['counts']['target_dev_games']} games / {plan['counts']['target_dev_positions']} positions, "
        f"validation {plan['counts']['target_validation_games']} games / "
        f"{plan['counts']['target_validation_positions']} positions; source {plan['counts']['source_train_games']} "
        f"training games / {plan['counts']['source_train_positions']} positions.",
        "",
        "Target validation (mean ±half the seed range); policy CE = NLL of the human move over the legal moves; "
        "win CE vs equal odds; `steps` = mean best step / planned steps.",
        "",
        "| arm | size | seeds | policy CE | top-1 | top-5 | win CE (equal odds) | winner acc | best / planned steps | hours |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["size"]), []).append(r)
    for n in plan["sizes"]:
        for arm in ARMS:
            rs = by.get((arm, n), [])
            if not rs:
                continue
            lines.append(
                f"| {arm} | {n} | {len(rs)} | {_fmt([r['policy_ce'] for r in rs])} | {_fmt([r['top1'] for r in rs], 3)} | "
                f"{_fmt([r['top5'] for r in rs], 3)} | {_fmt([r['value_ce'] for r in rs])} "
                f"({_mean([r['value_baseline'] for r in rs]):.4f}) | {_fmt([r['winner_acc'] for r in rs], 3)} | "
                f"{_mean([r['best_step'] or 0 for r in rs]):.0f} / {_mean([r['planned_steps'] or 0 for r in rs]):.0f}"
                f"{' (bound)' if any(r['budget_bound'] for r in rs) else ''} | "
                f"{_mean([(r['seconds'] or 0) / 3600 for r in rs]):.2f} |"
            )
    lines += ["", "Top-1 by family (target validation, mean over seeds; n = positions):", ""]
    fams = [f for f in FAMILIES_SHOWN if any(f in r["families"] for r in rows if r["arm"] != "c_pre")]
    others = sorted({f for r in rows if r["arm"] != "c_pre" for f in r["families"]} - set(fams))
    fams += others
    if fams:
        n_of = {}
        for r in rows:
            if r["arm"] != "c_pre":
                n_of.update(r["family_n"])
        lines.append("| arm | size | " + " | ".join(f"{f} ({n_of.get(f, 0)})" for f in fams) + " |")
        lines.append("|---|---|" + "---|" * len(fams))
        for n in plan["sizes"]:
            for arm in ARMS:
                rs = by.get((arm, n), [])
                if rs:
                    cells = [
                        f"{_mean([r['families'][f] for r in rs if f in r['families']]):.3f}"
                        if any(f in r["families"] for r in rs) else "-"
                        for f in fams
                    ]
                    lines.append(f"| {arm} | {n} | " + " | ".join(cells) + " |")
    pre = by.get(("c_pre", None), [])
    joint_src = [r for r in rows if r["arm"] == "b" and r.get("source_policy_ce") is not None]
    ft_src = [r for r in rows if r["arm"] == "c" and r.get("source_policy_ce") is not None]
    if pre or joint_src:
        lines += ["", "Source validation (interference / retention):", "",
                  "| model | size | policy CE | top-1 |", "|---|---|---|---|"]
        if pre:
            lines.append(f"| c_pre (source only) | - | {_fmt([r['policy_ce'] for r in pre])} | {_fmt([r['top1'] for r in pre], 3)} |")
        for name, rs_all in (("b (joint)", joint_src), ("c (after fine-tuning)", ft_src)):
            for n in plan["sizes"]:
                rs = [r for r in rs_all if r["size"] == n]
                if rs:
                    lines.append(
                        f"| {name} | {n} | {_fmt([r['source_policy_ce'] for r in rs])} | {_fmt([r['source_top1'] for r in rs], 3)} |"
                    )
    lines += ["", f"Decision rule (factor {args.factor} x the larger seed range):", ""]
    for n, entry in decision["per_size"].items():
        for arm, e in entry.items():
            if "status" in e:
                lines.append(f"* size {n}, arm {arm}: {e['status']}")
                continue
            lines.append(
                f"* size {n}, arm {arm}: policy CE gain {e['policy_gain']:+.4f} vs spread {e['policy_spread']:.4f} "
                f"({'PASS' if e['policy_pass'] else 'no'}); win CE change {e['value_loss']:+.4f} vs spread "
                f"{e['value_spread']:.4f} ({'ok' if e['value_ok'] else 'LOSS'}) -> {'PASS' if e['pass'] else 'fail'}"
                + (" (worse than target-only)" if e["worse"] else "")
            )
    lines += ["", f"**Verdict: {decision['verdict']}**"]
    for w in decision["warnings"]:
        lines.append(f"\nWarning: {w}")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    (out / "report.json").write_text(json.dumps({"rows": rows, "decision": decision}, indent=2))
    print(text)
    return decision


def status(args) -> None:
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text())
    for cell in _order(plan["cells"]):
        cdir = _cell_dir(out, cell)
        state = "done" if _done(out, cell) else ("started" if cdir.exists() else "pending")
        extra = ""
        hist = cdir / "history.jsonl"
        if state == "started" and hist.exists():
            lines = hist.read_text().splitlines()
            last = json.loads(lines[-1]) if lines else {}
            extra = f" step {last.get('step')} ({last.get('kind')})"
        print(f"{cell['name']:>16} {state}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="Write <out>/plan.json and the split files")
    p.add_argument("--target", required=True, help="TITLE=PATH of the target dataset (convert-generic)")
    p.add_argument("--source", required=True, help="TITLE=PATH of the source dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", default="100,300,all", help="Target training-set sizes in games ('all' = the pool)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--dev-games", type=int, default=50, help="Target training games held out for early stopping")
    p.add_argument("--subset-seed", type=int, default=0, help="Order of the nested target subsets and the dev games")
    p.add_argument("--pseudo-target-games", type=int, default=None, help="Same dataset as both: the target's games")
    p.add_argument("--target-val-games", type=int, default=None, help="Cap on target validation games (default all)")
    p.add_argument("--steps", type=int, default=7812, help="Step cap of every arm's target phase (1M positions at 128)")
    p.add_argument("--pretrain-steps", type=int, default=3906, help="Arm c's source phase (= arm b's source share at the cap)")
    p.add_argument("--pretrain-evals", type=int, default=8, help="Evaluations during arm c's source phase")
    p.add_argument("--eval-positions", type=int, default=100_000, help="Target positions between evaluations")
    p.add_argument("--min-eval-positions", type=int, default=32_000)
    p.add_argument("--joint-share", type=float, default=0.5, help="Arm b: share of target positions in a batch")
    p.add_argument("--source-val-games", type=int, default=30, help="Source validation games for the interference check")
    p.add_argument("--model", default="probe", help="'probe' or a JSON GenericModelConfig")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--buffer-positions", type=int, default=16384)
    p.add_argument("--refit-eval-steps", type=int, default=400)
    p.add_argument("--refit-patience", type=int, default=2)
    p.add_argument("--refit-max-steps", type=int, default=1600)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=make_plan)
    p = sub.add_parser("run", help="Run the plan's pending cells")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=4, help="Cells at a time")
    p.add_argument("--threads", type=int, default=1, help="torch threads per cell")
    p.add_argument("--workers", type=int, default=1, help="Loader workers per cell")
    p.add_argument("--arms", default=None, help="Only these arms (a,b,c)")
    p.add_argument("--cells", default=None, help="Only these cells (names)")
    p.set_defaults(func=run_plan)
    p = sub.add_parser("report", help="Table per arm x size, and the decision rule's verdict")
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=float, default=2.0, help="A gain counts above factor x the seed range")
    p.set_defaults(func=report)
    p = sub.add_parser("status", help="Each cell's state")
    p.add_argument("--out", required=True)
    p.set_defaults(func=status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
