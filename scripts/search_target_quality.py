"""Which search policy target carries real signal at the generator's budgets?

The search-learning generator trains the policy toward its PUCT search's root
(pruned visits by default). On positions of saved search games
(``eval_head_to_head`` collections), ``search`` runs the generator's search
configuration -- mean Q, c_puct 0.05, FPU 0, dedupe, 8 leaves a batch, priors at
the learner's temperature, the network's own value (a single shared checkpoint)
or a ``<policy>+<value>`` composite -- at each ``--config`` (``<c_puct>[:qnorm]@<readouts>``,
e.g. ``0.05@48``, ``0.625:qnorm@200``) on a fresh tree and stores each root's
statistics (``search_targets.root_stats`` + pruned visits + the move it would
play). ``score`` then computes every target variant from those roots and judges
it against a reference search (default ``0.05@800``: the same search with 4x / 16x
the budget, the better decision-maker), per budget:

``--reference-data`` takes the reference from another directory searched on the
same positions -- e.g. another network's search, whose value errors are not the
judged network's (the same network's deeper search shares its value's noise).

- ``H``: the target's entropy (nats); ``KL``: KL(target || policy prior);
- ``top1``: the target's argmax is the reference's move;
- ``regret``: Q_ref(reference move) - E_target[Q_ref] (Q_ref: the reference
  tree's mean value per root move for the mover, unvisited moves completed by its
  v_mix); ``gain``: E_target[Q_ref] - E_prior[Q_ref] (> 0: the target moves
  probability toward what more search prefers);
- ``low``: target mass on moves the policy gives < 1%; ``unref``: target mass on
  moves the reference never visited; ``spread``: mass on moves that are neither the
  policy's argmax nor the reference's move (above the prior's own: flattening
  pressure -- a policy trained toward the target spreads there);
- ``dis``: share of positions where the target's argmax is not the policy's;
  ``prec``: of those, the share where it is the reference's move; ``recall``: of
  the positions where the reference's move is not the policy's argmax, the share
  where the target's argmax is the reference's move; ``m_ref``: mean target mass on
  the reference's move there (the policy's own mass there: the ``prior`` row).

``pressure``: how a CE step toward each target moves the policy's mass off its
argmax, with and without KataGo's surprise weights (the flattening push).
``paired``: paired regret / top-1 differences between two targets.
``steps`` breaks the policy's and the targets' entropies down by step type for
searches by several checkpoints (where a training run's policy broadens).

``play`` rows judge each search's own move (its most visited, as the generator
plays it) the same way (top1, regret) -- the q-normalization question.

Usage::

    uv run python scripts/search_target_quality.py search --model <ckpt.pth> --out logs/target_quality/run1 \\
        --collection logs/eval/az4_u200_vs_az1_u320 --games 40 --per-game 5 --config 0.05@48 0.05@200 0.05@800
    uv run python scripts/search_target_quality.py score --data logs/target_quality/run1 --reference 0.05@800
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

LOGGER = logging.getLogger("search_target_quality")
NUM_PLAYERS = 4


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def parse_config(spec: str) -> dict:
    """``<c_puct>[:qnorm]@<readouts>`` -> ``{name, c_puct, qnorm, readouts}``."""
    head, _, readouts = spec.partition("@")
    c_puct, _, options = head.partition(":")
    flags = set(filter(None, options.split(",")))
    unknown = flags - {"qnorm"}
    if unknown or not readouts:
        raise SystemExit(f"bad --config {spec!r}: <c_puct>[:qnorm]@<readouts>")
    return {"name": spec, "c_puct": float(c_puct), "qnorm": "qnorm" in flags, "readouts": int(readouts)}


def generator_settings(config: dict, temperature: float) -> dict:
    """The search-learning generator's settings for ``config`` (az4/az5's search)."""
    from rl18xx.agent.alphazero import search_learning as sl

    cfg = sl.SearchLearningConfig(
        policy_checkpoint="",
        value_checkpoint="",
        learner_temperature=temperature,
        prior_temperature=temperature,
        target="visits",
        visit_target="pruned",
        n_full=config["readouts"],
        c_puct_init=config["c_puct"],
        mean_q=True,
        fpu_reduction=0.0,
        dedupe_leaves=True,
        parallel_readouts=8,
        q_normalize=config["qnorm"],
    )
    return sl.worker_settings(cfg, 0)


def root_record(tree) -> dict:
    from rl18xx.agent.alphazero.search_targets import root_stats

    stats = root_stats(tree)
    legal = stats["legal"]
    stats["pruned"] = np.asarray(tree.pruned_child_n_at_root(), dtype=np.float64)[legal]
    stats["best"] = int(tree.pick_best_action())
    return stats


def search(args) -> None:
    import torch

    from diag_sibling_values import sample_positions, step_type

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.composite_model import load_policy_value
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, TreeSearch
    from rl18xx.agent.alphazero.policy_gradient import use_tf32

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    positions = []
    for collection in args.collection:
        positions.extend(sample_positions(Path(collection), args.games, args.per_game, rng))
        LOGGER.info(f"{collection}: {len(positions)} positions so far")
    if args.limit:
        positions = positions[: args.limit]
    meta = []
    for game_id, i, n, rs, action, record in positions:
        legal = [int(x) for x in rs.factored_legal_indices()]
        stage = ["early", "mid", "late"][min(2, 3 * i // max(n, 1))]
        meta.append({"game": game_id, "action": i, "n_actions": n, "stage": stage,
                     "step": step_type(rs, legal), "legal": len(legal)})
    meta_path = out / "positions.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"{meta_path} holds other positions (another --seed / --collection?)")
    meta_path.write_text(json.dumps(meta))

    use_tf32()
    torch.set_num_threads(args.torch_threads)
    net = load_policy_value(args.model)
    temperature = args.temperature if args.temperature else sl.run_temperature(args.model.split("+")[0].split("#")[0])
    evaluator = ModelEvaluator(net)
    searcher = LockstepSearcher({"m": evaluator}, dedupe_maps=True)
    for spec in args.config:
        config = parse_config(spec)
        path = out / f"{spec.replace('/', '_')}.pkl"
        if path.exists() and not args.overwrite:
            LOGGER.info(f"{spec}: {path} exists, skipped")
            continue
        settings = generator_settings(config, temperature)
        gen = sl.Generator(settings, {"m": evaluator})
        chunk = args.chunk if config["readouts"] <= 400 else max(8, args.chunk * 400 // config["readouts"])
        records, t0, states0 = [], time.time(), evaluator.states
        for b in range(0, len(positions), chunk):
            part = positions[b : b + chunk]
            searches = [
                TreeSearch(gen._tree(rs), config["readouts"], key="m", budget="total") for _, _, _, rs, _, _ in part
            ]
            with torch.no_grad():
                searcher.run(searches)
            records.extend(root_record(s.tree) for s in searches)
            del searches
            LOGGER.info(f"{spec}: {len(records)} / {len(positions)} positions, {time.time() - t0:.0f}s, "
                        f"{(evaluator.states - states0) / max(len(records), 1):.1f} evaluations per position")
        with path.open("wb") as f:
            pickle.dump({"config": config, "temperature": temperature, "records": records,
                         "seconds": time.time() - t0, "evaluations": evaluator.states - states0}, f)
        LOGGER.info(f"{spec}: wrote {path}")


# ---------------------------------------------------------------------------
# Targets and scores
# ---------------------------------------------------------------------------


def target_variants() -> dict:
    """name -> f(stats) -> distribution over the legal moves."""
    from rl18xx.agent.alphazero import search_targets as st
    from rl18xx.agent.alphazero.search_learning import sharpen_target

    def norm(x):
        x = np.asarray(x, dtype=np.float64)
        return x / x.sum() if x.sum() > 0 else np.full(len(x), 1.0 / len(x))

    def onehot(s):
        out = np.zeros(len(s["legal"]))
        out[int(np.flatnonzero(s["legal"] == s["best"])[0])] = 1.0
        return out

    return {
        "prior": lambda s: norm(s["child_prior"]),
        "visits_raw": lambda s: norm(s["child_n"]),
        "visits_pruned": lambda s: norm(s["pruned"]) if s["pruned"].sum() > 0 else onehot(s),
        "visits_pruned^2": lambda s: sharpen_target(s["pruned"], s["legal"], s["best"], 0.5),
        "onehot": onehot,
        "cq": lambda s: st.completed_q_target(s, 50.0, 1.0),
        "cq_c0.3": lambda s: st.completed_q_target(s, 50.0, 0.3),
        "cq_c3": lambda s: st.completed_q_target(s, 50.0, 3.0),
        "cq_rescale_c0.1": lambda s: st.completed_q_target(s, 50.0, 0.1, rescale=True),
        "rpo": lambda s: st.rpo_target(s),
        "rpo_c0.5": lambda s: st.rpo_target(s, c=0.5),
        "rpo_c1.25": lambda s: st.rpo_target(s, c=1.25),
        "rpo_qnorm": lambda s: st.rpo_target(s, normalize=True),
        "rpo_qnorm_c1.25": lambda s: st.rpo_target(s, c=1.25, normalize=True),
        "rpo_qnorm_c2.5": lambda s: st.rpo_target(s, c=2.5, normalize=True),
    }


def kl(p, q) -> float:
    nz = p > 0
    return float((p[nz] * (np.log(p[nz]) - np.log(np.clip(q[nz], 1e-300, None)))).sum())


def entropy(p) -> float:
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def score_target(targets: list, priors: list, refs: list) -> dict:
    """Metrics of one target variant over the positions (see the module docs)."""
    rows = {k: [] for k in ("H", "KL", "top1", "regret", "gain", "low", "unref", "dis", "spread")}
    prec, recall, m_ref = [], [], []
    for t, p, r in zip(targets, priors, refs):
        q, best = r["q"], r["best_i"]
        rows["H"].append(entropy(t))
        rows["KL"].append(kl(t, p))
        top = int(np.argmax(t))
        rows["top1"].append(float(top == best))
        rows["regret"].append(float(q[best] - t @ q))
        rows["gain"].append(float(t @ q - p @ q))
        rows["low"].append(float(t[p < 0.01].sum()))
        rows["unref"].append(float(t[~r["visited"]].sum()))
        disagree = top != int(np.argmax(p))
        rows["dis"].append(float(disagree))
        # Mass on moves that are neither the policy's argmax nor the reference's move: what a
        # policy trained toward the target spreads (the flattening pressure, vs the prior's own).
        keep = {int(np.argmax(p)), best}
        rows["spread"].append(float(1.0 - sum(t[i] for i in keep)))
        if disagree:
            prec.append(float(top == best))
        if best != int(np.argmax(p)):
            recall.append(float(top == best))
            m_ref.append(float(t[best]))
    out = {}
    for k, v in rows.items():
        v = np.asarray(v)
        out[k] = float(v.mean())
        out[f"{k}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    out.update(prec=float(np.mean(prec)) if prec else float("nan"), n_dis=len(prec),
               recall=float(np.mean(recall)) if recall else float("nan"), n_override=len(recall),
               m_ref=float(np.mean(m_ref)) if m_ref else float("nan"), n=len(targets))
    return out


def reference_values(stats: dict) -> dict:
    q = stats["completed_q"].copy()
    best_i = int(np.flatnonzero(stats["legal"] == stats["best"])[0])
    return {"q": q, "best_i": best_i, "visited": stats["child_n"] > 0}


def score(args) -> None:
    data = Path(args.data)
    meta = json.loads((data / "positions.json").read_text())
    runs = {}
    for path in sorted(data.glob("*.pkl")):
        with path.open("rb") as f:
            run = pickle.load(f)
        runs[run["config"]["name"]] = run
    ref_dir = Path(args.reference_data) if args.reference_data else data
    if args.reference_data:
        if json.loads((ref_dir / "positions.json").read_text()) != meta:
            raise SystemExit(f"{ref_dir} was searched on other positions")
        with (ref_dir / f"{args.reference}.pkl").open("rb") as f:
            ref_records = pickle.load(f)["records"]
    elif args.reference in runs:
        ref_records = runs[args.reference]["records"]
    else:
        raise SystemExit(f"no reference run {args.reference!r} in {data} ({sorted(runs)})")
    refs = [reference_values(s) for s in ref_records]
    variants = target_variants()
    lines = []
    select = None
    if args.stage or args.step:
        select = [i for i, m in enumerate(meta) if (not args.stage or m["stage"] == args.stage)
                  and (not args.step or m["step"] == args.step)]

    def pick(xs):
        return xs if select is None else [xs[i] for i in select]

    for name, run in sorted(runs.items(), key=lambda kv: (kv[1]["config"]["qnorm"], kv[1]["config"]["c_puct"],
                                                          kv[1]["config"]["readouts"])):
        records = run["records"]
        if len(records) != len(ref_records):
            raise SystemExit(f"{name}: {len(records)} positions, reference {len(ref_records)}")
        for a, b in zip(records, ref_records):
            if not np.array_equal(a["legal"], b["legal"]):
                raise SystemExit(f"{name}: a position's legal moves differ from the reference's")
        priors = [variants["prior"](s) for s in records]
        evals = run["evaluations"] / max(len(records), 1)
        # The search's own move (q-normalization: does the search choose like more search?).
        play = [np.eye(len(s["legal"]))[int(np.flatnonzero(s["legal"] == s["best"])[0])] for s in records]
        line = {"config": name, "target": "play", "evals": round(evals, 1),
                **score_target(pick(play), pick(priors), pick(refs))}
        lines.append(line)
        if name == args.reference and not args.reference_data:
            continue
        kinds = args.targets or list(variants)
        for kind in kinds:
            targets = [variants[kind](s) for s in records]
            lines.append({"config": name, "target": kind, "evals": round(evals, 1),
                          **score_target(pick(targets), pick(priors), pick(refs))})
    out_path = data / (args.out or "scores.jsonl")
    with out_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    cols = ["H", "KL", "top1", "regret", "gain", "low", "spread", "dis", "prec", "recall", "m_ref"]
    print(f"positions {len(pick(meta))}, reference {args.reference} ({ref_dir})")
    print(f"{'config':<18} {'target':<18} {'evals':>6} " + " ".join(f"{c:>7}" for c in cols))
    for line in lines:
        vals = []
        for c in cols:
            v = line[c]
            vals.append(f"{v:7.4f}" if c in ("regret", "gain") else f"{v:7.3f}")
        print(f"{line['config']:<18} {line['target']:<18} {line['evals']:>6} " + " ".join(vals))
    print(f"(standard errors: regret ~{np.nanmean([l['regret_se'] for l in lines]):.4f}, "
          f"top1 ~{np.nanmean([l['top1_se'] for l in lines]):.3f}; written to {out_path})")


def steps(args) -> None:
    """Per step type (stock / track / token / route / dividend / train / other), for each
    ``--run <label>=<dir>/<config>`` (a search of the positions by some checkpoint): the
    policy's entropy (its tree priors), the targets' entropies, KL(target || policy) and
    the share of positions where the target's argmax is the policy's -- where a policy
    broadens, and whether its targets do."""
    variants = target_variants()
    kinds = args.targets or ["visits_pruned", "cq"]
    out = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        path = Path(path)
        meta = json.loads((path.parent / "positions.json").read_text())
        with path.open("rb") as f:
            records = pickle.load(f)["records"]
        groups = {}
        for m, s in zip(meta, records):
            groups.setdefault(m["step"], []).append(s)
            groups.setdefault("all", []).append(s)
        for step, rs in sorted(groups.items(), key=lambda kv: (kv[0] != "all", kv[0])):
            priors = [variants["prior"](s) for s in rs]
            line = {"run": label, "step": step, "n": len(rs), "H_policy": float(np.mean([entropy(p) for p in priors]))}
            for kind in kinds:
                ts = [variants[kind](s) for s in rs]
                line[f"H_{kind}"] = float(np.mean([entropy(t) for t in ts]))
                line[f"KL_{kind}"] = float(np.mean([kl(t, p) for t, p in zip(ts, priors)]))
                line[f"top1_{kind}"] = float(np.mean([np.argmax(t) == np.argmax(p) for t, p in zip(ts, priors)]))
            out.append(line)
    keys = [k for k in out[0] if k not in ("run", "step", "n")]
    print(f"{'run':<22} {'step':<9} {'n':>4} " + " ".join(f"{k:>18}" for k in keys))
    for line in out:
        print(f"{line['run']:<22} {line['step']:<9} {line['n']:>4} " + " ".join(f"{line[k]:18.3f}" for k in keys))
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(line) for line in out) + "\n")


def _surprise_weights(s: np.ndarray, a: float = 0.5, cap: float = 8.0) -> np.ndarray:
    """``search_learning.surprise_weights`` in numpy (KataGo's policy surprise weights)."""
    s = np.clip(s, 0, None)
    w = np.minimum((1 - a) + a * s / max(s.mean(), 1e-12), cap)
    return w * len(w) / w.sum()


def _load_targets(data: Path, config: str, kind: str) -> tuple:
    """``(targets, priors, records)`` of a target variant (or ``play``: the search's move) on a run."""
    variants = target_variants()
    with (data / f"{config}.pkl").open("rb") as f:
        records = pickle.load(f)["records"]
    priors = [variants["prior"](s) for s in records]
    if kind == "play":
        targets = [np.eye(len(s["legal"]))[int(np.flatnonzero(s["legal"] == s["best"])[0])] for s in records]
    else:
        targets = [variants[kind](s) for s in records]
    return targets, priors, records


def pressure(args) -> None:
    """How one CE step toward each target moves the policy's mass off its own argmax: the mean of
    (target off-argmax mass - policy off-argmax mass) over the positions, plain and with KataGo's
    surprise weights (a, cap as ``--surprise-weight`` / ``--surprise-cap``) -- > 0 broadens."""
    data = Path(args.data)
    print(f"{'config':<16} {'target':<18} {'off_t':>6} {'off_p':>6} {'push':>7} {'push_w':>7} {'H_t':>6} {'H_p':>6}")
    for config in args.config:
        for kind in args.targets or ["visits_pruned", "visits_pruned^2", "onehot", "cq", "cq_c3", "rpo"]:
            targets, priors, _ = _load_targets(data, config, kind)
            off_t = np.array([1 - t[int(np.argmax(p))] for t, p in zip(targets, priors)])
            off_p = np.array([1 - p[int(np.argmax(p))] for p in priors])
            w = _surprise_weights(np.array([kl(t, p) for t, p in zip(targets, priors)]), args.a, args.cap)
            print(f"{config:<16} {kind:<18} {off_t.mean():6.3f} {off_p.mean():6.3f} {(off_t - off_p).mean():7.3f} "
                  f"{(w * (off_t - off_p)).mean():7.3f} {np.mean([entropy(t) for t in targets]):6.3f} "
                  f"{np.mean([entropy(p) for p in priors]):6.3f}")


def paired(args) -> None:
    """Paired per-position differences of regret and top-1 between two ``<config>/<target>`` specs
    (``~``-separated pairs) against a reference run's pickle."""
    data = Path(args.data)
    with Path(args.reference).open("rb") as f:
        refs = [reference_values(s) for s in pickle.load(f)["records"]]

    def regrets(spec):
        config, kind = spec.rsplit("/", 1)
        targets, _, _ = _load_targets(data, config, kind)
        r = np.array([float(ref["q"][ref["best_i"]] - t @ ref["q"]) for t, ref in zip(targets, refs)])
        top = np.array([float(int(np.argmax(t)) == ref["best_i"]) for t, ref in zip(targets, refs)])
        return r, top

    for pair in args.pair:
        a, b = pair.split("~")
        (ra, ta), (rb, tb) = regrets(a), regrets(b)
        d, dt = ra - rb, ta - tb
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x))  # noqa: E731
        print(f"{a} - {b}: regret {1e3 * ra.mean():.2f} - {1e3 * rb.mean():.2f} = {1e3 * d.mean():+.2f} "
              f"+- {1e3 * se(d):.2f} (x1e-3); top-1 {dt.mean():+.4f} +- {se(dt):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search")
    p.add_argument("--model", required=True, help="a shared checkpoint (its own value) or <policy>+<value>")
    p.add_argument("--collection", action="append", required=True)
    p.add_argument("--games", type=int, default=40, help="games per collection")
    p.add_argument("--per-game", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--config", nargs="+", required=True, help="<c_puct>[:qnorm]@<readouts>")
    p.add_argument("--temperature", type=float, default=0.0, help="priors' temperature (0: the checkpoint run's)")
    p.add_argument("--chunk", type=int, default=64, help="trees searched in lockstep")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--data", required=True)
    p.add_argument("--reference", default="0.05@800")
    p.add_argument("--reference-data", default=None,
                   help="the reference run's directory, if another (another network's search of the same positions)")
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--stage", choices=("early", "mid", "late"), default=None)
    p.add_argument("--step", default=None)
    p.add_argument("--out", default=None)
    p = sub.add_parser("pressure")
    p.add_argument("--data", required=True)
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--a", type=float, default=0.5, help="surprise weight")
    p.add_argument("--cap", type=float, default=8.0)
    p = sub.add_parser("paired")
    p.add_argument("--data", required=True)
    p.add_argument("--reference", required=True, help="the reference run's .pkl")
    p.add_argument("--pair", nargs="+", required=True, help="<config>/<target>~<config>/<target>")
    p = sub.add_parser("steps")
    p.add_argument("--run", action="append", required=True, help="<label>=<data dir>/<config>.pkl")
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"search": search, "score": score, "steps": steps, "pressure": pressure, "paired": paired}[args.cmd](args)


if __name__ == "__main__":
    main()
