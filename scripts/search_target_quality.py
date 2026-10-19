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
        positions.extend(sample_positions(Path(collection), args.games, args.per_game, rng, args.game_offset))
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


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties half (Mann-Whitney)."""
    from scipy.stats import rankdata

    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    pos, neg = labels.sum(), (~labels).sum()
    if not pos or not neg:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


SIGNALS = ("z", "gap", "share", "t_mass", "log_ratio", "nontrack")


def _passes(row: dict, rule: str) -> bool:
    """``rule``: comma-separated ``<signal>=<min>`` conditions, all required (``all``: none)."""
    if rule == "all":
        return True
    for cond in rule.split(","):
        sig, _, value = cond.partition("=")
        if row[sig] < float(value):
            return False
    return True


def confidence(args) -> None:
    """The confidence filter (``search_targets.confident_target``) offline: on each run
    (``--config``) and target kind, the departures (target argmax != the policy's) with
    their confidence z and simpler signals -- the q gap, ``share`` (the target move's
    share of the two moves' visits), ``t_mass`` (the target's mass on its move),
    ``log_ratio`` (log prior ratio of the two moves), ``nontrack`` (1 outside track
    lays). Per rule (``--thresholds``: ``<signal>=<min>[,...]``, all required), the
    departures kept and, against each reference: precision (the kept departure is the
    reference's move), recall (of the positions where the reference's move is not the
    policy's, the share a kept departure finds), ``better`` (the reference values the
    kept departure's move above the policy's), and the filtered target's regret / its
    change from the unfiltered target (> 0: filtering lost value by that reference) /
    push off the policy's argmax. ``pair``: with a ``pairwise`` file
    (``pairwise.<config>.<target>.<--pairwise>.json``), each departure's two moves
    searched as roots by an independent network -- the kept share it values above the
    policy's move and the share of the total pairwise gain (sum over departures of
    q(target move) - q(policy move)) the kept departures carry. ``--reference`` /
    ``--reference-data`` as ``score``; ``--self-reference`` names the same network's
    deeper run in ``--data``."""
    from rl18xx.agent.alphazero import search_targets as st

    data = Path(args.data)
    meta = json.loads((data / "positions.json").read_text())
    refs = {}
    with (data / f"{args.self_reference}.pkl").open("rb") as f:
        refs["self"] = [reference_values(s) for s in pickle.load(f)["records"]]
    if args.reference_data:
        ref_dir = Path(args.reference_data)
        if json.loads((ref_dir / "positions.json").read_text()) != meta:
            raise SystemExit(f"{ref_dir} was searched on other positions")
        with (ref_dir / f"{args.reference}.pkl").open("rb") as f:
            refs["az1"] = [reference_values(s) for s in pickle.load(f)["records"]]
    variants = target_variants()
    lines = []
    select = list(range(len(meta)))
    for config in args.config:
        with (data / f"{config}.pkl").open("rb") as f:
            records = pickle.load(f)["records"]
        if "child_q_var" not in records[0]:
            raise SystemExit(f"{config}: no child_q_var (searched by an older engine build)")
        select = [i for i, m in enumerate(meta) if not args.step or m["step"] == args.step]
        for kind in args.targets:
            pair_path = data / f"pairwise.{config}.{kind}.{args.pairwise}.json"
            pair = {}
            if pair_path.exists():
                for e in json.loads(pair_path.read_text())["departures"]:
                    pair[e["i"]] = e["t_mean"] - e["p_mean"]
            rows = []
            for i in select:
                s = records[i]
                prior = s["child_prior"] / s["child_prior"].sum()
                t = variants[kind](s)
                p_top, t_top = int(np.argmax(prior)), int(np.argmax(t))
                n = s["child_n"]
                rows.append({
                    "i": i, "t": t, "prior": prior, "dep": t_top != p_top, "p_top": p_top, "t_top": t_top,
                    "z": min(st.departure_confidence(s, t, args.prior_weight), 1e6),
                    "gap": float(s["completed_q"][t_top] - s["completed_q"][p_top]),
                    "share": float(n[t_top] / max(n[t_top] + n[p_top], 1.0)),
                    "t_mass": float(t[t_top]),
                    "log_ratio": float(np.log(max(prior[t_top], 1e-12) / prior[p_top])),
                    "nontrack": float(meta[i]["step"] != "track"),
                    "pair": pair.get(i),
                })
            deps = [r for r in rows if r["dep"]]
            paired = [r for r in deps if r["pair"] is not None]
            # Signal quality: AUC of each signal for "the departure is the reference's move" / "better".
            for ref_name, ref in refs.items():
                labels = np.array([r["t_top"] == ref[r["i"]]["best_i"] for r in deps])
                better = np.array([ref[r["i"]]["q"][r["t_top"]] > ref[r["i"]]["q"][r["p_top"]] for r in deps])
                lines.append({
                    "config": config, "target": kind, "ref": ref_name, "kind": "auc", "departures": len(deps),
                    "prec_all": float(labels.mean()) if len(deps) else float("nan"),
                    "better_all": float(better.mean()) if len(deps) else float("nan"),
                    **{f"auc_{sig}": _auc([r[sig] for r in deps], labels) for sig in SIGNALS},
                    **{f"auc_better_{sig}": _auc([r[sig] for r in deps], better) for sig in SIGNALS},
                })
            if paired:
                labels = np.array([r["pair"] > 0 for r in paired])
                lines.append({
                    "config": config, "target": kind, "ref": "pair", "kind": "auc", "departures": len(paired),
                    "prec_all": float("nan"), "better_all": float(labels.mean()),
                    "gain_all": float(np.mean([r["pair"] for r in paired])),
                    **{f"auc_{sig}": float("nan") for sig in SIGNALS},
                    **{f"auc_better_{sig}": _auc([r[sig] for r in paired], labels) for sig in SIGNALS},
                })
            for rule in args.thresholds:
                kept = [r for r in deps if _passes(r, rule)]
                for ref_name, ref in refs.items():
                    overrides = [r for r in rows if ref[r["i"]]["best_i"] != r["p_top"]]
                    found = sum(1 for r in kept if r["t_top"] == ref[r["i"]]["best_i"])
                    better = sum(1 for r in kept if ref[r["i"]]["q"][r["t_top"]] > ref[r["i"]]["q"][r["p_top"]])
                    regret, push, regret0 = [], [], []
                    for r in rows:
                        q = ref[r["i"]]["q"]
                        t = r["t"]
                        if r["dep"] and not _passes(r, rule):
                            soft = args.mode == "soft" and rule.startswith("z=") and "," not in rule
                            if soft and float(rule[2:]) > 0:
                                c = min(max(r["z"], 0.0) / float(rule[2:]), 1.0)
                                t = r["prior"] + c * (r["t"] - r["prior"])
                            else:
                                t = r["prior"]
                        regret.append(float(q[ref[r["i"]]["best_i"]] - t @ q))
                        regret0.append(float(q[ref[r["i"]]["best_i"]] - r["t"] @ q))
                        push.append(float((1 - t[r["p_top"]]) - (1 - r["prior"][r["p_top"]])))
                    d = np.array(regret) - np.array(regret0)
                    lines.append({
                        "config": config, "target": kind, "ref": ref_name, "kind": "threshold", "rule": rule,
                        "departures": len(deps), "kept": len(kept), "kept_share": len(kept) / max(len(deps), 1),
                        "precision": found / len(kept) if kept else float("nan"),
                        "recall": found / max(len(overrides), 1),
                        "better": better / len(kept) if kept else float("nan"),
                        "regret": float(np.mean(regret)), "regret_vs_unfiltered": float(d.mean()),
                        "regret_vs_unfiltered_se": float(d.std(ddof=1) / np.sqrt(len(d))), "push": float(np.mean(push)),
                    })
                if paired:
                    kp = [r for r in paired if _passes(r, rule)]
                    total = sum(r["pair"] for r in paired)
                    lines.append({
                        "config": config, "target": kind, "ref": "pair", "kind": "threshold", "rule": rule,
                        "departures": len(paired), "kept": len(kp), "kept_share": len(kp) / len(paired),
                        "precision": float("nan"), "recall": float("nan"),
                        "better": float(np.mean([r["pair"] > 0 for r in kp])) if kp else float("nan"),
                        "gain_kept": float(np.mean([r["pair"] for r in kp])) if kp else float("nan"),
                        "gain_dropped": float(np.mean([r["pair"] for r in paired if not _passes(r, rule)]))
                        if len(kp) < len(paired) else float("nan"),
                        "gain_share": sum(r["pair"] for r in kp) / total if total else float("nan"),
                        "regret": float("nan"), "regret_vs_unfiltered": float("nan"), "push": float("nan"),
                    })
    out = data / (args.out or "confidence.jsonl")
    out.write_text("".join(json.dumps(line) + "\n" for line in lines))
    print(f"positions {len(select)}{' step ' + args.step if args.step else ''}; written to {out}")
    for line in lines:
        if line["kind"] == "auc":
            print(f"{line['config']:<9} {line['target']:<13} ref {line['ref']:<5} departures {line['departures']:4d} "
                  f"prec {line['prec_all']:.3f} better {line['better_all']:.3f} | AUC(match) "
                  + " ".join(f"{sig} {line['auc_' + sig]:.2f}" for sig in SIGNALS)
                  + " | AUC(better) " + " ".join(f"{sig} {line['auc_better_' + sig]:.2f}" for sig in SIGNALS))
    print(f"{'config':<9} {'target':<13} {'ref':<5} {'rule':<16} {'kept':>5} {'share':>6} {'prec':>6} {'recall':>6} "
          f"{'better':>6} {'regret':>8} {'d_regret':>9} {'push':>7} {'gain_k':>7} {'gain_d':>7} {'gain%':>6}")
    for line in lines:
        if line["kind"] == "threshold":
            print(f"{line['config']:<9} {line['target']:<13} {line['ref']:<5} {line['rule']:<16} {line['kept']:5d} "
                  f"{line['kept_share']:6.3f} {line['precision']:6.3f} {line['recall']:6.3f} {line['better']:6.3f} "
                  f"{1e3 * line['regret']:8.3f} {1e3 * line['regret_vs_unfiltered']:+9.3f} {line['push']:+7.3f} "
                  f"{1e3 * line.get('gain_kept', float('nan')):7.2f} "
                  f"{1e3 * line.get('gain_dropped', float('nan')):7.2f} "
                  f"{line.get('gain_share', float('nan')):6.3f}")


def pairwise(args) -> None:
    """For each departure of a target (its argmax is not the policy's) on a run, the two
    children -- the target's move and the policy's top move, built as the judged search
    builds them (``--expander``: the judged network's priors and price head) -- are each
    searched as a root by ``--model`` (an independent network) with ``--readouts``; the
    mover's value of each (the root's mean backed-up value, and its most visited move's)
    is a direct, prior-free comparison of the two moves. Writes ``pairwise.<config>.<target>.json``."""
    import engine_rs
    import torch

    from diag_sibling_values import sample_positions
    from reanalyze_siblings import reanalysis_settings, search_values

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.composite_model import load_policy_value
    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, TreeSearch
    from rl18xx.agent.alphazero.policy_gradient import use_tf32
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
    from rl18xx.rust_adapter import RustGameAdapter

    data = Path(args.data)
    meta = json.loads((data / "positions.json").read_text())
    rng = np.random.default_rng(args.seed)
    positions = []
    for collection in args.collection:
        positions.extend(sample_positions(Path(collection), args.games, args.per_game, rng, args.game_offset))
    if len(positions) != len(meta) or any(
        p[0] != m["game"] or p[1] != m["action"] for p, m in zip(positions, meta)
    ):
        raise SystemExit("the collection arguments do not give the run's positions")
    with (data / f"{args.config}.pkl").open("rb") as f:
        records = pickle.load(f)["records"]
    variants = target_variants()
    deps = []
    for i, s in enumerate(records):
        t = variants[args.target](s)
        p_top, t_top = int(np.argmax(s["child_prior"])), int(np.argmax(t))
        if p_top != t_top:
            deps.append((i, int(s["legal"][t_top]), int(s["legal"][p_top])))
    LOGGER.info(f"{args.config}/{args.target}: {len(deps)} departures")

    use_tf32()
    torch.set_num_threads(args.torch_threads)
    net = load_policy_value(args.model)
    expander = load_policy_value(args.expander)
    evaluator, expand_eval = ModelEvaluator(net), ModelEvaluator(expander)
    gen = sl.Generator(reanalysis_settings(args.readouts, 0.05, args.temperature), {"m": evaluator})
    searcher = LockstepSearcher({"m": evaluator, "x": expand_eval}, dedupe_maps=True)
    out = []
    for b in range(0, len(deps), args.chunk):
        part = deps[b : b + args.chunk]
        players, roots = [], []
        for i, _, _ in part:
            rs = positions[i][3]
            config = SelfPlayConfig(
                network=expander, use_inference_server=False, num_readouts=1, min_readouts=1, c_puct_init=0.05,
                c_puct_by_round={}, mcts_mean_q=True, fpu_reduction=0.0, dedupe_leaves=True, parallel_readouts=8,
                prior_temperature=args.temperature, softpick_move_cutoff=0, dirichlet_noise_weight=0.0,
                enable_resign=False, noresign_holdout_rate=0.0, max_game_length=1000,
                player_count_distribution={NUM_PLAYERS: 1.0}, auction_unlock=True,
            )
            player = RustMCTSPlayer(config)
            player.initialize_game(RustGameAdapter(rs.pickle_clone()))
            players.append(player)
            s = player.start_search(key="x")
            if s is not None:
                roots.append(s)
        with torch.no_grad():
            searcher.round(roots)
        jobs, entries = [], []
        for (i, t_move, p_move), player in zip(part, players):
            tree = player._rust_player
            mover = int(positions[i][3].encode_canonical_py()[3])
            entry = {"i": i, "t": t_move, "p": p_move, "mover": mover}
            for side, move in (("t", t_move), ("p", p_move)):
                game = tree.get_game_for_idx(tree.maybe_add_child(tree.root_idx, move, None))
                if bool(game.finished):
                    v = np.asarray(engine_rs.mcts_terminal_value(game, True), dtype=np.float64)[:NUM_PLAYERS]
                    entry[f"{side}_mean"] = entry[f"{side}_best"] = float(v[mover] / max(v.sum(), 1e-9))
                else:
                    jobs.append((entry, side, TreeSearch(gen._tree(game), args.readouts, key="m", budget="total")))
            entries.append(entry)
        with torch.no_grad():
            searcher.run([s for _, _, s in jobs])
        for entry, side, s in jobs:
            for kind, value in search_values(s.tree).items():
                entry[f"{side}_{kind}"] = float(value[entry["mover"]])
        out.extend(entries)
        LOGGER.info(f"{len(out)} / {len(deps)} departures compared, {evaluator.states} evaluations")
    path = data / f"pairwise.{args.config}.{args.target}.{args.name}.json"
    path.write_text(json.dumps({"model": args.model, "readouts": args.readouts, "departures": out}))
    better = np.array([e["t_mean"] > e["p_mean"] for e in out])
    LOGGER.info(f"wrote {path}: the target's move valued above the policy's in {better.mean():.3f} of departures")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search")
    p.add_argument("--model", required=True, help="a shared checkpoint (its own value) or <policy>+<value>")
    p.add_argument("--collection", action="append", required=True)
    p.add_argument("--games", type=int, default=40, help="games per collection")
    p.add_argument("--per-game", type=int, default=5)
    p.add_argument("--game-offset", type=int, default=0, help="skip each collection's first games")
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
    p = sub.add_parser("confidence")
    p.add_argument("--data", required=True)
    p.add_argument("--config", nargs="+", required=True, help="the runs whose targets to filter (e.g. 0.05@48)")
    p.add_argument("--self-reference", default="0.05@800", help="the same network's deeper run in --data")
    p.add_argument("--reference", default="0.05@800")
    p.add_argument("--reference-data", default=None, help="an independent network's run directory (az1)")
    p.add_argument("--targets", nargs="+", default=["cq", "visits_pruned"])
    p.add_argument("--thresholds", nargs="+", default=["all", "z=1", "z=2", "z=3", "share=0.5", "nontrack=1"],
                   help="<signal>=<min>[,<signal>=<min>...] (all required), signals " + ", ".join(SIGNALS))
    p.add_argument("--pairwise", default="az1",
                   help="the pairwise file's name (pairwise.<config>.<target>.<name>.json)")
    p.add_argument("--mode", choices=("prior", "soft"), default="prior")
    p.add_argument("--prior-weight", type=float, default=2.0)
    p.add_argument("--step", default=None)
    p.add_argument("--out", default=None)
    p = sub.add_parser("pairwise")
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--target", default="cq")
    p.add_argument("--model", required=True, help="the independent network that searches both children")
    p.add_argument("--expander", required=True, help="the judged network (builds the children as its search did)")
    p.add_argument("--name", default="az1")
    p.add_argument("--readouts", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--collection", action="append", required=True)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--per-game", type=int, default=5)
    p.add_argument("--game-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"search": search, "score": score, "steps": steps, "pressure": pressure, "paired": paired,
     "confidence": confidence, "pairwise": pairwise}[args.cmd](args)


if __name__ == "__main__":
    main()
