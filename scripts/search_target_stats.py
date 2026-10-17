"""How far search targets are from the policy they come from, on human
positions: for each position, gen4's policy at its temperature (the tree's
priors) against the value-driven search's raw visit distribution, the same
with KataGo's policy-target pruning (``pruned_child_n_at_root``), and the
Gumbel root's completed-Q target. Per target: its entropy, KL(target ||
policy) (what the cross-entropy loss pushes down), the share of positions
whose target argmax is the policy's, and the target mass on moves the policy
gives under 1%. CPU only; the network runs in-process.

    uv run python scripts/search_target_stats.py --model <learner.pth>+<critic.pth> --positions 40 --readouts 64 200
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--readouts", type=int, nargs="+", default=[64, 200])
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gumbel-k", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from bench_search_harness import build_position, load_network, position_prefixes

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, PolicyRequest, TreeSearch

    net = load_network(args.model, args.torch_threads)
    positions = [g for g in (build_position(p) for p in position_prefixes(args.positions, args.seed)) if g is not None]
    positions = [g for g in positions if len(g._game.factored_legal_indices()) > 1]
    searcher = LockstepSearcher({None: ModelEvaluator(net)})
    requests = [PolicyRequest(g._game, [int(i) for i in g._game.factored_legal_indices()]) for g in positions]
    searcher.run([], requests)
    policies = []
    for r in requests:
        p = np.power(np.clip(np.asarray(r.priors, dtype=np.float64), 1e-30, None), 1.0 / args.temperature)
        policies.append(p / p.sum())

    def summarize(name, targets):
        ent, kl, agree, low = [], [], [], []
        for p, t in zip(policies, targets):
            t = t / t.sum()
            nz = t > 0
            ent.append(float(-(t[nz] * np.log(t[nz])).sum()))
            kl.append(float((t[nz] * (np.log(t[nz]) - np.log(np.clip(p[nz], 1e-30, None)))).sum()))
            agree.append(float(np.argmax(t) == np.argmax(p)))
            low.append(float(t[p < 0.01].sum()))
        return {"target": name, "positions": len(targets), "entropy": round(float(np.mean(ent)), 3),
                "kl_target_policy": round(float(np.mean(kl)), 3), "argmax_agreement": round(float(np.mean(agree)), 3),
                "mass_on_policy_below_1pct": round(float(np.mean(low)), 3)}

    out = [summarize("policy", policies)]
    for readouts in args.readouts:
        for target in ("visits", "gumbel"):
            cfg = sl.SearchLearningConfig(
                policy_checkpoint="", value_checkpoint="", learner_temperature=args.temperature,
                prior_temperature=args.temperature, target=target, n_full=readouts, gumbel_k=args.gumbel_k,
            )
            gen = sl.Generator(sl.worker_settings(cfg, 0), {None: ModelEvaluator(net)})
            searches = [TreeSearch(gen._tree(g._game), readouts) for g in positions]
            searcher.run(searches)
            trees = [s.tree for s in searches]
            legal = [np.asarray(t.legal_action_indices_at_root(), dtype=np.int64) for t in trees]
            if target == "visits":
                raw = [np.asarray(t.child_n_at_root(), dtype=np.float64)[ix] for t, ix in zip(trees, legal)]
                pruned = [np.asarray(t.pruned_child_n_at_root(), dtype=np.float64)[ix] for t, ix in zip(trees, legal)]
                out.append({"readouts": readouts, **summarize("visits", raw)})
                out.append({"readouts": readouts, **summarize("visits_pruned", pruned)})
            else:
                gumbel = [np.asarray(t.gumbel_policy_target(), dtype=np.float64)[ix] for t, ix in zip(trees, legal)]
                out.append({"readouts": readouts, **summarize("gumbel", gumbel)})
    for line in out:
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
