"""Held-out report for shared-network candidates: the refit's win CE protocol and policy drift.

On a held-out collection's positions (every 25th action after the 20th with a choice,
``value_refit.heldout_selector``, as ``refit_value_heads.py`` reports): each
candidate's win CE against the outcome by game stage (``value_refit.stage_report``)
and its policy's drift from ``--anchor`` (KL(anchor || candidate) at T over the legal
moves, top-move agreement; ``value_refit.policy_drift``).

Usage::

    uv run python scripts/diag_shared_report.py --heldout EV/az1_u200_sys64 \\
        --anchor model_checkpoints_search/az1_20261017/learner/280.pth \\
        --candidate critic=model_checkpoints_search/az1_20261017/critic/280.pth --candidate shared=...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heldout", nargs="+", required=True)
    parser.add_argument("--anchor", required=True, help="The policy drift is measured from this network")
    parser.add_argument("--candidate", action="append", required=True, help="name=<checkpoint>")
    parser.add_argument("--no-drift", action="append", default=[], help="Candidates without a policy (critics)")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    from rl18xx.agent.alphazero import value_refit as vr
    from rl18xx.agent.alphazero.policy_gradient import _load, use_tf32

    use_tf32()
    positions = vr.collect(args.heldout, lambda gid, n: vr.heldout_selector)
    data = vr.extract(positions, None, {}, None)  # outcome targets only
    anchor = _load(args.anchor).eval()
    device = next(anchor.parameters()).device
    report = {"heldout": args.heldout, "positions": len(positions), "candidates": {}}
    for item in args.candidate:
        name, _, path = item.partition("=")
        model = _load(path).eval()
        with torch.no_grad():
            stats = vr.stage_report(data, vr.model_predictions(model, positions, device))
            if name not in args.no_drift:
                stats.update(vr.policy_drift(anchor, model, positions, args.temperature, device))
        report["candidates"][name] = {"path": path, **stats}
        print(vr.format_report(name, stats), "| drift", {k: round(stats[k], 4) for k in stats if k.startswith(("policy", "top1"))})
        del model
        torch.cuda.empty_cache()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
