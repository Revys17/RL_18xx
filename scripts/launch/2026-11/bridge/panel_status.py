"""Panel means and the plateau rule of a new title's policy-gradient run, from title_recipe.sh's eval rounds.

Reads ``<eval dir>/u<U>_*/summary.json`` (scripts/eval_generic.py) and prints, per evaluated update: the fixed panel
(4 players, the learner alone vs SL@0.5, SL@1.0 and the frozen snapshots; fair 0.25 each) with its mean and standard
error, every player count against the SL start, and the chain match (4p 2v2 vs an earlier snapshot; fair 0.50).

Plateau rule (proposed in docs/rl_model.md "GPU run queue", gen6's call made mechanical): over the rounds with the
complete panel, the run has plateaued when the last two rounds both fail to beat the best panel mean of the rounds
before them (gen6: u2200 .621 and u2400 .618 did not beat u2000's .632). It is printed, never acted on: the user stops
a run (``touch <run>/STOP``).

    python scripts/launch/2026-11/bridge/panel_status.py <eval dir> [--panel-updates 200 500]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def label(spec: str) -> str:
    """A player spec -> a short name: ``.../final.pth@0.5`` -> ``sl@0.5``, ``.../200.pth@0.5`` -> ``u200@0.5``."""
    path, _, temp = spec.rpartition("@")
    stem = Path(path).stem
    name = "sl" if stem in ("final", "refit", "best_joint") else f"u{stem}" if stem.isdigit() else stem
    return f"{name}@{temp}"


def load(eval_dir: Path) -> dict:
    """``{update: {kind: [summary, ...]}}`` with kind ``panel``, ``<n>p_sl`` or ``chain``."""
    rounds: dict = defaultdict(lambda: defaultdict(list))
    for path in sorted(eval_dir.glob("u*/summary.json")):
        m = re.match(r"u(\d+)_(.+)", path.parent.name)
        if not m:
            continue
        update, name = int(m.group(1)), m.group(2)
        kind = "panel" if "_panel_" in f"_{name}" else "chain" if re.search(r"_u\d+$", name) else name
        for summary in json.loads(path.read_text()):
            rounds[update][kind].append(summary)
    return rounds


def panel_mean(summaries: list) -> tuple:
    scores = [s["A_score"] for s in summaries]
    se = math.sqrt(sum(s["A_score_se"] ** 2 for s in summaries)) / len(summaries)
    return sum(scores) / len(scores), se


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--panel-updates", type=int, nargs="*", default=[200, 500])
    args = parser.parse_args()
    rounds = load(args.eval_dir)
    if not rounds:
        print(f"no eval rounds in {args.eval_dir}")
        return
    full_members = {"sl@0.5", "sl@1.0"} | {f"u{u}@0.5" for u in args.panel_updates}
    full = []
    for update in sorted(rounds):
        kinds = rounds[update]
        print(f"u{update}:")
        if kinds.get("panel"):
            members = {label(s["B"]): s for s in kinds["panel"]}
            mean, se = panel_mean(list(members.values()))
            complete = set(members) == full_members
            parts = ", ".join(f"{k} {s['A_score']:.3f}+-{s['A_score_se']:.3f}" for k, s in sorted(members.items()))
            print(f"  panel 4p 1v3 (fair 0.25): {parts}; mean {mean:.3f} +- {se:.3f}{'' if complete else ' (partial)'}")
            if complete:
                full.append((update, mean, se))
        for kind in sorted(k for k in kinds if k not in ("panel", "chain")):
            for s in kinds[kind]:
                print(
                    f"  {kind}: {s['A_score']:.3f} +- {s['A_score_se']:.3f} (fair {s['A_score_equal']:.2f}, "
                    f"{s['games']} games, worth share {s['A_worth_share']:.3f})"
                )
        for s in kinds.get("chain", []):
            print(f"  chain 4p 2v2 vs {label(s['B'])}: {s['A_score']:.3f} +- {s['A_score_se']:.3f} (fair 0.50)")
    if len(full) < 3:
        print(f"plateau rule: {len(full)} complete panel round(s); needs 3")
        return
    best_before = max(m for _, m, _ in full[:-2])
    last_two = full[-2:]
    if all(m <= best_before for _, m, _ in last_two):
        print(
            f"PLATEAU: u{last_two[0][0]} {last_two[0][1]:.3f} and u{last_two[1][0]} {last_two[1][1]:.3f} did not beat "
            f"the best earlier panel mean {best_before:.3f} -- consider stopping (touch <run>/STOP)"
        )
    else:
        last = [round(m, 3) for _, m, _ in last_two]
        print(f"no plateau: best earlier panel mean {best_before:.3f}, last two {last}")


if __name__ == "__main__":
    main()
