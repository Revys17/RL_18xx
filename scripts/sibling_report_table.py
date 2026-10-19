"""A compact ``diag_sibling_values.py score`` report: one metric per (source, group) cell.

    python3 scripts/sibling_report_table.py logs/shared_diag/siblings_v2/report.critic280.json spearman top1 regret
"""

import json
import sys

report = json.load(open(sys.argv[1]))["summary"]
metrics = sys.argv[2:] or ["spearman", "top1"]
groups = [g for g in report if g == "all" or g.startswith(("stage:", "step:"))]
sources = [s for s in report["all"] if s != "positions"]
for metric in metrics:
    print(f"\n{metric}")
    print(f"{'source':14s} " + " ".join(f"{g.split(':')[-1][:8]:>8s}" for g in groups))
    print(f"{'(positions)':14s} " + " ".join(f"{report[g]['positions']:8d}" for g in groups))
    for s in sources:
        print(f"{s[:14]:14s} " + " ".join(f"{report[g][s][metric]:8.3f}" for g in groups))
