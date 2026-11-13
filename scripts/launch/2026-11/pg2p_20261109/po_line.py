#!/usr/bin/env python3
"""po_line.py <policy_only out dir> <label>: one results.txt line from an eval_policy_only summary."""
import json, sys

s = json.load(open(f"{sys.argv[1]}/summary.json"))[0]
terms = ", ".join(f"{k} {v}" for k, v in sorted(s.get("terminations", {}).items()))
print(f"{sys.argv[2]}: A {s['A_score']:.3f} +- {s['A_score_se']:.3f} (fair {s['A_score_equal']:.2f}; {s['games']} games; "
      f"worth share {s['A_worth_share']:.3f}; mean decisions {s['mean_decisions']:.0f}; {terms})")
