"""Print a distill_value_on_children run's evaluations compactly (from its log or history.json)."""

import ast
import json
import sys
from pathlib import Path

KEYS = ("step", "val_kl_target", "val_kl", "val_price_kl", "val_top1", "test_all_spearman", "test_all_top1",
        "test_all_regret", "test_step:stock_spearman", "test_step:track_spearman", "test_step:train_spearman",
        "test_outcome_ce")

for arg in sys.argv[1:]:
    path = Path(arg)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        rows = [ast.literal_eval(line[line.index("{"):]) for line in path.read_text().splitlines()
                if " INFO step " in line and ": {" in line]
    print(f"== {arg}")
    print(" ".join(f"{k.replace('test_', '').replace('step:', '')[:12]:>12s}" for k in KEYS))
    for r in rows:
        print(" ".join(f"{r.get(k, float('nan')):12.4f}" if k != "step" else f"{r[k]:12d}" for k in KEYS))
