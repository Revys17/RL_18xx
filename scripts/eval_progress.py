"""Side A's running score in an eval_head_to_head output directory (games.jsonl so far)."""

import json
import math
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    rows = [json.loads(line) for line in (Path(arg) / "games.jsonl").read_text().splitlines() if line.strip()]
    scores = [sum(r["win_share"][s] for s, side in enumerate(r["seats"]) if side == "A") for r in rows]
    n = len(scores)
    mean = sum(scores) / n if n else float("nan")
    se = math.sqrt(sum((x - mean) ** 2 for x in scores) / (n - 1) / n) if n > 1 else float("nan")
    print(f"{arg}: {n} games, A score {mean:.3f} +- {se:.3f}")
