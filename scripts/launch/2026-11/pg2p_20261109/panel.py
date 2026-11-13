"""Fixed-panel table from results.txt: each hundred's score vs u700 / u900 / u1100 and the mean."""
import re
P = [700, 900, 1100]
res = {}
for l in open("/home/revys/longrun/experiments/pg2p_20261109/results.txt"):
    m = re.match(r"policy-only u(\d+)_vs_u(\d+): A ([\d.]+) \+- ([\d.]+)", l)
    if m:
        res[(int(m[1]), int(m[2]))] = (float(m[3]), float(m[4]))
hs = sorted({a for a, b in res if a % 100 == 0 and a >= 1000})
print("update  " + "  ".join(f"vs u{b:<5d}" for b in P) + "  mean(available)")
for h in hs:
    cells, vals = [], []
    for b in P:
        if (h, b) in res:
            s, se = res[(h, b)]; cells.append(f"{s:.3f}+-{se:.3f}"); vals.append(s)
        else:
            cells.append(" " * 11)
    print(f"u{h:<6d}" + "  ".join(f"{c:11s}" for c in cells) + (f"  {sum(vals)/len(vals):.3f} ({len(vals)})" if vals else ""))
