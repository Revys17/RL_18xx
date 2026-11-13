"""gen5's plateau rule (2026-11-10; the user: keep going until it plateaus). A hundred is FLAT when the checkpoint
beats the one 200 updates older by less than 0.53 at 2v2 (fair 0.50) AND less than 0.28 alone against three of it
(fair 0.25); 800 games each, policy-only at T 0.5, auction starts. The run stops after two flat hundreds in a row,
counted from u1000 on (earliest stop u1100). Prints the table, then STOP <u> or GO."""
import re, sys

res = open(sys.argv[1]).read()
pat = re.compile(r"policy-only u(\d+)_(2v2|1v3)_vs_u(\d+): A ([0-9.]+) \+- ([0-9.]+)")
got = {}
for u, kind, p, a, se in pat.findall(res):
    if int(u) - int(p) == 200:
        got[(int(u), kind)] = (float(a), float(se))
flat_run = 0
verdict = "GO"
for u in sorted({u for u, _ in got}):
    if (u, "2v2") not in got or (u, "1v3") not in got:
        continue
    a2, s2 = got[(u, "2v2")]
    a1, s1 = got[(u, "1v3")]
    flat = a2 < 0.53 and a1 < 0.28
    if u >= 1000:
        flat_run = flat_run + 1 if flat else 0
        if flat_run >= 2 and verdict == "GO":
            verdict = f"STOP {u}"
    print(f"u{u} vs u{u-200}: 2v2 {a2:.3f}+-{s2:.3f}  1v3 {a1:.3f}+-{s1:.3f}  {'flat' if flat else 'gain'}")
print(verdict)
