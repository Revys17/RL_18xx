#!/usr/bin/env python3
"""elo.py: Bradley-Terry Elo of the run's checkpoints from every self-vs-self policy-only match in results.txt
(uA_vs_uB lines; u0 = checkpoint 10 at T 0.5), u0 pinned at 0. Prints Elo by update and the gain per 100 updates."""
import math, re
import numpy as np

X = "/home/revys/longrun/experiments/pg2p_20261109"
games = []
for line in open(f"{X}/results.txt"):
    m = re.match(r"policy-only u(\d+)_vs_u(\d+): A ([0-9.]+) \+- [0-9.]+ \(fair 0.50; (\d+) games", line)
    if m:
        a, b, s, n = int(m[1]), int(m[2]), float(m[3]), int(m[4])
        if a % 100 == 0 and b % 100 == 0:  # the hundreds are linked many ways; the 50s only through one chain
            games.append((a, b, s, n))
ids = sorted({x for g in games for x in g[:2]})
ix = {u: i for i, u in enumerate(ids)}
r = np.zeros(len(ids))
for _ in range(5000):  # MLE by gradient ascent on the binomial likelihood (ratings in natural-log odds units)
    grad = np.zeros(len(ids))
    for a, b, s, n in games:
        p = 1 / (1 + math.exp(-(r[ix[a]] - r[ix[b]])))
        grad[ix[a]] += n * (s - p); grad[ix[b]] -= n * (s - p)
    r += 0.0005 * grad
    r -= r[ix[0]]
elo = r * 400 / math.log(10)
prev = None
for u in ids:
    gain = "" if prev is None else f"  (+{(elo[ix[u]] - elo[ix[prev]]) / (u - prev) * 100:.0f} per 100 updates)"
    print(f"u{u:5d}: {elo[ix[u]]:7.0f}{gain}")
    prev = u
print(f"{len(games)} matches")
