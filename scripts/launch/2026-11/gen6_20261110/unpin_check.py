"""Run from the work dir with the deployment's python: per-count pins (must be none), exploiter and pg2p shares."""
import json
from rl18xx.agent.alphazero import policy_gradient as pg

s = json.load(open("model_checkpoints_pg/gen6_20261110/state.json"))
lg = pg.league_for_run(pg.PGConfig(**s["config"]), s["league"])
for n in lg.counts:
    l = lg.league(n)
    w = dict(zip(l.members, l.weights()))
    assert not l.min_shares, (n, l.min_shares)
    assert not any(f"/gx_{c}_20261010/" in m for m in l.members for c in ("prr", "nyc", "cpr", "bo", "co", "erie", "nynh", "bm"))
    ex = sum(v for m, v in w.items() if "/gx" in m)
    print(f"{n}p members {len(l.members)} exploiters {ex:.3f} pg2p {w.get('model_checkpoints_pg/pg2p_20261109/learner/2000.pth', 0):.3f} gen5 {w.get('model_checkpoints_pg/gen5_20261107/learner/2800.pth', 0):.3f}")
