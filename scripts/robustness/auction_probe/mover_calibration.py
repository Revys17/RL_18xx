"""Calibration of the critics' search head (placement score) for the player to move vs the others, on positions of
u1528's searched match games: mean (actual placement points - predicted), by round type."""
import sys, json, collections
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[3]
sys.path[:0] = [str(_R / "scripts/robustness"), str(_R / "scripts"), str(_R)]
import numpy as np, torch; torch.set_num_threads(16)
from regression_set import load_system, acting_player, clean_action
from rl18xx.agent.alphazero.start_positions import new_game
from rl18xx.agent.alphazero.policy_gradient import default_placement_points
sys.path.insert(0, str(_P(__file__).resolve().parent))
from _specs import spec, compared
systems = {k: load_system(v, "@400/0.05:mean,dedupe,par=8,ptemp=1.0") for k, v in compared().items()}
D = "/home/revys/longrun/experiments/search_u900_20261124/u1528_vs_bots"
recs = {json.loads(l)["game_idx"]: json.loads(l) for l in open(f"{D}/games.jsonl")}
pts = default_placement_points(4)
print("placement points", pts)
ngames, per_game = int(sys.argv[1]), int(sys.argv[2])
acc = collections.defaultdict(list)
rng = np.random.default_rng(1)
for g in list(recs)[:ngames]:
    r = recs[g]
    if r["termination"] != "finished":
        continue
    nw = r["net_worth"]; order = sorted(range(4), key=lambda s: -nw[s])
    outcome = [0.0] * 4
    for rank, s in enumerate(order):
        ties = [t for t in range(4) if nw[t] == nw[s]]
        ranks = [order.index(t) for t in ties]
        outcome[s] = float(np.mean([pts[k] for k in ranks]))
    acts = json.load(open(f"{D}/games/{g}.json"))
    pick = set(rng.choice(len(acts), size=min(per_game, len(acts)), replace=False).tolist())
    game = new_game(4, auction_unlock=False, no_buyback_rule=True)
    for k, a in enumerate(acts):
        if k in pick and not game.finished:
            mover = acting_player(game)
            rt = game._game.round.round_type
            if mover is not None:
                for name, s in systems.items():
                    for pid in (1, 2, 3, 4):
                        v = s.values(game, pid)["search"]
                        acc[(name, rt, "mover" if pid == mover else "other")].append(outcome[pid - 1] - v)
                        acc[(name, "all", "mover" if pid == mover else "other")].append(outcome[pid - 1] - v)
        game.process_action(clean_action(a))
for key in sorted(acc):
    v = np.array(acc[key])
    print(f"{key[0]:6} {key[1]:10} {key[2]:6} n {len(v):5d}  actual - predicted {v.mean():+.3f} +- {v.std()/np.sqrt(len(v)):.3f}")
