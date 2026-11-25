"""The critics' value swing on a pass that changes nothing but whose turn it is (auction, stock round): passer
before -> after, next mover before -> after. Positions from u1528's searched match games."""
import sys, json, collections
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[3]
sys.path[:0] = [str(_R / "scripts/robustness"), str(_R / "scripts"), str(_R)]
import numpy as np, torch; torch.set_num_threads(16)
from regression_set import load_system, acting_player, clean_action
from rl18xx.agent.alphazero.start_positions import new_game
LR1 = "/home/revys/longrun/work/lr1/model_checkpoints_pg/gen6_20261110"
A4 = "/home/revys/workspace/RL_18xx/model_checkpoints_generic_pg/robust_A4_20261118"
SPECS = {"gen6": f"{LR1}/learner/2400.pth+{LR1}/critic/2400.pth#score",
         "u1528": f"generic:{A4}/learner/1528.pth+generic:{A4}/critic/1528.pth#score"}
systems = {k: load_system(v, "@400/0.05:mean,dedupe,par=8,ptemp=1.0") for k, v in SPECS.items()}
E = "/home/revys/longrun/experiments/search_u900_20261124/u1528_vs_bots/games"
ngames = int(sys.argv[1]); per_game = int(sys.argv[2])
acc = collections.defaultdict(list)
rng = np.random.default_rng(0)
for g in range(ngames):
    acts = json.load(open(f"{E}/{g}.json"))
    game = new_game(4, auction_unlock=False, no_buyback_rule=True)
    cands = []
    for k, a in enumerate(acts):
        rt = game._game.round.round_type
        if a.get("type") == "pass" and rt in ("Auction", "Stock") and a.get("entity_type") == "player":
            cands.append(k)
        game.process_action(clean_action(a))
    pick = set(rng.choice(cands, size=min(per_game, len(cands)), replace=False).tolist()) if cands else set()
    game = new_game(4, auction_unlock=False, no_buyback_rule=True)
    for k, a in enumerate(acts):
        if k in pick:
            rt = game._game.round.round_type; passer = acting_player(game)
            cash0 = [p.cash for p in game.players]
            before = game.pickle_clone()
            game.process_action(clean_action(a))
            nxt = acting_player(game)
            if game._game.round.round_type == rt and nxt not in (None, passer) and [p.cash for p in game.players] == cash0:
                for name, s in systems.items():
                    vb = {p: s.values(before, p)["search"] for p in (passer, nxt)}
                    va = {p: s.values(game, p)["search"] for p in (passer, nxt)}
                    acc[(name, rt, "passer")].append(va[passer] - vb[passer])
                    acc[(name, rt, "next")].append(va[nxt] - vb[nxt])
            continue
        game.process_action(clean_action(a))
for key in sorted(acc):
    v = np.array(acc[key])
    print(f"{key[0]:6} {key[1]:8} {key[2]:7} n {len(v):4d}  mean change {v.mean():+.3f} +- {v.std()/np.sqrt(len(v)):.3f}")
