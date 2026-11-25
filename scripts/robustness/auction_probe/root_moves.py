"""What the live bots' search sees at one auction position: root moves (prior, visits, Q for the mover), each bid
slot's price grandchildren, and the critic's own 1-ply value of chosen moves for the mover (search head)."""
import sys, json
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[3]
sys.path[:0] = [str(_R / "scripts/robustness"), str(_R / "scripts"), str(_R)]
import numpy as np, torch
torch.set_num_threads(8)
from regression_set import load_system, legal_moves, apply_index, acting_player, match_indices, root_stats
from rl18xx.agent.alphazero.start_positions import new_game

LR1 = "/home/revys/longrun/work/lr1/model_checkpoints_pg/gen6_20261110"
A4 = "/home/revys/workspace/RL_18xx/model_checkpoints_generic_pg/robust_A4_20261118"
SPECS = {
    "gen6": f"{LR1}/learner/2400.pth+{LR1}/critic/2400.pth#score",
    "u1528": f"generic:{A4}/learner/1528.pth+generic:{A4}/critic/1528.pth#score",
}
import os; SEARCH = os.environ.get("SEARCH", "@400/0.05:mean,dedupe,par=8,ptemp=1.0")
who = sys.argv[1]
script = [tuple(x.split(":")) for x in sys.argv[2:]]  # company:price bids played in turn order before the root
sysm = load_system(SPECS[who], SEARCH)
game = new_game(4, auction_unlock=False, no_buyback_rule=True)
for company, price in script:
    pid = acting_player(game)
    if company == "pass":
        idx = [i for i, pr in legal_moves(game) if apply_index(game, i)[1]["type"] == "pass"][0]
        game._game.apply_action_index(idx, None)
    else:
        idx = match_indices(game, {"type": "bid", "entity": pid, "entity_type": "player", "company": company, "price": int(price)})[0]
        game._game.apply_action_index(idx, int(price))
mover = acting_player(game)
print(f"== {who}: after {script}; mover player {mover}")
moves = dict(legal_moves(game))
def name(i, price=None):
    pr = moves[i]
    _, a = apply_index(game, i, price if price is not None else (pr[0] if pr else None))
    return f"{a['type']} {a.get('company','')}".strip()
player = sysm.grown_player(game)
tree = player._rust_player
st = root_stats(player)
pg = tree.price_grandchildren_at_root()
print(f"root_value {st['root_value']:.3f} raw {st['raw_value']:.3f} N {st['root_n']:.0f}; chosen {name(st['chosen'])}")
for i, n, q, p in sorted(zip(st["legal"], st["child_n"], st["child_q"], st["child_prior"]), key=lambda x: -x[1]):
    pr = moves[i]
    extra = ""
    if i in pg:
        extra = "  prices " + ", ".join(f"${k}:{v:.0f}" for k, v in sorted(pg[i].items()) if v > 0)
        try:
            sc = tree.price_child_scores(tree.get_root_idx(), i)
            extra += "  q " + ", ".join(f"${k}:{v[0]:.3f}" for k, v in sorted(sc.items()) if pg[i].get(k, 0) > 0)
        except Exception as e:
            extra += f" (scores: {e})"
    print(f"  {name(i):10} range {pr}  prior {p:.3f}  N {n:4.0f}  Q {q:.3f}{extra}")
# 1-ply critic value of the mover after each move at a few prices
print("1-ply value for the mover (search head / win / score):")
for i, pr in moves.items():
    prices = [None] if pr is None or pr[0] == pr[1] else sorted({pr[0], min(pr[1], pr[0] + 30), min(pr[1], pr[0] + 80)})
    for price in prices:
        after, a = apply_index(game, i, price if price is not None else (pr[0] if pr else None))
        v = sysm.values(after, mover)
        print(f"  {a['type']:5} {a.get('company',''):3} {a.get('price','')!s:>4}  search {v['search']:.3f}  win {v['win']:.3f}  score {v['score']:.3f}")
