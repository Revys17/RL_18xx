"""The critic's value (search head = score) of every seat along a scripted auction line, to the first SR."""
import sys
from pathlib import Path as _P
_R = _P(__file__).resolve().parents[3]
sys.path[:0] = [str(_R / "scripts/robustness"), str(_R / "scripts"), str(_R)]
import torch; torch.set_num_threads(8)
from regression_set import load_system, legal_moves, apply_index, acting_player, match_indices
from rl18xx.agent.alphazero.start_positions import new_game
sys.path.insert(0, str(_P(__file__).resolve().parent))
from _specs import spec, compared
sysm = load_system(spec(sys.argv[1]), "@400/0.05:mean,dedupe,par=8,ptemp=1.0")
game = new_game(4, auction_unlock=False, no_buyback_rule=True)
def vals(g):
    return " ".join(f"P{p}:{sysm.values(g, p)['search']:.2f}" for p in (1, 2, 3, 4))
print(f"{'start':24} {vals(game)}")
for step in sys.argv[2:]:
    while True:  # forced moves
        lm = legal_moves(game)
        if len(lm) == 1 and (lm[0][1] is None or lm[0][1][0] == lm[0][1][1]):
            game._game.apply_action_index(lm[0][0], lm[0][1][0] if lm[0][1] else None); continue
        break
    if game._game.round.round_type != "Auction":
        break
    pid = acting_player(game)
    if step == "pass":
        idx = [i for i, pr in legal_moves(game) if apply_index(game, i)[1]["type"] == "pass"][0]
        game._game.apply_action_index(idx, None)
    else:
        c, p = step.split(":")
        idx = match_indices(game, {"type": "bid", "entity": pid, "entity_type": "player", "company": c, "price": int(p)})
        if not idx:
            print(f"P{pid} {step}: not legal; legal: {[apply_index(game, i, pr[0] if pr else None)[1] for i, pr in legal_moves(game)][:8]}"); break
        game._game.apply_action_index(idx[0], int(p))
    print(f"P{pid} {step:20} {vals(game)}   round {game._game.round.round_type}")
