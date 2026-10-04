"""Mean-Q selection (``RustMCTSPlayer.set_q_config``): the value of a move,
not just its prior, decides where the search spends its visits.

Values here are 0-1 win shares. Minigo's ``W / (1 + N)`` with unvisited
children at 0 reads an unexplored move as a certain loss, so a better move
the prior doesn't favor barely gets looked at; the mean value with first-play
urgency at the parent's value finds it.
"""

import numpy as np
import pytest

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE
from rl18xx.rust_adapter import RustGameAdapter

engine_rs = pytest.importorskip("engine_rs")

GOOD, BAD = 0.6, 0.1


def _search(mean_q: bool, readouts: int = 64) -> tuple[float, float]:
    """Sequential readouts from the opening auction where passing at the root
    is worth GOOD to the player to move and anything else BAD, but the prior
    gives pass the least weight. Returns (share of root visits on pass, share
    on the best other move)."""
    game = RustGameAdapter(engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(4)}))
    player = engine_rs.RustMCTSPlayer(game)
    player.set_search_config(0.25, 19652.0, {}, 1000)
    if mean_q:
        player.set_q_config(True, 0.0)
    root_player = player.root_active_player_index()
    root_len = len(player.root_game_object().raw_actions)
    legal = [int(i) for i in player.legal_action_indices_at_root()]
    mapper = ActionMapper()
    pass_slot = next(i for i in legal if type(mapper.map_index_to_action(i, game)).__name__ == "Pass")
    priors = np.zeros(POLICY_SIZE, dtype=np.float32)
    priors[legal] = 0.98 / (len(legal) - 1)
    priors[pass_slot] = 0.02
    uniform = np.full(POLICY_SIZE, 1.0 / POLICY_SIZE, dtype=np.float32)

    for _ in range(readouts):
        leaf = player.select_leaf()
        first = leaf == 0
        value = np.zeros(VALUE_SIZE, dtype=np.float32)
        actions = player.get_game_for_idx(leaf).raw_actions
        if len(actions) > root_len:
            mine = GOOD if actions[root_len]["type"] == "pass" else BAD
            value[:4] = (1.0 - mine) / 3
            value[root_player] = mine
        else:
            value[:4] = 0.25
        player.incorporate_results(leaf, priors if first else uniform, value)

    visits = np.asarray(player.child_n_at_root())
    total = visits.sum()
    return float(visits[pass_slot] / total), float(np.delete(visits, pass_slot).max() / total)


def test_mean_q_puts_the_most_visits_on_the_better_move():
    # The search still widens in prior order (the parent's value is no
    # promise), but once pass is tried its value takes every later visit.
    pass_share, best_other = _search(mean_q=True)
    assert pass_share > 2 * best_other


def test_minigo_q_never_tries_the_better_move():
    # Unvisited children sit at 0 -- a certain loss on a 0-1 scale -- so a
    # move the prior dislikes is never explored at 64 readouts.
    assert _search(mean_q=False)[0] < 0.05
