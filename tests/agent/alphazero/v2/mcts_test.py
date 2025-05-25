import pytest
import copy
import math
import numpy as np
from typing import Tuple

from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper

from rl18xx.agent.alphazero.v2.mcts import MCTSNode, POLICY_SIZE
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.v2.config import MegaConfig
GameObjects = Tuple[BaseGame, ActionHelper, ActionMapper]

@pytest.fixture
def game_objects() -> GameObjects:
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game_instance = game_class(players)
    return (game_instance, ActionHelper(game_instance), ActionMapper())

@pytest.fixture
def near_terminal_game_objects(game_objects) -> GameObjects:
    initial_game_state, action_helper, action_mapper = game_objects

    # Auction
    initial_game_state.process_action(
        action_helper.get_all_choices()[-2]
    )  # [20:39] -- Phase 2 (Operating Rounds: 1 | Train Limit: 4 | Available Tiles: Yellow) --
    # [20:39] Player 1 bids $600 for Baltimore & Ohio
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Schuylkill Valley for $20
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 3 buys Champlain & St.Lawrence for $40
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 4 buys Delaware & Hudson for $70
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 1 passes bidding
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Mohawk & Hudson for $110
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 3 buys Camden & Amboy for $160
    # [20:39] Player 3 receives a 10% share of PRR
    # [20:39] Player 1 wins the auction for Baltimore & Ohio with the only bid of $600
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [20:39] Player 1 pars B&O at $67
    # [20:39] Player 1 receives a 20% share of B&O
    # [20:39] Player 1 becomes the president of B&O
    # [20:39] Player 4 has priority deal
    # [20:39] -- Stock Round 1 --
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:39] Player 4 buys a 10% share of B&O from the IPO for $67
    # [20:39] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 2 buys a 10% share of B&O from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 3 buys a 10% share of B&O from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:13] B&O floats
    # [21:13] B&O receives $670
    # [21:13] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [21:13] Player 2 pars PRR at $67
    # [21:13] Player 2 buys a 20% share of PRR from the IPO for $134
    # [21:13] Player 2 becomes the president of PRR
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:13] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 2 buys a 10% share of PRR from the IPO for $67
    # [21:13] PRR floats
    # [21:13] PRR receives $670
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:14] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:14] Player 4 becomes the president of B&O
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 4 becomes the president of PRR
    # [21:14] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 3 passes
    # [21:15] Player 4 has no valid actions and passes
    # [21:15] PRR's share price moves up from 71
    # [21:15] Player 1 has priority deal
    # [21:15] -- Operating Round 1.1 (of 1) --
    # [21:15] Player 1 collects $30 from Baltimore & Ohio
    # [21:15] Player 2 collects $5 from Schuylkill Valley
    # [21:15] Player 2 collects $20 from Mohawk & Hudson
    # [21:15] Player 3 collects $10 from Champlain & St.Lawrence
    # [21:15] Player 3 collects $25 from Camden & Amboy
    # [21:15] Player 4 collects $15 from Delaware & Hudson
    # [21:15] Player 4 operates PRR
    # [21:15] PRR places a token on H12
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:16] PRR lays tile #57 with rotation 1 on H10 (Pittsburgh)
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:16] PRR passes place a token
    # [21:16] PRR skips run routes
    # [21:16] PRR does not run
    # [21:16] PRR's share price moves left from 67
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:17] PRR passes buy trains
    # [21:17] PRR skips buy companies
    # [21:17] Player 4 operates B&O
    # [21:17] B&O places a token on I15
    initial_game_state.process_action(
        action_helper.get_all_choices()[4]
    )  # [21:17] B&O spends $80 and lays tile #57 with rotation 0 on J14 (Washington)
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:17] B&O passes place a token
    # [21:17] B&O skips run routes
    # [21:17] B&O does not run
    # [21:17] B&O's share price moves left from 65
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:22] B&O buys a 2 train for $590 from PRR
    # [21:22] Baltimore & Ohio closes
    # [21:22] B&O skips buy companies
    # [21:22] -- Stock Round 2 --
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [21:23] Player 1 passes
    # [23:26] Player 2 pars NYC at $67
    initial_game_state.process_action(
        action_helper.get_all_choices()[31]
    )  # [23:26] Player 2 buys a 20% share of NYC from the IPO for $134
    # [23:26] Player 2 becomes the president of NYC
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:26] Player 2 exchanges Mohawk & Hudson from the IPO for a 10% share of NYC
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 2 declines to sell shares
    initial_game_state.process_action(action_helper.get_all_choices()[13])  # [23:26] Player 3 pars C&O at $67
    # [23:26] Player 3 buys a 20% share of C&O from the IPO for $134
    # [23:26] Player 3 becomes the president of C&O
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 3 declines to sell shares
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:26] Player 4 sells 3 shares of B&O and receives $195
    # [23:26] Player 1 becomes the president of B&O
    # [23:26] B&O's share price moves down from 50
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])
    # [23:27] Player 1 has no valid actions and passes
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 2 buys a 10% share of NYC from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 2 declines to sell shares
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 3 declines to sell shares
    initial_game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    # [23:27] NYC floats
    # [23:27] NYC receives $670
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 4 declines to sell shares
    # [23:27] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[2])  # [23:27] Player 2 sells 3 shares of PRR and receives $201
    # [23:27] PRR's share price moves down from 60
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 2 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])
    initial_game_state.process_action(action_helper.get_all_choices()[1])  # [23:27] Player 3 sells 2 shares of PRR and receives $120
    # [23:27] PRR's share price moves down from 40
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    initial_game_state.process_action(action_helper.get_all_choices()[-1])
    initial_game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 4 buys a 10% share of C&O from the IPO for $67
    # [23:27] C&O floats
    # [23:27] C&O receives $670
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 declines to sell shares
    # [23:35] Player 1 has no valid actions and passes
    initial_game_state.process_action(action_helper.get_all_choices()[20])  # [23:35] Player 2 sells a 10% share of B&O and receives $50
    # [23:35] B&O's share price moves down from 40
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 declines to buy shares
    initial_game_state.process_action(action_helper.get_all_choices()[4])  # [23:35] Player 3 sells a 10% share of B&O and receives $40
    # [23:35] B&O's share price moves down from 30
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 declines to buy shares
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 1 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 passes
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 passes
    # [23:35] Player 4 has priority deal
    # [23:35] -- Operating Round 2.1 (of 1) --
    # [23:35] Player 4 collects $15 from Delaware & Hudson
    # [23:35] Player 2 collects $5 from Schuylkill Valley
    # [23:35] Player 3 collects $10 from Champlain & St.Lawrence
    # [23:35] Player 3 collects $25 from Camden & Amboy
    # [23:35] Player 2 operates NYC
    # [23:35] NYC places a token on E19
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] NYC passes lay/upgrade track
    # [23:35] NYC skips place a token
    # [23:35] NYC skips run routes
    # [23:35] NYC does not run
    # [23:35] NYC's share price moves left from 65
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] NYC buys a 2 train for $80 from The Depot
    # [23:36] NYC skips buy companies
    # [23:36] Player 3 operates C&O
    # [23:36] C&O places a token on F6
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes lay/upgrade track
    # [23:36] C&O skips place a token
    # [23:36] C&O skips run routes
    # [23:36] C&O does not run
    # [23:36] C&O's share price moves left from 65
    initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] C&O buys a 3 train for $180 from The Depot
    # [23:36] -- Phase 3 (Operating Rounds: 2 | Train Limit: 4 | Available Tiles: Yellow, Green) --
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes buy trains
    # [23:36] C&O passes buy companies
    # [23:36] Player 4 operates PRR
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes lay/upgrade track
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes place a token
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR runs a 2 train for $30: H12-H10
    initial_game_state.process_action(
        action_helper.get_all_choices()[-1]
    )  # [23:36] PRR pays out 3 per share (12 to Player 4, $3 to Player 3)
    # [23:36] PRR's share price moves right from 50
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    initial_game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 4 train for $300 from The Depot
    # [23:36] -- Phase 4 (Operating Rounds: 2 | Train Limit: 3 | Available Tiles: Yellow, Green) --
    # [23:36] -- Event: 2 trains rust ( B&O x1, PRR x1, NYC x4) --
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes buy companies
    # [23:36] Player 1 operates B&O
    initial_game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] B&O passes lay/upgrade track
    # [23:36] B&O skips place a token
    # [23:36] B&O skips run routes
    # [23:36] B&O does not run
    # [23:36] B&O's share price moves left from 20
    # initial_game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] Player bankrupts
    return initial_game_state, action_helper, action_mapper

@pytest.fixture
def mcts_config() -> MegaConfig:
    return MegaConfig()


def assertNoPendingVirtualLosses(root):
    """Raise an error if any node in this subtree has vlosses pending."""
    queue = [root]
    while queue:
        current = queue.pop()
        assert current.losses_applied == 0
        queue.extend(current.children.values())

# Tests

def test_upper_bound_confidence(game_objects, mcts_config: MegaConfig):
    game_instance, action_helper, action_mapper = game_objects

    probs = np.array([.02] * POLICY_SIZE)
    root = MCTSNode(game_instance)
    leaf = root.select_leaf()
    assert root == leaf
    value = np.array([0.5, 0.0, 0.0, 0.0])
    leaf.incorporate_results(probs, value, root)

    num_legal_actions = len(action_helper.get_all_choices_limited())
    # 0.02 are normalized to 1/ num_legal_actions
    assert np.isclose(
        root.child_prior[0],
        1.0/num_legal_actions,
        rtol=1e-6), f"root.child_prior[0]: {root.child_prior[0]}, should be 1/{num_legal_actions} {1.0/num_legal_actions}"
    assert np.isclose(
        root.child_prior[1],
        1.0/num_legal_actions,
        rtol=1e-6), f"root.child_prior[1]: {root.child_prior[1]}, should be 1/{num_legal_actions} {1.0/num_legal_actions}"

    puct_policy = lambda n: 2.0 * (math.log((1.0 + n + mcts_config.c_puct_base) / mcts_config.c_puct_base) + mcts_config.c_puct_init) * 1.0 / num_legal_actions

    assert root.N == 1
    assert np.isclose(
        root.child_U[0],
        puct_policy(root.N),
        rtol=1e-6), f"root.child_U[0]: {root.child_U[0]}, should be {puct_policy(root.N)}"

    leaf = root.select_leaf()
    assert root != leaf

    # With the first child expanded.
    assert root.N == 1
    assert np.isclose(
        root.child_U[0], puct_policy(root.N) * math.sqrt(1) / (1 + 0), rtol=1e-6)
    assert np.isclose(
        root.child_U[1], puct_policy(root.N) * math.sqrt(1) / (1 + 0), rtol=1e-6)

    leaf.add_virtual_loss(up_to=root)
    leaf2 = root.select_leaf()

    assert leaf2 not in (root, leaf)

    leaf.revert_virtual_loss(up_to=root)
    value = np.array([0.3, 0.0, 0.0, 0.0])
    leaf.incorporate_results(probs, value, root)
    leaf2.incorporate_results(probs, value, root)

    # With the 2nd child expanded.
    assert root.N == 3
    assert np.isclose(
        root.child_U[0], puct_policy(root.N) * math.sqrt(2) / (1 + 1), rtol=1e-6)
    assert np.isclose(
        root.child_U[1], puct_policy(root.N) * math.sqrt(2) / (1 + 1), rtol=1e-6)
    assert np.isclose(
        root.child_U[2], puct_policy(root.N) * math.sqrt(2) / (1 + 0), rtol=1e-6)

def test_select_leaf(game_objects, mcts_config: MegaConfig):
    game_instance, action_helper, action_mapper = game_objects

    flattened = 0
    probs = np.array([.02] * POLICY_SIZE)
    probs[flattened] = 0.4
    root = MCTSNode(game_instance)
    value = np.array([0.0, 0.0, 0.0, 0.0])
    root.select_leaf().incorporate_results(probs, value, root)

    assert root.active_player_index == root.player_mapping[game_instance.active_players()[0].id]
    assert root.select_leaf() == root.children[flattened]

def test_backup_incorporate_results(game_objects):
    game_instance, action_helper, action_mapper = game_objects

    probs = np.array([.02] * POLICY_SIZE)
    root = MCTSNode(game_instance)
    value = np.array([0.0, 0.0, 0.0, 0.0])
    root.select_leaf().incorporate_results(probs, value, root)

    leaf = root.select_leaf()
    value = np.array([1.0, -1.0, -1.0, -1.0])
    leaf.incorporate_results(probs, value, root)  # P2 wins!

    # Root was visited twice: first at the root, then at this child.
    assert root.N == 2
    # Root has 0 as a prior and two visits with value 0, -1
    assert math.isclose(root.Q_perspective, 1.0 / 3, rel_tol=1e-6)
    # Leaf should have one visit
    assert root.child_N[leaf.fmove] == 1
    assert leaf.N == 1
    # And that leaf's value had its parent's Q (0) as a prior, so the Q
    # should now be the average of 0, -1
    assert np.allclose(root.child_Q[leaf.fmove], np.array([0.5, -0.5, -0.5, -0.5]))
    assert np.allclose(leaf.Q, np.array([0.5, -0.5, -0.5, -0.5]))

    # We're assuming that select_leaf() returns a leaf like:
    #   root
    #     \
    #     leaf
    #       \
    #       leaf2
    # which happens in this test because root is W to play and leaf was a W win.
    assert root.active_player_index == root.player_mapping[game_instance.active_players()[0].id]
    leaf2 = root.select_leaf()
    value = np.array([0.2, -0.2, -0.2, -0.2])
    leaf2.incorporate_results(probs, value, root)  # another white semi-win
    assert root.N == 3
    # average of 0, 0, -1, -0.2
    assert np.allclose(root.Q, np.array([0.3, -0.3, -0.3, -0.3]))

    assert leaf.N == 2
    assert leaf2.N == 1
    # average of 0, -1, -0.2
    assert np.allclose(root.child_Q[leaf.fmove], leaf.Q)
    assert np.allclose(leaf.Q, np.array([0.4, -0.4, -0.4, -0.4]))
    # average of -1, -0.2
    assert np.allclose(leaf.child_Q[leaf2.fmove], np.array([0.6, -0.6, -0.6, -0.6]))
    assert np.allclose(leaf2.Q, np.array([0.6, -0.6, -0.6, -0.6]))

def test_do_not_explore_past_finish(near_terminal_game_objects):
    game_instance, action_helper, action_mapper = near_terminal_game_objects
    probs = np.array([0.02] * POLICY_SIZE, dtype=np.float32)
    root = MCTSNode(game_instance)
    value = np.array([0.0, 0.0, 0.0, 0.0])
    root.select_leaf().incorporate_results(probs, value, root)
    action_index = action_mapper.get_index_for_action(action_helper.get_all_choices()[-1])
    end_game_action = root.maybe_add_child(action_index) # Only choice is bankrupt
    with pytest.raises(AssertionError):
        value = np.array([0.0, 0.0, 0.0, 0.0])
        end_game_action.incorporate_results(probs, value, root)
    node_to_explore = end_game_action.select_leaf()
    # should just stop exploring at the end position.
    assert end_game_action == node_to_explore

def test_add_child(game_objects):
    game_instance, action_helper, action_mapper = game_objects

    root = MCTSNode(game_instance)
    action_index = action_mapper.get_index_for_action(action_helper.get_all_choices()[0])
    child = root.maybe_add_child(action_index)
    assert action_index in root.children
    assert root == child.parent
    assert child.fmove == action_index

def test_add_child_idempotency(game_objects):
    game_instance, action_helper, action_mapper = game_objects
    root = MCTSNode(game_instance)
    
    action_index = action_mapper.get_index_for_action(action_helper.get_all_choices()[0])
    child = root.maybe_add_child(action_index)
    current_children = copy.copy(root.children)
    child2 = root.maybe_add_child(action_index)
    assert child == child2
    assert current_children == root.children

def test_never_select_illegal_moves(game_objects):
    game_instance, action_helper, action_mapper = game_objects

    probs = np.array([0.02] * POLICY_SIZE)
    # let's say the NN were to accidentally put a high weight on an illegal move
    probs[9999] = 0.99
    root = MCTSNode(game_instance)
    value = np.array([0.0, 0.0, 0.0, 0.0])
    root.incorporate_results(probs, value, root)
    # and let's say the root were visited a lot of times, which pumps up the
    # action score for unvisited moves...
    root.N = 100000

    root.child_N[action_mapper.get_legal_action_indices(game_instance)] = 10000
    # this should not throw an error...
    leaf = root.select_leaf()
    # the returned leaf should not be the illegal move
    assert 9999 != leaf.fmove

    # and even after injecting noise, we should still not select an illegal move
    for i in range(10):
        root.inject_noise()
        leaf = root.select_leaf()
        assert 9999 != leaf.fmove

def test_dont_pick_unexpanded_child(game_objects):
    game_instance, action_helper, action_mapper = game_objects

    probs = np.array([0.001] * POLICY_SIZE)
    # make one move really likely so that tree search goes down that path twice
    # even with a virtual loss
    probs[0] = 0.999
    root = MCTSNode(game_instance)
    value = np.array([0.0, 0.0, 0.0, 0.0])
    root.incorporate_results(probs, value, root)
    root.N = 5
    leaf1 = root.select_leaf()
    assert 0 == leaf1.fmove
    leaf1.add_virtual_loss(up_to=root)
    # the second select_leaf pick should return the same thing, since the child
    # hasn't yet been sent to neural net for eval + result incorporation
    leaf2 = root.select_leaf()
    assert leaf1 == leaf2

def test_normalize_policy(game_objects):
    game_instance, action_helper, action_mapper = game_objects

    # sum of probs > 1.0
    probs = np.array([2.0] * POLICY_SIZE)

    root = MCTSNode(game_instance)
    value = np.array([0.0, 0.0, 0.0, 0.0])
    root.incorporate_results(probs, value, root)
    root.N = 0

    # Policy sums to 1.0, only legal moves have non-zero values.
    num_legal_actions = len(action_helper.get_all_choices_limited())
    assert math.isclose(1.0, sum(root.child_prior))
    assert num_legal_actions == np.count_nonzero(root.child_prior)
    illegal_moves = 1 - root.legal_action_mask
    assert 0 == sum(root.child_prior * illegal_moves)

def test_inject_noise_only_legal_moves(game_objects):
    game_instance, action_helper, action_mapper = game_objects

    probs = np.array([0.02] * POLICY_SIZE)
    root = MCTSNode(game_instance)
    value = np.array([0.0, 0.0, 0.0, 0.0])
    root.incorporate_results(probs, value, root)
    root.N = 0

    uniform_policy = 1 / root.num_legal_actions
    expected_policy = uniform_policy * (root.legal_action_mask)

    assert np.allclose(root.child_prior, expected_policy, atol=1e-6), f"root.child_prior: {root.child_prior}, expected_policy: {expected_policy}"

    root.inject_noise()

    expected_policy_legal_moves = expected_policy[root.legal_action_indices]
    child_prior_legal_moves = root.child_prior[root.legal_action_indices]

    bound_tolerance = 1e-7
    noise_weight = root.config.dirichlet_noise_weight

    lower_bound_check_val = (expected_policy_legal_moves * (1 - noise_weight)) - child_prior_legal_moves
    assert (lower_bound_check_val <= bound_tolerance).all(), \
        f"Lower bound failed. Diff: {lower_bound_check_val}, Max diff: {np.max(lower_bound_check_val)}"
    upper_bound_check_val = child_prior_legal_moves - (expected_policy_legal_moves * (1 - noise_weight) + noise_weight)
    assert (upper_bound_check_val <= bound_tolerance).all(), \
        f"Upper bound failed. Diff: {upper_bound_check_val}, Max diff: {np.max(upper_bound_check_val)}"

    # Policy sums to 1.0, only legal moves have non-zero values.
    assert np.isclose(1.0, sum(root.child_prior), rtol=1e-6)
    assert 0 == sum(root.child_prior * (1 - root.legal_action_mask))