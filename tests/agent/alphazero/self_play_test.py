import torch

from unittest import mock
import numpy as np
from rl18xx.agent.alphazero.self_play import MCTSPlayer
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper

# Fixtures


class DummyNet:
    def __init__(self, fake_priors=None, fake_log_priors=None, fake_value=None):
        if fake_priors is None:
            fake_priors = torch.ones(26535, dtype=torch.float32) / 26535
        if fake_log_priors is None:
            fake_log_priors = torch.log(fake_priors)
        if fake_value is None:
            fake_value = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.fake_priors = fake_priors
        self.fake_log_priors = fake_log_priors
        self.fake_value = fake_value

    def run(self, game_state):
        return self.fake_priors, self.fake_log_priors, self.fake_value

    def run_many(self, game_states):
        if not game_states:
            raise ValueError("No positions passed!")
        return (
            [self.fake_priors] * len(game_states),
            [self.fake_log_priors] * len(game_states),
            [self.fake_value] * len(game_states),
        )

    def run_encoded(self, encoded_game_state):
        return self.fake_priors, self.fake_log_priors, self.fake_value

    def run_many_encoded(self, encoded_game_states):
        return (
            [self.fake_priors] * len(encoded_game_states),
            [self.fake_log_priors] * len(encoded_game_states),
            [self.fake_value] * len(encoded_game_states),
        )


def get_fresh_game_state():
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {"1": "Player 1", "2": "Player 2", "3": "Player 3", "4": "Player 4"}
    game_instance = game_class(players)
    return game_instance


def get_almost_done_game_state():
    game_state = get_fresh_game_state()
    action_helper = ActionHelper(game_state)
    # Auction
    game_state.process_action(
        action_helper.get_all_choices()[-2]
    )  # [20:39] -- Phase 2 (Operating Rounds: 1 | Train Limit: 4 | Available Tiles: Yellow) --
    # [20:39] Player 1 bids $600 for Baltimore & Ohio
    game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Schuylkill Valley for $20
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:39] Player 3 buys Champlain & St.Lawrence for $40
    game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 4 buys Delaware & Hudson for $70
    game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 1 passes bidding
    game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 2 buys Mohawk & Hudson for $110
    game_state.process_action(action_helper.get_all_choices()[0])  # [20:39] Player 3 buys Camden & Amboy for $160
    # [20:39] Player 3 receives a 10% share of PRR
    # [20:39] Player 1 wins the auction for Baltimore & Ohio with the only bid of $600
    game_state.process_action(action_helper.get_all_choices()[-1])  # [20:39] Player 1 pars B&O at $67
    # [20:39] Player 1 receives a 20% share of B&O
    # [20:39] Player 1 becomes the president of B&O
    # [20:39] Player 4 has priority deal
    # [20:39] -- Stock Round 1 --
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [20:39] Player 4 buys a 10% share of B&O from the IPO for $67
    # [20:39] Player 1 has no valid actions and passes
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 2 buys a 10% share of B&O from the IPO for $67
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 3 buys a 10% share of B&O from the IPO for $67
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:13] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:13] B&O floats
    # [21:13] B&O receives $670
    # [21:13] Player 1 has no valid actions and passes
    game_state.process_action(action_helper.get_all_choices()[-2])  # [21:13] Player 2 pars PRR at $67
    # [21:13] Player 2 buys a 20% share of PRR from the IPO for $134
    # [21:13] Player 2 becomes the president of PRR
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:13] Player 1 has no valid actions and passes
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 2 buys a 10% share of PRR from the IPO for $67
    # [21:13] PRR floats
    # [21:13] PRR receives $670
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:14] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:14] Player 4 becomes the president of B&O
    # [21:14] Player 1 has no valid actions and passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 2 passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:14] Player 3 passes
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 4 becomes the president of PRR
    # [21:14] Player 1 has no valid actions and passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 2 passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:15] Player 3 passes
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
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [21:16] PRR lays tile #57 with rotation 1 on H10 (Pittsburgh)
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:16] PRR passes place a token
    # [21:16] PRR skips run routes
    # [21:16] PRR does not run
    # [21:16] PRR's share price moves left from 67
    game_state.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices()[0])  # [21:16] PRR buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:17] PRR passes buy trains
    # [21:17] PRR skips buy companies
    # [21:17] Player 4 operates B&O
    # [21:17] B&O places a token on I15
    game_state.process_action(
        action_helper.get_all_choices()[4]
    )  # [21:17] B&O spends $80 and lays tile #57 with rotation 0 on J14 (Washington)
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:17] B&O passes place a token
    # [21:17] B&O skips run routes
    # [21:17] B&O does not run
    # [21:17] B&O's share price moves left from 65
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:22] B&O buys a 2 train for $590 from PRR
    # [21:22] Baltimore & Ohio closes
    # [21:22] B&O skips buy companies
    # [21:22] -- Stock Round 2 --
    game_state.process_action(action_helper.get_all_choices()[-1])  # [21:23] Player 1 passes
    # [23:26] Player 2 pars NYC at $67
    game_state.process_action(
        action_helper.get_all_choices()[31]
    )  # [23:26] Player 2 buys a 20% share of NYC from the IPO for $134
    # [23:26] Player 2 becomes the president of NYC
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:26] Player 2 exchanges Mohawk & Hudson from the IPO for a 10% share of NYC
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 2 declines to sell shares
    game_state.process_action(action_helper.get_all_choices()[13])  # [23:26] Player 3 pars C&O at $67
    # [23:26] Player 3 buys a 20% share of C&O from the IPO for $134
    # [23:26] Player 3 becomes the president of C&O
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:26] Player 3 declines to sell shares
    game_state.process_action(
        action_helper.get_all_choices()[-2]
    )  # [23:26] Player 4 sells 3 shares of B&O and receives $195
    # [23:26] Player 1 becomes the president of B&O
    # [23:26] B&O's share price moves down from 50
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    game_state.process_action(action_helper.get_all_choices()[-1])
    # [23:27] Player 1 has no valid actions and passes
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 2 buys a 10% share of NYC from the IPO for $67
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 2 declines to sell shares
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 3 declines to sell shares
    game_state.process_action(
        action_helper.get_all_choices()[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    # [23:27] NYC floats
    # [23:27] NYC receives $670
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:27] Player 4 declines to sell shares
    # [23:27] Player 1 has no valid actions and passes
    game_state.process_action(
        action_helper.get_all_choices()[2]
    )  # [23:27] Player 2 sells 3 shares of PRR and receives $201
    # [23:27] PRR's share price moves down from 60
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 2 buys a 10% share of C&O from the IPO for $67
    game_state.process_action(action_helper.get_all_choices()[-1])
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 sells 2 shares of PRR and receives $120
    # [23:27] PRR's share price moves down from 40
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    game_state.process_action(action_helper.get_all_choices()[-1])
    game_state.process_action(
        action_helper.get_all_choices()[1]
    )  # [23:27] Player 4 buys a 10% share of C&O from the IPO for $67
    # [23:27] C&O floats
    # [23:27] C&O receives $670
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 declines to sell shares
    # [23:35] Player 1 has no valid actions and passes
    game_state.process_action(
        action_helper.get_all_choices()[20]
    )  # [23:35] Player 2 sells a 10% share of B&O and receives $50
    # [23:35] B&O's share price moves down from 40
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 declines to buy shares
    game_state.process_action(
        action_helper.get_all_choices()[4]
    )  # [23:35] Player 3 sells a 10% share of B&O and receives $40
    # [23:35] B&O's share price moves down from 30
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 declines to buy shares
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 4 passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 1 passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 2 passes
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] Player 3 passes
    return game_state


def terminal_game_state():
    game_state = get_almost_done_game_state()
    action_helper = ActionHelper(game_state)
    # [23:35] Player 4 has priority deal
    # [23:35] -- Operating Round 2.1 (of 1) --
    # [23:35] Player 4 collects $15 from Delaware & Hudson
    # [23:35] Player 2 collects $5 from Schuylkill Valley
    # [23:35] Player 3 collects $10 from Champlain & St.Lawrence
    # [23:35] Player 3 collects $25 from Camden & Amboy
    # [23:35] Player 2 operates NYC
    # [23:35] NYC places a token on E19
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:35] NYC passes lay/upgrade track
    # [23:35] NYC skips place a token
    # [23:35] NYC skips run routes
    # [23:35] NYC does not run
    # [23:35] NYC's share price moves left from 65
    game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices()[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] NYC buys a 2 train for $80 from The Depot
    # [23:36] NYC skips buy companies
    # [23:36] Player 3 operates C&O
    # [23:36] C&O places a token on F6
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes lay/upgrade track
    # [23:36] C&O skips place a token
    # [23:36] C&O skips run routes
    # [23:36] C&O does not run
    # [23:36] C&O's share price moves left from 65
    game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] C&O buys a 3 train for $180 from The Depot
    # [23:36] -- Phase 3 (Operating Rounds: 2 | Train Limit: 4 | Available Tiles: Yellow, Green) --
    game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] C&O passes buy trains
    # [23:36] C&O passes buy companies
    # [23:36] Player 4 operates PRR
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes lay/upgrade track
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes place a token
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR runs a 2 train for $30: H12-H10
    game_state.process_action(
        action_helper.get_all_choices()[-1]
    )  # [23:36] PRR pays out 3 per share (12 to Player 4, $3 to Player 3)
    # [23:36] PRR's share price moves right from 50
    game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices()[-2])  # [23:36] PRR buys a 4 train for $300 from The Depot
    # [23:36] -- Phase 4 (Operating Rounds: 2 | Train Limit: 3 | Available Tiles: Yellow, Green) --
    # [23:36] -- Event: 2 trains rust ( B&O x1, PRR x1, NYC x4) --
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] PRR passes buy companies
    # [23:36] Player 1 operates B&O
    game_state.process_action(action_helper.get_all_choices()[-1])  # [23:36] B&O passes lay/upgrade track
    # [23:36] B&O skips place a token
    # [23:36] B&O skips run routes
    # [23:36] B&O does not run
    # [23:36] B&O's share price moves left from 20
    # game_state.process_action(action_helper.get_all_choices()[0])  # [23:36] Player bankrupts
    return game_state


def initialize_basic_player(game_state=None):
    player = MCTSPlayer(SelfPlayConfig(network=DummyNet()))
    player.initialize_game(game_state)
    first_node = player.root.select_leaf()
    with torch.no_grad():
        priors, _, values = player.config.network.run_encoded(player.root.encoded_game_state)
    first_node.incorporate_results(priors, values, up_to=player.root)
    return player


def initialize_almost_done_player():
    probs = torch.tensor([0.001] * 26535)
    probs[2:5] = 0.2  # some legal moves along the top.
    probs[-1] = 0.2  # passing is also ok
    net = DummyNet(fake_priors=probs)
    player = MCTSPlayer(SelfPlayConfig(network=net))
    # root position is white to play with no history == white passed.
    player.initialize_game(get_almost_done_game_state())
    return player


def assert_no_pending_virtual_losses(root):
    """Raise an error if any node in this subtree has vlosses pending."""
    queue = [root]
    while queue:
        current = queue.pop()
        assert current.losses_applied == 0
        queue.extend(current.children.values())


# TESTS


def test_inject_noise():
    player = initialize_basic_player()
    sum_priors = np.sum(player.root.child_prior)
    # dummyNet should return normalized priors.
    assert np.isclose(1, sum_priors)
    legal_child_U = player.root.child_U[np.where(player.root.child_U > 0)]
    assert np.all(legal_child_U == legal_child_U[0])

    player.root.inject_noise()
    new_sum_priors = np.sum(player.root.child_prior)
    # priors should still be normalized after injecting noise
    assert np.allclose(sum_priors, new_sum_priors)

    # With dirichlet noise, majority of density should be in one node.
    max_p = np.max(player.root.child_prior)
    assert max_p > 3.0 / 26535


def test_pick_moves():
    player = initialize_basic_player()
    root = player.root
    root.child_N_compressed[0] = 10
    root.child_N_compressed[1] = 5
    root.child_N_compressed[2] = 1

    root.game_dict["move_number"] = 600

    # Assert we're picking deterministically
    assert root.game_dict["move_number"] > player.config.softpick_move_cutoff
    move = player.pick_move()
    assert move == 0

    root.game_dict["move_number"] = 10
    # But if we're in the early part of the game, pick randomly
    assert root.game_dict["move_number"] < player.config.softpick_move_cutoff

    with mock.patch("random.random", lambda: 0.5):
        move = player.pick_move()
        assert move == 0

    with mock.patch("random.random", lambda: 0.99):
        move = player.pick_move()
        assert move == 2


def test_parallel_tree_search():
    player = initialize_almost_done_player()
    # initialize the tree so that the root node has populated children.
    player.tree_search(parallel_readouts=1)
    # virtual losses should enable multiple searches to happen simultaneously
    # without throwing an error...
    for _ in range(5):
        player.tree_search(parallel_readouts=4)
    # uncomment to debug this test
    # print(player.root.describe())
    assert player.root.N == 21
    assert sum(player.root.child_N) == 20

    assert_no_pending_virtual_losses(player.root)


def test_ridiculously_parallel_tree_search():
    player = initialize_almost_done_player()
    # Test that an almost complete game
    # will tree search with # parallelism > # legal moves.
    for _ in range(10):
        player.tree_search(parallel_readouts=50)
    assert_no_pending_virtual_losses(player.root)


def test_cold_start_parallel_tree_search():
    # Test that parallel tree search doesn't trip on an empty tree
    player = MCTSPlayer(SelfPlayConfig(network=DummyNet(fake_value=torch.tensor([0.17, 0.0, 0.0, 0.0]))))
    player.initialize_game()
    assert player.root.N == 0
    assert not player.root.is_expanded
    leaves = player.tree_search(parallel_readouts=4)
    assert len(leaves) == 4
    assert player.root == leaves[0]

    assert_no_pending_virtual_losses(player.root)
    # Even though the root gets selected 4 times by tree search, its
    # final visit count should just be 1.
    assert player.root.N == 1
    # 0.085 = average(0, 0.17), since 0 is the prior on the root.
    assert np.isclose(0.085, player.root.Q_perspective)


def test_extract_data_normal_end():
    player = initialize_basic_player(terminal_game_state())

    player.searches_pi = [np.zeros(26535)] * 87
    player.tree_search()
    player.play_move(player.pick_move())  # only one legal move
    assert player.root.is_done()
    player.set_result(player.root.game_result())

    data = list(player.extract_data())
    assert len(data) == 88
    game_state, _, result = data[-1]
    assert np.array_equal(result, [-1.0, -1.0, 1.0, -1.0])
    assert len(game_state.raw_actions) == 87

    expected_actions = terminal_game_state().raw_actions
    for i, (game_state, _, _) in enumerate(data):
        assert len(game_state.raw_actions) == i
        if i == 0:
            continue

        expected_action = expected_actions[i - 1]
        if not isinstance(expected_action, dict):
            expected_action = expected_action.args_to_dict()
        actual_action = game_state.raw_actions[-1]
        if not isinstance(actual_action, dict):
            actual_action = actual_action.args_to_dict()

        del expected_action["created_at"]
        del actual_action["created_at"]
        assert expected_action == actual_action, f"{i}: {expected_action} != {actual_action}"
