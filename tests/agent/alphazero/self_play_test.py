import pytest
import torch

from unittest import mock
import numpy as np
from rl18xx.agent.alphazero.self_play import MCTSPlayer
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import VALUE_SIZE
from rl18xx.game.gamemap import GameMap
from rl18xx.game.action_helper import ActionHelper
from rl18xx.rust_adapter import RustGameAdapter
from engine_rs import BaseGame as RustGame

# Fixtures


class DummyNet:
    def __init__(self, fake_priors=None, fake_log_priors=None, fake_value=None):
        if fake_priors is None:
            fake_priors = torch.ones(26537, dtype=torch.float32) / 26537
        if fake_log_priors is None:
            fake_log_priors = torch.log(fake_priors)
        if fake_value is None:
            fake_value = torch.zeros(VALUE_SIZE, dtype=torch.float32)
        self.fake_priors = fake_priors
        self.fake_log_priors = fake_log_priors
        self.fake_value = fake_value

    def encoder_type(self):
        return "GNN"

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


_PLAYER_NAMES = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}


def _new_rust_game():
    """Construct a fresh ``RustGameAdapter`` for replay/test use."""
    return RustGameAdapter(RustGame(_PLAYER_NAMES))


def _replay_on_rust(py_game):
    """Build a ``RustGameAdapter`` and replay ``py_game.raw_actions`` on it.

    The fixture below constructs the desired game state using the Python engine
    (because the Python engine's ``ActionHelper.get_all_choices`` enumerates
    actions the way the original index-based fixture expects). Once the desired
    state is reached, we replay the action history on a fresh Rust engine —
    the tests that consume the fixture (``test_extract_data_normal_end`` and
    the parallel-tree-search tests) all exercise paths that now require a
    ``RustGameAdapter``. Replay is a faithful, decoupled translation: action
    dicts are engine-agnostic, and both engines accept them.
    """
    rust_game = _new_rust_game()
    for action in py_game.raw_actions:
        d = action if isinstance(action, dict) else action.to_dict()
        # ``created_at`` is a wall-clock timestamp that has no semantic effect on
        # game state and the Rust engine is happy without it.
        d = {k: v for k, v in d.items() if k != "created_at"}
        rust_game.process_action(d)
    return rust_game


def get_fresh_game_state():
    """A fresh empty 4-player 1830 ``RustGameAdapter`` (the engine the rest of
    the AlphaZero pipeline runs on)."""
    return _new_rust_game()


def _get_fresh_python_game():
    """Internal: Python-engine fixture used only by ``get_almost_done_game_state``
    / ``terminal_game_state`` to step through the move sequence by action index.
    See ``_replay_on_rust`` for the rationale on the hybrid approach."""
    game_class = GameMap().game_by_title("1830")
    return game_class(_PLAYER_NAMES)


def _build_almost_done_python_game():
    """Build the 'almost done' game state on the Python engine and return it
    without converting to Rust. Used both by ``get_almost_done_game_state``
    (which wraps the result in a Rust adapter) and by ``terminal_game_state``
    (which continues stepping with the Python engine before its own wrap)."""
    g = _get_fresh_python_game()
    action_helper = ActionHelper()
    # action_helper.print_enabled = True


    # Auction
    g.process_action(
        action_helper.get_all_choices(g)[-2]
    )  # [20:39] -- Phase 2 (Operating Rounds: 1 | Train Limit: 4 | Available Tiles: Yellow) --
    # [20:39] Player 1 bids $600 for Baltimore & Ohio
    g.process_action(action_helper.get_all_choices(g)[0])  # [20:39] Player 2 buys Schuylkill Valley for $20
    g.process_action(action_helper.get_all_choices(g)[0])  # [20:39] Player 3 buys Champlain & St.Lawrence for $40
    g.process_action(action_helper.get_all_choices(g)[0])  # [20:39] Player 4 buys Delaware & Hudson for $70
    g.process_action(action_helper.get_all_choices(g)[0])  # [20:39] Player 1 passes bidding
    g.process_action(action_helper.get_all_choices(g)[0])  # [20:39] Player 2 buys Mohawk & Hudson for $110
    g.process_action(action_helper.get_all_choices(g)[0])  # [20:39] Player 3 buys Camden & Amboy for $160
    # [20:39] Player 3 receives a 10% share of PRR
    # [20:39] Player 1 wins the auction for Baltimore & Ohio with the only bid of $600
    g.process_action(action_helper.get_all_choices(g)[-1])  # [20:39] Player 1 pars B&O at $67
    # [20:39] Player 1 receives a 20% share of B&O
    # [20:39] Player 1 becomes the president of B&O
    # [20:39] Player 4 has priority deal
    action_helper.print_summary(g, json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 0, "shares": {"B&O": 20}, "companies": ["BO"]},
            "Player 2": {"cash": 470, "shares": {}, "companies": ["SV", "MH"]},
            "Player 3": {"cash": 400, "shares": {"PRR": 10}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 530, "shares": {}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYC": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": [], "share_price": 67},
            "C&O": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state(g) == expected_state, "State does not match expected state after auction"
    # [20:39] -- Stock Round 1 --
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [20:39] Player 4 buys a 10% share of B&O from the IPO for $67
    # [20:39] Player 1 has no valid actions and passes
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [21:13] Player 2 buys a 10% share of B&O from the IPO for $67
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [21:13] Player 3 buys a 10% share of B&O from the IPO for $67
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [21:13] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:13] B&O floats
    # [21:13] B&O receives $670
    # [21:13] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices(g)[-2])  # [21:13] Player 2 pars PRR at $67
    # [21:13] Player 2 buys a 20% share of PRR from the IPO for $134
    # [21:13] Player 2 becomes the president of PRR
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:13] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:13] Player 1 has no valid actions and passes
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:13] Player 2 buys a 10% share of PRR from the IPO for $67
    # [21:13] PRR floats
    # [21:13] PRR receives $670
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:13] Player 3 buys a 10% share of PRR from the IPO for $67
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [21:14] Player 4 buys a 10% share of B&O from the IPO for $67
    # [21:14] Player 4 becomes the president of B&O
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:14] Player 2 passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:14] Player 3 passes
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:14] Player 2 passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:14] Player 3 passes
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:14] Player 2 passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:14] Player 3 passes
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:14] Player 4 buys a 10% share of PRR from the IPO for $67
    # [21:14] Player 4 becomes the president of PRR
    # [21:14] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:15] Player 2 passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:15] Player 3 passes
    # [21:15] Player 4 has no valid actions and passes
    # [21:15] PRR's share price moves up from 71
    # [21:15] Player 1 has priority deal
    action_helper.print_summary(g, json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 30, "shares": {"B&O": 20}, "companies": ["BO"]},
            "Player 2": {"cash": 227, "shares": {"B&O": 10, "PRR": 30}, "companies": ["SV", "MH"]},
            "Player 3": {"cash": 234, "shares": {"PRR": 30, "B&O": 10}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 76, "shares": {"B&O": 30, "PRR": 40}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 670, "companies": [], "trains": [], "share_price": 71},
            "NYC": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 670, "companies": [], "trains": [], "share_price": 67},
            "C&O": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state(g) == expected_state, "State does not match expected state after stock round 1"

    # [21:15] -- Operating Round 1.1 (of 1) --
    # [21:15] Player 1 collects $30 from Baltimore & Ohio
    # [21:15] Player 2 collects $5 from Schuylkill Valley
    # [21:15] Player 2 collects $20 from Mohawk & Hudson
    # [21:15] Player 3 collects $10 from Champlain & St.Lawrence
    # [21:15] Player 3 collects $25 from Camden & Amboy
    # [21:15] Player 4 collects $15 from Delaware & Hudson
    # [21:15] Player 4 operates PRR
    # [21:15] PRR places a token on H12

    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [21:16] PRR lays tile #57 with rotation 1 on H10 (Pittsburgh)
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:16] PRR passes place a token
    # [21:16] PRR skips run routes
    # [21:16] PRR does not run
    # [21:16] PRR's share price moves left from 67
    g.process_action(action_helper.get_all_choices(g)[1])  # [21:16] PRR buys a 2 train for $80 from The Depot
    g.process_action(action_helper.get_all_choices(g)[1])  # [21:16] PRR buys a 2 train for $80 from The Depot
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:17] PRR passes buy trains
    # [21:17] PRR skips buy companies
    # [21:17] Player 4 operates B&O
    # [21:17] B&O places a token on I15
    g.process_action(
        action_helper.get_all_choices(g)[5]
    )  # [21:17] B&O spends $80 and lays tile #57 with rotation 0 on J14 (Washington)
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:17] B&O passes place a token
    # [21:17] B&O skips run routes
    # [21:17] B&O does not run
    # [21:17] B&O's share price moves left from 65
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:22] B&O buys a 2 train for $590 from PRR
    # [21:22] Baltimore & Ohio closes
    # [21:22] B&O skips buy companies
    action_helper.print_summary(g, json_format=True)
    expected_state = {
        "players": {
            "Player 1": {"cash": 30, "shares": {"B&O": 20}, "companies": []},
            "Player 2": {"cash": 227, "shares": {"B&O": 10, "PRR": 30}, "companies": ["SV", "MH"]},
            "Player 3": {"cash": 234, "shares": {"PRR": 30, "B&O": 10}, "companies": ["CS", "CA"]},
            "Player 4": {"cash": 76, "shares": {"B&O": 30, "PRR": 40}, "companies": ["DH"]},
        },
        "corporations": {
            "PRR": {"cash": 1100, "companies": [], "trains": ["2"], "share_price": 67},
            "NYC": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "CPR": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&O": {"cash": 0, "companies": [], "trains": ["2"], "share_price": 65},
            "C&O": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "ERIE": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "NYNH": {"cash": 0, "companies": [], "trains": [], "share_price": None},
            "B&M": {"cash": 0, "companies": [], "trains": [], "share_price": None},
        },
    }
    assert action_helper.get_state(g) == expected_state, "State does not match expected state after operating round 1"

    # [21:22] -- Stock Round 2 --
    g.process_action(action_helper.get_all_choices(g)[-1])  # [21:23] Player 1 passes
    # [23:26] Player 2 pars NYC at $67
    g.process_action(
        action_helper.get_all_choices(g)[31]
    )  # [23:26] Player 2 buys a 20% share of NYC from the IPO for $134
    # [23:26] Player 2 becomes the president of NYC
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [23:26] Player 2 exchanges Mohawk & Hudson from the IPO for a 10% share of NYC
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:26] Player 2 declines to sell shares
    g.process_action(action_helper.get_all_choices(g)[13])  # [23:26] Player 3 pars C&O at $67
    # [23:26] Player 3 buys a 20% share of C&O from the IPO for $134
    # [23:26] Player 3 becomes the president of C&O
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:26] Player 3 declines to sell shares
    g.process_action(action_helper.get_all_choices(g)[-2])  # [23:26] Player 4 sells 3 shares of B&O and receives $195
    # [23:26] Player 1 becomes the president of B&O
    # [23:26] B&O's share price moves down from 50
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    g.process_action(action_helper.get_all_choices(g)[-1])
    # [23:27] Player 1 has no valid actions and passes
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [23:27] Player 2 buys a 10% share of NYC from the IPO for $67
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:27] Player 2 declines to sell shares
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:27] Player 3 declines to sell shares
    g.process_action(
        action_helper.get_all_choices(g)[0]
    )  # [23:27] Player 4 buys a 10% share of NYC from the IPO for $67
    # [23:27] NYC floats
    # [23:27] NYC receives $670
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:27] Player 4 declines to sell shares
    # [23:27] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices(g)[2])  # [23:27] Player 2 sells 3 shares of PRR and receives $201
    # [23:27] PRR's share price moves down from 60
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [23:27] Player 2 buys a 10% share of C&O from the IPO for $67
    g.process_action(action_helper.get_all_choices(g)[-1])
    g.process_action(action_helper.get_all_choices(g)[1])  # [23:27] Player 3 sells 2 shares of PRR and receives $120
    # [23:27] PRR's share price moves down from 40
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [23:27] Player 3 buys a 10% share of C&O from the IPO for $67
    g.process_action(action_helper.get_all_choices(g)[-1])
    g.process_action(
        action_helper.get_all_choices(g)[1]
    )  # [23:27] Player 4 buys a 10% share of C&O from the IPO for $67
    # [23:27] C&O floats
    # [23:27] C&O receives $670
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:35] Player 4 declines to sell shares
    # [23:35] Player 1 has no valid actions and passes
    g.process_action(action_helper.get_all_choices(g)[20])  # [23:35] Player 2 sells a 10% share of B&O and receives $50
    # [23:35] B&O's share price moves down from 40
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:35] Player 2 declines to buy shares
    g.process_action(action_helper.get_all_choices(g)[4])  # [23:35] Player 3 sells a 10% share of B&O and receives $40
    # [23:35] B&O's share price moves down from 30
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:35] Player 3 declines to buy shares
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:35] Player 4 passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:35] Player 1 passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:35] Player 2 passes
    g.process_action(action_helper.get_all_choices(g)[-1])  # [23:35] Player 3 passes
    return g


def get_almost_done_game_state():
    """Public fixture: same game state as ``_build_almost_done_python_game``,
    but returned as a ``RustGameAdapter`` so downstream consumers (MCTS, the
    encoder, ``extract_data``) run on the Rust engine."""
    return _replay_on_rust(_build_almost_done_python_game())


def terminal_game_state():
    game_state = _build_almost_done_python_game()
    action_helper = ActionHelper()
    # [23:35] Player 4 has priority deal
    # [23:35] -- Operating Round 2.1 (of 1) --
    # [23:35] Player 4 collects $15 from Delaware & Hudson
    # [23:35] Player 2 collects $5 from Schuylkill Valley
    # [23:35] Player 3 collects $10 from Champlain & St.Lawrence
    # [23:35] Player 3 collects $25 from Camden & Amboy
    # [23:35] Player 2 operates NYC
    # [23:35] NYC places a token on E19
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:35] NYC passes lay/upgrade track
    # [23:35] NYC skips place a token
    # [23:35] NYC skips run routes
    # [23:35] NYC does not run
    # [23:35] NYC's share price moves left from 65
    game_state.process_action(action_helper.get_all_choices(game_state)[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices(game_state)[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices(game_state)[0])  # [23:35] NYC buys a 2 train for $80 from The Depot
    game_state.process_action(action_helper.get_all_choices(game_state)[0])  # [23:36] NYC buys a 2 train for $80 from The Depot
    # [23:36] NYC skips buy companies
    # [23:36] Player 3 operates C&O
    # [23:36] C&O places a token on F6
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:36] C&O passes lay/upgrade track
    # [23:36] C&O skips place a token
    # [23:36] C&O skips run routes
    # [23:36] C&O does not run
    # [23:36] C&O's share price moves left from 65
    game_state.process_action(action_helper.get_all_choices(game_state)[0])  # [23:36] C&O buys a 3 train for $180 from The Depot
    # [23:36] -- Phase 3 (Operating Rounds: 2 | Train Limit: 4 | Available Tiles: Yellow, Green) --
    game_state.process_action(action_helper.get_all_choices(game_state)[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices(game_state)[-2])  # [23:36] C&O buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:36] C&O passes buy trains
    # [23:36] C&O passes buy companies
    # [23:36] Player 4 operates PRR
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:36] PRR passes lay/upgrade track
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:36] PRR passes place a token
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:36] PRR runs a 2 train for $30: H12-H10
    game_state.process_action(
        action_helper.get_all_choices(game_state)[-1]
    )  # [23:36] PRR pays out 3 per share (12 to Player 4, $3 to Player 3)
    # [23:36] PRR's share price moves right from 50
    game_state.process_action(action_helper.get_all_choices(game_state)[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices(game_state)[-2])  # [23:36] PRR buys a 3 train for $180 from The Depot
    game_state.process_action(action_helper.get_all_choices(game_state)[-2])  # [23:36] PRR buys a 4 train for $300 from The Depot
    # [23:36] -- Phase 4 (Operating Rounds: 2 | Train Limit: 3 | Available Tiles: Yellow, Green) --
    # [23:36] -- Event: 2 trains rust ( B&O x1, PRR x1, NYC x4) --
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:36] PRR passes buy companies
    # [23:36] Player 1 operates B&O
    game_state.process_action(action_helper.get_all_choices(game_state)[-1])  # [23:36] B&O passes lay/upgrade track
    # [23:36] B&O skips place a token
    # [23:36] B&O skips run routes
    # [23:36] B&O does not run
    # [23:36] B&O's share price moves left from 20
    # game_state.process_action(action_helper.get_all_choices(game_state)[0])  # [23:36] Player bankrupts
    return _replay_on_rust(game_state)


def initialize_basic_player(game_state=None):
    player = MCTSPlayer(SelfPlayConfig(network=DummyNet(), use_score_values=False, backup_discount=1.0))
    player.initialize_game(game_state)
    first_node = player.root.select_leaf()
    first_node.ensure_encoded()
    with torch.no_grad():
        priors, _, values = player.config.network.run_encoded(first_node.encoded_game_state)
    first_node.incorporate_results(priors, values, up_to=player.root)
    return player


def initialize_almost_done_player():
    probs = torch.tensor([0.001] * 26537)
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
    assert max_p > 3.0 / 26537


def test_pick_moves():
    player = initialize_basic_player()
    root = player.root
    root.child_N_compressed[0] = 10
    root.child_N_compressed[1] = 5
    root.child_N_compressed[2] = 1

    root.game_object.raw_actions =["a"] * 600

    print(root.legal_action_indices)

    # Assert we're picking deterministically
    assert root.game_object.move_number > player.config.softpick_move_cutoff
    move = player.pick_move()
    assert move == 0

    root.game_object.raw_actions =["a"] * 10
    # But if we're in the early part of the game, pick randomly
    assert root.game_object.move_number < player.config.softpick_move_cutoff

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
    player = MCTSPlayer(SelfPlayConfig(network=DummyNet(fake_value=torch.tensor([0.17, 0.0, 0.0, 0.0, 0.0, 0.0]))))
    player.initialize_game()
    assert player.root.N == 0
    assert not player.root.is_expanded
    leaves = player.tree_search(parallel_readouts=4)
    assert len(leaves) == 4
    assert player.root == leaves[0]

    assert_no_pending_virtual_losses(player.root)
    # The root is selected 4 times in the parallel batch. The first visit
    # expands the node; the duplicates still back up their network value so
    # information from each evaluation is not lost. Therefore N==4 and W is
    # the sum of the 4 backed-up values.
    assert player.root.N == 4
    # Q = W / (1 + N); each of the 4 backups contributed 0.17 to W[0].
    # 0.17 * 4 / (1 + 4) = 0.136
    assert np.isclose(0.136, player.root.Q_perspective)


def test_extract_price_targets_categorical_only_is_empty():
    """A categorical move (no price range) should produce an empty
    ``price_targets`` entry — pretraining handles only its own observed
    price; self-play's MCTS-visit aggregation kicks in only for PW slots."""
    player = initialize_basic_player()
    # The root's first legal action in the initial game (auction phase) is a
    # categorical-only ``Pass``/``Bid`` — pass is categorical, the bid for
    # a non-private-with-min-bid is, but the very first action in our default
    # game tree is categorical (pass). Pick any legal index; pass will be in
    # the legal set, ``_extract_price_targets`` returns ``[]`` whenever the
    # ``price_range`` is missing or degenerate.
    targets = player._extract_price_targets(
        action_obj=None,
        action_index=player.root.legal_action_indices[0],
        price_range=None,
    )
    assert targets == []


def test_extract_price_targets_pw_slot_aggregates_visits():
    """Synthetic price grandchildren under a PW slot should appear in the
    output as visit-weighted ``(slot_idx, price, weight, min, max)`` tuples."""
    from unittest.mock import MagicMock

    player = initialize_basic_player()
    action_index = player.root.legal_action_indices[0]

    # Fake out the slot resolver so we don't rely on a concrete action.
    fake_action_obj = object()
    player.root.action_mapper = MagicMock()
    player.root.action_mapper.price_head_slot_for_action.return_value = (
        "Bid", 7, 100, 50, 500,
    )

    # Synthesize two visited price grandchildren.
    def _gc(price, visits):
        gc = MagicMock()
        gc.sampled_price = price
        gc.N = visits
        return gc

    player.root.price_children = {action_index: {100: _gc(100, 3), 200: _gc(200, 1)}}

    targets = player._extract_price_targets(
        action_obj=fake_action_obj,
        action_index=action_index,
        price_range=(50, 500),
    )
    # Two entries, visit-weighted, with bounds from slot_info.
    assert len(targets) == 2
    slots, prices, weights, mins, maxs = zip(*targets)
    assert all(s == 7 for s in slots)
    assert sorted(prices) == [100.0, 200.0]
    # Weights sum to 1 (3/4 + 1/4)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(m == 50.0 for m in mins)
    assert all(m == 500.0 for m in maxs)


def test_extract_data_normal_end():
    player = initialize_basic_player(terminal_game_state())

    player.searches_pi = [np.zeros(26537)] * 87
    player.tree_search()
    player.play_move(player.pick_move())  # only one legal move
    assert player.root.is_done()
    player.set_result(player.root.game_result())

    data = list(player.extract_data())
    assert len(data) == 88
    game_state, _, _, result, _price_targets = data[-1]
    # Max-N value vector: 4 real player slots + 2 padded slots. Legacy
    # win/loss uses -1.0 padding so the dual-target derivation's
    # ``value > -0.5`` winner detection excludes phantom slots.
    assert np.array_equal(result, [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    assert len(game_state.raw_actions) == 87

    expected_actions = terminal_game_state().raw_actions
    for i, (game_state, _, _, _, _price_targets) in enumerate(data):
        assert len(game_state.raw_actions) == i
        if i == 0:
            continue

        expected_action = expected_actions[i - 1]
        if not isinstance(expected_action, dict):
            expected_action = expected_action.args_to_dict()
        actual_action = game_state.raw_actions[-1]
        if not isinstance(actual_action, dict):
            actual_action = actual_action.args_to_dict()

        # ``created_at`` is a wall-clock timestamp (set only by the Python
        # engine); strip it so the comparison is engine-agnostic.
        expected_action = {k: v for k, v in expected_action.items() if k != "created_at"}
        actual_action = {k: v for k, v in actual_action.items() if k != "created_at"}
        assert expected_action == actual_action, f"{i}: {expected_action} != {actual_action}"


# ---------------------------------------------------------------------------
# Additional _extract_price_targets edge cases
# ---------------------------------------------------------------------------


def _gc_mock(price, visits):
    """Helper: build a fake price grandchild with given (price, visit count)."""
    from unittest.mock import MagicMock

    gc = MagicMock()
    gc.sampled_price = price
    gc.N = visits
    return gc


def test_extract_price_targets_excludes_zero_visit_grandchildren():
    """Price grandchildren that were sampled but never visited (N == 0)
    must not appear in the output — they would otherwise dilute the
    visit-weighted price target with un-evaluated samples."""
    from unittest.mock import MagicMock

    player = initialize_basic_player()
    action_index = player.root.legal_action_indices[0]

    player.root.action_mapper = MagicMock()
    player.root.action_mapper.price_head_slot_for_action.return_value = (
        "Bid", 3, 100, 50, 500,
    )
    # Three grandchildren: 100 and 200 visited; 300 sampled but never visited.
    player.root.price_children = {
        action_index: {
            100: _gc_mock(100, 3),
            200: _gc_mock(200, 1),
            300: _gc_mock(300, 0),  # zero visits — must be excluded
        }
    }

    targets = player._extract_price_targets(
        action_obj=object(),
        action_index=action_index,
        price_range=(50, 500),
    )

    prices = [p for _, p, *_ in targets]
    assert 300 not in prices, "zero-visit grandchild leaked into targets"
    assert sorted(prices) == [100.0, 200.0]
    # Weights still sum to 1 (zero-visit ignored in numerator AND denominator? —
    # denominator uses *all* grandchildren; verify the behaviour).
    weights = [w for _, _, w, _, _ in targets]
    # total_visits = 3 + 1 + 0 = 4 → weights are 3/4 and 1/4.
    assert abs(sum(weights) - 1.0) < 1e-6
    assert sorted(weights) == [0.25, 0.75]


def test_extract_price_targets_degenerate_range_returns_empty():
    """A degenerate ``price_range`` (lo == hi) is a fixed-price slot; no
    progressive widening, no price target."""
    player = initialize_basic_player()
    action_index = player.root.legal_action_indices[0]
    targets = player._extract_price_targets(
        action_obj=object(),
        action_index=action_index,
        price_range=(100, 100),
    )
    assert targets == []


def test_extract_price_targets_multiple_grandchildren_visit_weighted():
    """Three visited grandchildren with distinct visit counts should be
    reported as visit-weighted entries (each one's weight == N_i / sum_N)."""
    from unittest.mock import MagicMock

    player = initialize_basic_player()
    action_index = player.root.legal_action_indices[0]

    player.root.action_mapper = MagicMock()
    player.root.action_mapper.price_head_slot_for_action.return_value = (
        "BuyTrain", 12, 150, 10, 900,
    )
    player.root.price_children = {
        action_index: {
            100: _gc_mock(100, 2),
            200: _gc_mock(200, 5),
            300: _gc_mock(300, 3),
        }
    }

    targets = player._extract_price_targets(
        action_obj=object(),
        action_index=action_index,
        price_range=(10, 900),
    )

    assert len(targets) == 3
    # Map price -> weight for stable assertions.
    by_price = {p: w for _, p, w, _, _ in targets}
    # total visits = 2 + 5 + 3 = 10
    assert by_price[100.0] == pytest.approx(0.2)
    assert by_price[200.0] == pytest.approx(0.5)
    assert by_price[300.0] == pytest.approx(0.3)
    # All entries report the same slot, min, and max from slot_info.
    for slot, _price, _weight, lo, hi in targets:
        assert slot == 12
        assert lo == 10.0
        assert hi == 900.0
