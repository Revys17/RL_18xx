"""Drive one game between independent agents (gating, the arena).

Each seat's agent keeps its own search state, so a move has to reach every
agent as the concrete engine actions the acting agent committed — not as an
action index. A price-bearing index (Bid / cross-corp BuyTrain / BuyCompany)
only names a categorical slot; each agent's own tree would materialize it at
a different price and the agents' games would fork.
"""

from typing import Callable, Optional

from rl18xx.agent.agent import Agent

# Action-dict fields that differ between two applications of the same action.
_VOLATILE_ACTION_KEYS = ("id", "created_at")


class AgentDesyncError(RuntimeError):
    """An agent's game no longer matches the harness's source-of-truth game."""


def action_log_key(action_dict: dict) -> dict:
    """``action_dict`` without its per-application bookkeeping, for comparing
    engine logs across independently advanced copies of a game."""
    return {k: v for k, v in action_dict.items() if k not in _VOLATILE_ACTION_KEYS}


def _check_in_sync(game, agents: list[Agent], raw_before: int):
    expected_log = game.raw_actions
    expected_tail = [action_log_key(a) for a in expected_log[raw_before:]]
    for agent in agents:
        agent_log = agent.get_game_state().raw_actions
        agent_tail = [action_log_key(a) for a in agent_log[raw_before:]]
        if len(agent_log) != len(expected_log) or agent_tail != expected_tail:
            raise AgentDesyncError(
                f"{agent} desynchronized from the harness game after action {raw_before}: "
                f"expected {expected_tail}, agent has {agent_tail}"
            )


def play_multi_agent_game(
    game,
    agent_by_player_id: dict,
    max_moves: int = 1000,
    on_commit: Optional[Callable[[Agent, int, list[dict]], None]] = None,
):
    """Play ``game`` to completion and return it; ``game`` is advanced in place.

    ``game`` is the single source of truth: each agent gets its own clone, the
    active player's agent searches and commits a move with ``play_move``, and
    the actions it committed are applied to ``game`` and replayed verbatim by
    every other agent. After each move every agent's game (the acting agent's
    included) must match ``game`` or ``AgentDesyncError`` is raised.

    ``max_moves`` should equal the agents' ``max_game_length``: MCTS roots end
    the game themselves at that length. ``on_commit(agent, move, action_dicts)``
    runs after each move.
    """
    agents = list({id(agent): agent for agent in agent_by_player_id.values()}.values())
    for agent in agents:
        agent.initialize_game(game.pickle_clone())

    while not game.finished:
        if game.move_number >= max_moves:
            game.end_game()
            break

        acting = agent_by_player_id[game.active_players()[0].id]
        move = acting.suggest_move()
        acting.play_move(move)
        committed = acting.committed_action_dicts()

        raw_before = len(game.raw_actions)
        for action_dict in committed:
            game.process_action(action_dict)
        for agent in agents:
            if agent is not acting:
                agent.play_action_dicts(move, committed)
        _check_in_sync(game, agents, raw_before)

        if on_commit is not None:
            on_commit(acting, move, committed)

    return game
