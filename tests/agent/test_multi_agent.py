"""Multi-agent harness sync (gating / arena).

Each seat's ``MCTSPlayer`` keeps its own tree. A price-bearing action index
(Bid / cross-corp BuyTrain / BuyCompany) only names a categorical slot, so
replaying the bare index made every non-acting agent materialize it at *its
own* price; the agents' games forked, and a later move was illegal in some
agent's game (e.g. ``Bid 605 exceeds max 600``). Moves are now broadcast as
the committed action dicts and every agent must stay on the harness's game.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.action_mapper import ActionMapper  # noqa: E402
from rl18xx.agent.alphazero.config import ModelTransformerConfig, SelfPlayConfig  # noqa: E402
from rl18xx.agent.alphazero.loop import _create_fresh_game  # noqa: E402
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel  # noqa: E402
from rl18xx.agent.alphazero.self_play import MCTSPlayer  # noqa: E402
from rl18xx.agent.multi_agent import AgentDesyncError, action_log_key, play_multi_agent_game  # noqa: E402

FACE_VALUE = {"SV": 20, "CS": 40, "DH": 70, "MH": 110, "CA": 160, "BO": 220}
# Initial 2-player position: player 1 may bid on DH anywhere in (75, 1200).
DH_BID_INDEX = 3


@pytest.fixture(scope="module")
def untrained_models():
    models = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
        model.eval()
        models.append(model)
    return models


def _config(network) -> SelfPlayConfig:
    return SelfPlayConfig(
        softpick_move_cutoff=0,
        num_readouts=16,
        min_readouts=4,
        parallel_readouts=4,
        network=network,
    )


def _log(game) -> list[dict]:
    return [action_log_key(a) for a in game.raw_actions]


def test_agents_with_different_models_stay_in_sync_through_private_auction(untrained_models):
    game = _create_fresh_game(num_players=2)
    agents = [MCTSPlayer(_config(model)) for model in untrained_models]
    agent_by_player_id = {player.id: agent for player, agent in zip(game.players, agents)}

    committed_bids = []

    def on_commit(agent, move, action_dicts):
        committed_bids.extend(d for d in action_dicts if d["type"] == "bid")
        for each in agents:
            assert _log(each.root.game_object) == _log(game)

    play_multi_agent_game(game, agent_by_player_id, max_moves=40, on_commit=on_commit)

    # The auction must have exercised price-bearing (progressive-widening) bids:
    # anything above face value was priced by the acting agent's own tree.
    assert any(bid["price"] != FACE_VALUE[bid["company"]] for bid in committed_bids), committed_bids
    for agent in agents:
        assert _log(agent.root.game_object) == _log(game)
        assert len(agent.played_actions) == len(agent.forced_action_dicts) == len(agent.searches_pi)


def test_play_action_dicts_applies_committed_price_over_own_tree():
    player = MCTSPlayer(SelfPlayConfig(network=None))
    player.initialize_game(_create_fresh_game(num_players=2))
    own_grandchild = player.root.maybe_add_child(DH_BID_INDEX, price=300)

    bid = ActionMapper().map_index_to_action_with_price(DH_BID_INDEX, player.root.game_object, 500).to_dict()
    player.play_action_dicts(DH_BID_INDEX, [bid])

    assert player.root is not own_grandchild
    last = player.root.game_object.raw_actions[-1]
    assert (last["type"], last["company"], last["price"]) == ("bid", "DH", 500)
    assert player.committed_action_dicts() == [bid]


def test_play_action_dicts_reuses_matching_price_grandchild():
    player = MCTSPlayer(SelfPlayConfig(network=None))
    player.initialize_game(_create_fresh_game(num_players=2))
    own_grandchild = player.root.maybe_add_child(DH_BID_INDEX, price=300)

    bid = ActionMapper().map_index_to_action_with_price(DH_BID_INDEX, player.root.game_object, 300).to_dict()
    player.play_action_dicts(DH_BID_INDEX, [bid])

    assert player.root is own_grandchild


class _IgnoresOtherAgentsMoves(MCTSPlayer):
    def play_action_dicts(self, action_index, action_dicts):
        return True


def test_harness_raises_when_an_agent_falls_out_of_sync(untrained_models):
    game = _create_fresh_game(num_players=2)
    agents = [MCTSPlayer(_config(untrained_models[0])), _IgnoresOtherAgentsMoves(_config(untrained_models[1]))]
    agent_by_player_id = {player.id: agent for player, agent in zip(game.players, agents)}

    with pytest.raises(AgentDesyncError):
        play_multi_agent_game(game, agent_by_player_id, max_moves=40)
