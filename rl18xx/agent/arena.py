from rl18xx.agent.agent import Agent
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.game.gamemap import GameMap
import logging
from rl18xx.agent.alphazero.self_play import MCTSPlayer
from rl18xx.agent.random.random_agent import RandomPlayer
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.checkpointer import get_latest_model
from rl18xx.client.game_sync import GameSync

LOGGER = logging.getLogger(__name__)

class Arena:
    def __init__(self, agent1: Agent, agent2: Agent, agent3: Agent, agent4: Agent, browser: bool = False):
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent3 = agent3
        self.agent4 = agent4
        self.agents = [agent1, agent2, agent3, agent4]
        self.browser = browser

    def get_fresh_game_state(self):
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
        return game_class(players)

    def play(self):
        game_state = self.get_fresh_game_state()
        for agent in self.agents:
            agent.initialize_game(game_state)

        agent_mapping = {
            p.id: agent
            for p, agent in zip(game_state.players, self.agents)
        }
        action_mapper = ActionMapper()
        if self.browser:
            game_sync = GameSync(game_state)

        LOGGER.info(f"Starting new game.")
        while not game_state.finished:
            if game_state.move_number >= 1000:
                game_state.end_game()
                break

            current_player = game_state.active_players()[0]
            move = agent_mapping[current_player.id].suggest_move()
            LOGGER.info(f"Player {current_player.id} (Agent {agent_mapping[current_player.id]}) plays {action_mapper.map_index_to_action(move, game_state)}")
            for agent in self.agents:
                agent.play_move(move)
            if self.browser:
                game_sync.take_synced_action(move)
            game_state = game_sync.local_game
        
        result = game_state.result()
        LOGGER.info(f"Game result: {result}")
        return game_state, agent_mapping


def test_mcts_agent_against_random_agent(browser: bool = False):
    model = get_latest_model("model_checkpoints")
    config = SelfPlayConfig(softpick_move_cutoff=0, dirichlet_noise_weight=0, network=model)
    agent1 = MCTSPlayer(config)
    agent2 = RandomPlayer()
    agent3 = RandomPlayer()
    agent4 = RandomPlayer()
    arena = Arena(agent1, agent2, agent3, agent4, browser=browser)
    final_game_state, agent_mapping = arena.play()

    agent_scores = {}
    for player_id, result in final_game_state.result().items():
        agent_scores[agent_mapping[player_id]] = result

    winner = max(agent_scores, key=agent_scores.get)
    LOGGER.info(f"Agent scores: {agent_scores}")
    LOGGER.info(f"Winner: {winner}")
    return final_game_state, agent_mapping, agent_scores, winner
