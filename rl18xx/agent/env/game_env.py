"""OpenAI Gym-like environment for 18XX games."""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ...game.gamemap import GameMap
from ...game.actionfinder import ActionHelper
from ..utils.state_utils import encode_full_state
from ..utils.action_utils import ActionSpace
from ..utils.reward_utils import RewardCalculator

class Game18XXEnv(gym.Env):
    """Environment wrapper for 18XX games."""
    
    def __init__(self, 
                 num_players: int = 4,
                 max_rounds: int = 30,
                 reward_weights: Optional[Dict[str, float]] = None):
        """Initialize environment.
        
        Args:
            num_players: Number of players in game
            max_rounds: Maximum number of rounds before termination
            reward_weights: Custom weights for reward calculation
        """
        super().__init__()
        
        # Game setup
        self.num_players = num_players
        self.max_rounds = max_rounds
        self.game_map = GameMap()
        self.game = None
        self.action_helper = None
        self.current_player_idx = 0
        
        # Action/observation handling
        self.action_space = ActionSpace()
        self.reward_calculator = RewardCalculator(**(reward_weights or {}))
        
        # Define observation space
        # This will need to match our state encoding dimensions
        self.observation_space = spaces.Dict({
            'board': spaces.Box(
                low=-1.0, high=1.0, 
                shape=(20, 20, 54),  # Adjust dimensions to match board encoding
                dtype=np.float32
            ),
            'players': spaces.Box(
                low=-1.0, high=1.0,
                shape=(num_players, 45),  # Adjust to match player encoding
                dtype=np.float32
            ),
            'corporations': spaces.Box(
                low=-1.0, high=1.0,
                shape=(10, 34),  # Adjust to match corporation encoding
                dtype=np.float32
            ),
            'market': spaces.Box(
                low=-1.0, high=1.0,
                shape=(57,),  # Adjust to match market encoding
                dtype=np.float32
            ),
            'phase': spaces.Box(
                low=-1.0, high=1.0,
                shape=(14,),  # Adjust to match phase encoding
                dtype=np.float32
            )
        })
        
        # Track episode progress
        self.steps = 0
        self.total_rounds = 0
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for initialization
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Initialize new game
        player_names = {str(i+1): f"Player {i+1}" for i in range(self.num_players)}
        self.game = self.game_map.game_by_title("1830")
        self.game = self.game(player_names)
        
        if seed is not None:
            self.game.seed = seed
            
        self.action_helper = ActionHelper(self.game)
        self.current_player_idx = 0
        self.steps = 0
        self.total_rounds = 0
        
        # Get initial observation
        observation = encode_full_state(self.game)
        
        # Info dict
        info = {
            'current_player': self.current_player_idx,
            'round_type': self.game.round.__class__.__name__,
            'valid_actions': self.get_valid_actions(),
            'action_mask': self.get_action_mask()
        }
        
        return observation, info
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action: Agent's action (from model)
            
        Returns:
            Tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Decode and validate action
        game_action = self.action_space.decode_action(action, self.game)
        if game_action is None:
            # Invalid action, return negative reward
            return self._get_current_state(), -1.0, False, False, {'error': 'Invalid action'}
            
        # Get current player for reward calculation
        current_player = self.game.current_entity
        
        # Process action
        try:
            self.game.process_action(game_action)
            self.steps += 1
            
            # Update round counter if needed
            if hasattr(self.game, 'round_counter'):
                self.total_rounds = self.game.round_counter
                
        except Exception as e:
            # Handle invalid actions or game errors
            return self._get_current_state(), -1.0, False, False, {'error': str(e)}
            
        # Get next state
        next_state = self._get_current_state()
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            current_player,
            self.game,
            game_action,
            self._is_terminal()
        )
        
        # Check termination
        terminated = self._is_terminal()
        truncated = self.total_rounds >= self.max_rounds
        
        # Info dict
        info = {
            'current_player': self.current_player_idx,
            'round_type': self.game.round.__class__.__name__,
            'valid_actions': self.get_valid_actions(),
            'action_mask': self.get_action_mask(),
            'last_action': game_action.__class__.__name__,
            'last_reward': reward
        }
        
        return next_state, reward, terminated, truncated, info
        
    def _get_current_state(self) -> Dict[str, np.ndarray]:
        """Get current encoded state."""
        return encode_full_state(self.game)
        
    def _is_terminal(self) -> bool:
        """Check if game is in terminal state."""
        return (
            self.game.finished or
            hasattr(self.game, 'bankrupt') and self.game.bankrupt
        )
        
    def get_valid_actions(self) -> List[np.ndarray]:
        """Get list of valid actions in current state."""
        valid_game_actions = self.action_helper.get_all_choices()
        return [
            self.action_space.encode_action(action)
            for action in valid_game_actions
        ]
        
    def get_action_mask(self) -> np.ndarray:
        """Get boolean mask of valid actions."""
        valid_game_actions = self.action_helper.get_all_choices()
        return self.action_space.get_action_mask(valid_game_actions)
        
    def render(self, mode: str = 'human'):
        """Render current game state."""
        if mode == 'human':
            # Print basic game state
            print(f"Round: {self.game.round.__class__.__name__}")
            print(f"Current Player: {self.game.current_entity.name}")
            print(f"Valid Actions: {len(self.get_valid_actions())}")
        else:
            return str(self.game)  # Basic string representation
            
    def close(self):
        """Clean up environment."""
        pass