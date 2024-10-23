from typing import List, Any, Dict
import numpy as np

class BaseAgent:
    """Base class for RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int):
        """Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def select_action(self, 
                     state: np.ndarray, 
                     valid_actions: List[Any], 
                     training: bool = True) -> Any:
        """Select an action given the current state.
        
        Args:
            state: Current state observation
            valid_actions: List of valid actions
            training: Whether the agent is training
            
        Returns:
            Selected action
        """
        raise NotImplementedError
    
    def learn(self, 
             state: np.ndarray,
             action: Any,
             reward: float,
             next_state: np.ndarray,
             done: bool) -> Dict:
        """Update agent's knowledge based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode is done
            
        Returns:
            Dict containing training metrics
        """
        raise NotImplementedError
    
    def save(self, path: str):
        """Save agent to file.
        
        Args:
            path: Path to save file
        """
        raise NotImplementedError
    
    def load(self, path: str):
        """Load agent from file.
        
        Args:
            path: Path to load file
        """
        raise NotImplementedError