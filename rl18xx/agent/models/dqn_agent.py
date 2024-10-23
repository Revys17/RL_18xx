from typing import List, Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from .base_agent import BaseAgent

from .network import Game18XXNetwork

class DQNetwork(nn.Module):
    """Deep Q-Network architecture using Game18XXNetwork."""
    
    def __init__(self, action_dim: int):
        """Initialize DQN.
        
        Args:
            action_dim: Dimension of action space
        """
        super().__init__()
        self.network = Game18XXNetwork(action_dim=action_dim)
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through network.
        
        Args:
            state_dict: Dictionary of state components
            
        Returns:
            Q-values for each action
        """
        action_logits, _ = self.network(state_dict)
        return action_logits

class DQNAgent(BaseAgent):
    """DQN agent implementation."""
    
    def __init__(self,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000):
        """Initialize DQN agent.
        
        Args:
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
            memory_size: Size of replay memory
            batch_size: Size of training batches
            target_update_freq: Frequency of target network updates
        """
        super().__init__(0, action_dim)  # state_dim not used with new network
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQNetwork(action_dim).to(self.device)
        self.target_net = DQNetwork(action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Target network update
        self.target_update_freq = target_update_freq
        self.steps = 0
    
    def select_action(self, 
                     state: Dict[str, np.ndarray], 
                     valid_actions: List[Any],
                     training: bool = True) -> Any:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Dictionary of state components
            valid_actions: List of valid actions
            training: Whether the agent is training
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            # Convert state dict to tensors
            state_tensors = {
                k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                for k, v in state.items()
            }
            
            # Get Q-values
            q_values = self.policy_net(state_tensors)
            
            # Create action mask from valid actions
            action_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
            for i, action in enumerate(valid_actions):
                action_mask[i] = True
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            masked_q_values[~action_mask] = float('-inf')
            
            # Select best valid action
            action_idx = masked_q_values.max(1)[1].item()
            return valid_actions[action_idx]
    
    def learn(self, 
             state: Dict[str, np.ndarray],
             action: Any,
             reward: float,
             next_state: Dict[str, np.ndarray],
             done: bool) -> Dict:
        """Update agent's knowledge using experience replay.
        
        Args:
            state: Current state components
            action: Action taken
            reward: Reward received
            next_state: Resulting state components
            done: Whether episode is done
            
        Returns:
            Dict containing training metrics
        """
        # Store transition in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Only train if we have enough samples
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert state dicts to tensors
        state_tensors = {
            k: torch.stack([
                torch.FloatTensor(s[k]) for s in states
            ]).to(self.device)
            for k in states[0].keys()
        }
        
        next_state_tensors = {
            k: torch.stack([
                torch.FloatTensor(s[k]) for s in next_states
            ]).to(self.device)
            for k in next_states[0].keys()
        }
        
        # Convert other elements to tensors
        action_indices = torch.tensor([
            valid_actions.index(a) for a, valid_actions in zip(actions, [s['valid_actions'] for s in states])
        ], device=self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(state_tensors).gather(1, action_indices.unsqueeze(1))
        
        # Compute next Q values with double DQN
        with torch.no_grad():
            # Get actions from policy network
            next_q_values = self.policy_net(next_state_tensors)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            
            # Get values from target network
            next_target_q_values = self.target_net(next_state_tensors)
            next_target_values = next_target_q_values.gather(1, next_actions).squeeze(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_target_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {"loss": loss.item()}
    
    def save(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save file
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path: str):
        """Load model from file.
        
        Args:
            path: Path to load file
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']