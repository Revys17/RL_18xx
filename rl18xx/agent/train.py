from typing import Dict, List
import argparse
import json
import os
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm

from .envs.game_env import Game18xxEnv
from .models.dqn_agent import DQNAgent
from .utils.reward_utils import RewardCalculator

def train(config: Dict):
    """Train the agent.
    
    Args:
        config: Training configuration
    """
    # Create environment
    env = Game18xxEnv(num_players=config['num_players'])
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq']
    )
    
    # Training loop
    rewards_history = []
    
    for episode in tqdm(range(config['num_episodes'])):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Learn from experience
            metrics = agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        
        # Log progress
        if (episode + 1) % config['log_frequency'] == 0:
            avg_reward = np.mean(rewards_history[-config['log_frequency']:])
            print(f"Episode {episode + 1}/{config['num_episodes']}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if (episode + 1) % config['save_frequency'] == 0:
            save_path = os.path.join(
                config['save_dir'],
                f"checkpoint_episode_{episode + 1}.pt"
            )
            agent.save(save_path)
    
    # Save final model
    final_path = os.path.join(config['save_dir'], "final_model.pt")
    agent.save(final_path)

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train 18XX RL agent')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['save_dir'], f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    config['save_dir'] = save_dir
    
    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train agent
    train(config)

if __name__ == "__main__":
    main()