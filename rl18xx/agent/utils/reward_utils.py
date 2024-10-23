"""Reward calculation utilities for 18XX games."""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from ...game.engine.entities import Player, Corporation, Share, Company
from ...game.engine.game.base import BaseGame
from ...game.engine.actions import (
    BaseAction, BuyShares, SellShares, LayTile, PlaceToken,
    RunRoutes, Dividend, BuyTrain, Bankrupt
)

class RewardCalculator:
    """Calculates rewards for game states and transitions."""
    
    def __init__(self, 
                 net_worth_weight: float = 1.0,
                 relative_position_weight: float = 0.5,
                 dividend_weight: float = 0.3,
                 route_weight: float = 0.2,
                 connectivity_weight: float = 0.2,
                 bankruptcy_penalty: float = -1.0):
        """Initialize reward calculator.
        
        Args:
            net_worth_weight: Weight for net worth changes
            relative_position_weight: Weight for relative position among players
            dividend_weight: Weight for dividend payments
            route_weight: Weight for route improvements
            connectivity_weight: Weight for network connectivity
            bankruptcy_penalty: Penalty for going bankrupt
        """
        self.net_worth_weight = net_worth_weight
        self.relative_position_weight = relative_position_weight
        self.dividend_weight = dividend_weight
        self.route_weight = route_weight
        self.connectivity_weight = connectivity_weight
        self.bankruptcy_penalty = bankruptcy_penalty
        
        # State tracking
        self.previous_net_worth = {}
        self.previous_routes = {}
        self.previous_connectivity = {}
        
    def calculate_net_worth(self, player: Player, game: BaseGame) -> float:
        """Calculate player's net worth.
        
        Args:
            player: Player object
            game: Current game state
            
        Returns:
            Float value of player's net worth
        """
        net_worth = player.cash
        
        # Add share values
        for corporation, shares in player.shares_by_corporation.items():
            if corporation.share_price:
                share_value = corporation.share_price.price * sum(share.percent for share in shares) / 100.0
                net_worth += share_value
        
        # Add private company values
        for company in player.companies:
            net_worth += company.value
            
        # Add value of corporations where player is president
        for corporation in game.corporations:
            if corporation.owner == player:
                # Add corporation's cash
                net_worth += corporation.cash
                # Add train values (at half price as they depreciate)
                net_worth += sum(train.price / 2 for train in corporation.trains)
                
        return net_worth
        
    def calculate_relative_position(self, 
                                 player: Player, 
                                 game: BaseGame) -> float:
        """Calculate player's position relative to others.
        
        Args:
            player: Player object
            game: Current game state
            
        Returns:
            Float value representing relative position (-1 to 1)
        """
        player_worth = self.calculate_net_worth(player, game)
        other_worths = [
            self.calculate_net_worth(p, game)
            for p in game.players
            if p != player
        ]
        
        if not other_worths:
            return 0.0
            
        # Calculate relative position
        avg_worth = np.mean(other_worths)
        max_worth = max(other_worths)
        min_worth = min(other_worths)
        
        if max_worth == min_worth:
            return 0.0
            
        # Normalize to [-1, 1]
        relative_pos = (player_worth - avg_worth) / (max_worth - min_worth)
        return np.clip(relative_pos, -1.0, 1.0)
        
    def calculate_route_value(self, 
                            corporation: Corporation,
                            game: BaseGame) -> float:
        """Calculate value of corporation's route network.
        
        Args:
            corporation: Corporation object
            game: Current game state
            
        Returns:
            Float value representing route network value
        """
        if not corporation.operated():
            return 0.0
            
        # Get most recent operating history
        latest_revenue = list(corporation.operating_history.values())[-1]
        
        # Add potential future revenue from tokens
        potential_revenue = 0
        for token in corporation.placed_tokens():
            if token.city:
                potential_revenue += token.city.revenue
                
        return latest_revenue + (potential_revenue * 0.5)  # Weight potential lower than actual
        
    def calculate_connectivity(self,
                             corporation: Corporation,
                             game: BaseGame) -> float:
        """Calculate network connectivity value.
        
        Args:
            corporation: Corporation object
            game: Current game state
            
        Returns:
            Float value representing network connectivity
        """
        if not corporation.operated():
            return 0.0
            
        # Count connected city pairs
        connected_pairs = 0
        visited = set()
        
        for token1 in corporation.placed_tokens():
            if not token1.city:
                continue
            for token2 in corporation.placed_tokens():
                if token1 == token2 or not token2.city:
                    continue
                    
                pair = tuple(sorted([token1.city.id, token2.city.id]))
                if pair in visited:
                    continue
                    
                # Check if cities are connected by corporation's network
                if game.graph.connected(token1.city, token2.city, corporation):
                    connected_pairs += 1
                visited.add(pair)
                
        return connected_pairs
        
    def calculate_action_reward(self,
                              action: BaseAction,
                              game: BaseGame) -> float:
        """Calculate immediate reward for an action.
        
        Args:
            action: Action taken
            game: Current game state
            
        Returns:
            Float value representing immediate reward
        """
        immediate_reward = 0.0
        
        if isinstance(action, Dividend):
            # Reward based on dividend amount and type
            if action.kind == 'full':
                immediate_reward += 1.0
            elif action.kind == 'half':
                immediate_reward += 0.5
                
        elif isinstance(action, LayTile):
            # Small reward for expanding network
            immediate_reward += 0.1
            
        elif isinstance(action, PlaceToken):
            # Reward based on city value
            if action.city:
                immediate_reward += action.city.revenue / 100.0
                
        elif isinstance(action, RunRoutes):
            # Reward based on route revenue
            if action.routes:
                total_revenue = sum(route.revenue for route in action.routes)
                immediate_reward += total_revenue / 1000.0
                
        elif isinstance(action, Bankrupt):
            immediate_reward += self.bankruptcy_penalty
            
        return immediate_reward
        
    def calculate_reward(self, 
                        player: Player,
                        game: BaseGame,
                        action: Optional[BaseAction] = None,
                        is_terminal: bool = False) -> float:
        """Calculate total reward for current state.
        
        Args:
            player: Player object
            game: Current game state
            action: Action taken (if any)
            is_terminal: Whether this is the final state
            
        Returns:
            Float value representing the reward
        """
        # Calculate net worth change
        current_net_worth = self.calculate_net_worth(player, game)
        net_worth_change = current_net_worth - self.previous_net_worth.get(player, current_net_worth)
        self.previous_net_worth[player] = current_net_worth
        
        # Calculate relative position
        relative_pos = self.calculate_relative_position(player, game)
        
        # Calculate reward components
        net_worth_reward = net_worth_change * self.net_worth_weight
        position_reward = relative_pos * self.relative_position_weight
        
        # Calculate route and connectivity rewards for corporations
        route_reward = 0.0
        connectivity_reward = 0.0
        for corporation in game.corporations:
            if corporation.owner == player:
                # Route value
                current_route_value = self.calculate_route_value(corporation, game)
                previous_route_value = self.previous_routes.get(corporation, current_route_value)
                route_reward += (current_route_value - previous_route_value) * self.route_weight
                self.previous_routes[corporation] = current_route_value
                
                # Connectivity
                current_connectivity = self.calculate_connectivity(corporation, game)
                previous_connectivity = self.previous_connectivity.get(corporation, current_connectivity)
                connectivity_reward += (current_connectivity - previous_connectivity) * self.connectivity_weight
                self.previous_connectivity[corporation] = current_connectivity
        
        # Add immediate action reward if applicable
        action_reward = self.calculate_action_reward(action, game) if action else 0.0
        
        # Terminal reward
        terminal_reward = 0.0
        if is_terminal:
            # Calculate final ranking
            final_worths = [(p, self.calculate_net_worth(p, game)) for p in game.players]
            final_worths.sort(key=lambda x: x[1], reverse=True)
            player_rank = next(i for i, (p, _) in enumerate(final_worths) if p == player)
            
            # Reward based on final position
            num_players = len(game.players)
            terminal_reward = (num_players - player_rank) / num_players
        
        return sum([
            net_worth_reward,
            position_reward,
            route_reward,
            connectivity_reward,
            action_reward,
            terminal_reward
        ])