"""Action space handling utilities for 18XX games."""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from ...game.engine.actions import (
    BaseAction, Pass, Bid, Par, BuyShares, SellShares,
    PlaceToken, LayTile, BuyTrain, RunRoutes, Dividend,
    BuyCompany, Bankrupt
)
from ...game.engine.game.base import BaseGame

# Action type groupings
STOCK_ACTIONS = {
    'Pass': 0,
    'Bid': 1,
    'Par': 2,
    'BuyShares': 3,
    'SellShares': 4
}

OPERATING_ACTIONS = {
    'LayTile': 5,
    'PlaceToken': 6,
    'RunRoutes': 7,
    'Dividend': 8,
    'BuyTrain': 9,
    'BuyCompany': 10
}

SPECIAL_ACTIONS = {
    'Bankrupt': 11,
    'Pass': 12
}

# Parameters for different action types
MAX_PRICE = 1000
MAX_ROTATION = 6
MAX_COMPANIES = 10
MAX_SHARES = 10
MAX_ROUTES = 5
MAX_HEXES = 100
MAX_CITIES = 3

class ActionSpace:
    """Handles conversion between game actions and agent actions."""
    
    def __init__(self):
        """Initialize action space.
        
        The action space is structured as follows:
        [0]: Action type (one-hot encoded)
        [1-10]: Target company/corporation one-hot
        [11-20]: Secondary company/corporation one-hot (for exchanges)
        [21]: Price/amount (normalized)
        [22-27]: Hex coordinate encoding
        [28-30]: City slot encoding
        [31-36]: Rotation encoding
        [37-41]: Route encoding
        [42]: Dividend type
        """
        self.action_dim = 43
        
        # Combine all action types
        self.action_types = {**STOCK_ACTIONS, **OPERATING_ACTIONS, **SPECIAL_ACTIONS}
        self.reverse_action_types = {v: k for k, v in self.action_types.items()}
        
    def encode_action(self, game_action: BaseAction) -> np.ndarray:
        """Convert game action to agent action representation.
        
        Args:
            game_action: Action object from game
            
        Returns:
            Numpy array representing the action
        """
        encoding = np.zeros(self.action_dim)
        
        # Encode action type
        action_type = game_action.__class__.__name__
        if action_type in self.action_types:
            encoding[self.action_types[action_type]] = 1
            
        # Encode various action parameters based on type
        if isinstance(game_action, (BuyShares, SellShares)):
            # Encode corporation and share info
            corp_idx = game_action.corporation.index if hasattr(game_action.corporation, 'index') else 0
            encoding[1 + corp_idx] = 1
            encoding[21] = game_action.share.percent / 100.0
            
        elif isinstance(game_action, Bid):
            # Encode company and price
            company_idx = game_action.company.index if hasattr(game_action.company, 'index') else 0
            encoding[1 + company_idx] = 1
            encoding[21] = game_action.price / MAX_PRICE
            
        elif isinstance(game_action, Par):
            # Encode corporation and price
            corp_idx = game_action.corporation.index if hasattr(game_action.corporation, 'index') else 0
            encoding[1 + corp_idx] = 1
            encoding[21] = game_action.share_price.price / MAX_PRICE
            
        elif isinstance(game_action, LayTile):
            # Encode hex, rotation
            if game_action.hex:
                hex_x = game_action.hex.x if hasattr(game_action.hex, 'x') else 0
                hex_y = game_action.hex.y if hasattr(game_action.hex, 'y') else 0
                encoding[22:24] = [hex_x / MAX_HEXES, hex_y / MAX_HEXES]
            encoding[31 + (game_action.rotation % 6)] = 1
            
        elif isinstance(game_action, PlaceToken):
            # Encode hex and city
            if game_action.city and game_action.city.hex:
                hex_x = game_action.city.hex.x if hasattr(game_action.city.hex, 'x') else 0
                hex_y = game_action.city.hex.y if hasattr(game_action.city.hex, 'y') else 0
                encoding[22:24] = [hex_x / MAX_HEXES, hex_y / MAX_HEXES]
                if hasattr(game_action.city, 'index'):
                    encoding[28 + min(game_action.city.index, 2)] = 1
                    
        elif isinstance(game_action, RunRoutes):
            # Encode route information
            for i, route in enumerate(game_action.routes[:MAX_ROUTES]):
                encoding[37 + i] = 1
                
        elif isinstance(game_action, Dividend):
            # Encode dividend type
            dividend_types = {'withhold': 0, 'half': 0.5, 'full': 1.0}
            encoding[42] = dividend_types.get(game_action.kind, 0)
            
        elif isinstance(game_action, BuyTrain):
            # Encode train purchase
            encoding[21] = game_action.price / MAX_PRICE
            if game_action.train.owner:
                owner_idx = game_action.train.owner.index if hasattr(game_action.train.owner, 'index') else 0
                encoding[11 + owner_idx] = 1
                
        return encoding
    
    def decode_action(self, agent_action: np.ndarray, game: BaseGame) -> Optional[BaseAction]:
        """Convert agent action representation to game action.
        
        Args:
            agent_action: Agent's action representation
            game: Current game instance
            
        Returns:
            Action object for the game or None if invalid
        """
        # Get action type
        action_type_idx = np.argmax(agent_action[:len(self.action_types)])
        action_type = self.reverse_action_types[action_type_idx]
        
        # Get target company/corporation
        target_idx = np.argmax(agent_action[1:11])
        
        try:
            if action_type == 'Pass':
                return Pass(game.current_entity)
                
            elif action_type == 'BuyShares':
                corporation = game.corporations[target_idx]
                share = next((s for s in corporation.shares if s.owner == corporation), None)
                if share:
                    return BuyShares(game.current_entity, share, share.price)
                    
            elif action_type == 'SellShares':
                corporation = game.corporations[target_idx]
                share = next((s for s in game.current_entity.shares 
                            if s.corporation() == corporation), None)
                if share:
                    return SellShares(game.current_entity, share.to_bundle())
                    
            elif action_type == 'LayTile':
                # Decode hex coordinates
                hex_x = int(agent_action[22] * MAX_HEXES)
                hex_y = int(agent_action[23] * MAX_HEXES)
                hex = game.hex_by_coordinates(f"{chr(65+hex_x)}{hex_y+1}")
                
                # Get rotation
                rotation = np.argmax(agent_action[31:37])
                
                if hex and hex.tile:
                    return LayTile(game.current_entity, hex.tile, hex, rotation)
                    
            elif action_type == 'PlaceToken':
                # Decode hex and city
                hex_x = int(agent_action[22] * MAX_HEXES)
                hex_y = int(agent_action[23] * MAX_HEXES)
                hex = game.hex_by_coordinates(f"{chr(65+hex_x)}{hex_y+1}")
                
                city_idx = np.argmax(agent_action[28:31])
                
                if hex and hex.tile and len(hex.tile.cities) > city_idx:
                    city = hex.tile.cities[city_idx]
                    return PlaceToken(game.current_entity, city, 
                                    city.get_slot(game.current_entity))
                    
            elif action_type == 'Dividend':
                dividend_type = 'full' if agent_action[42] > 0.66 else \
                              'half' if agent_action[42] > 0.33 else 'withhold'
                return Dividend(game.current_entity, dividend_type)
                
            elif action_type == 'BuyTrain':
                price = int(agent_action[21] * MAX_PRICE)
                seller_idx = np.argmax(agent_action[11:21])
                seller = game.corporations[seller_idx]
                train = next((t for t in seller.trains), None)
                if train:
                    return BuyTrain(game.current_entity, train, price)
                    
        except (IndexError, AttributeError):
            return None
            
        return None
    
    def get_action_mask(self, valid_actions: List[BaseAction]) -> np.ndarray:
        """Create mask for valid actions.
        
        Args:
            valid_actions: List of valid game actions
            
        Returns:
            Boolean mask array (True for valid actions)
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for action in valid_actions:
            encoded = self.encode_action(action)
            mask |= encoded.astype(bool)
            
        return mask