"""Board state encoding utilities for 18XX games."""
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
from ...game.engine.graph import Hex, Tile, Path, City, Town, Edge, Token, Offboard
from ...game.engine.entities import Corporation
from ...game.engine.game.base import BaseGame

# Constants for encoding
TILE_COLORS = ['yellow', 'green', 'brown', 'gray', 'red']
TRACK_GAUGES = ['normal', 'narrow', 'dual']
MAX_CORPORATIONS = 10
MAX_REVENUE = 100
PHASE_REVENUES = {
    '2': 1.0,    # Base revenue
    '3': 1.2,    # 20% increase
    '4': 1.5,    # 50% increase
    '5': 2.0,    # Double
    '6': 2.5,    # 2.5x
    'D': 3.0     # Triple
}

def encode_track_gauge(path: Path) -> np.ndarray:
    """Encode track gauge information.
    
    Args:
        path: The path to encode
        
    Returns:
        One-hot encoding of track gauge [normal, narrow, dual]
    """
    gauge_encoding = np.zeros(len(TRACK_GAUGES))
    if path.track in TRACK_GAUGES:
        gauge_encoding[TRACK_GAUGES.index(path.track)] = 1
    return gauge_encoding

def calculate_phase_revenue(base_revenue: int, phase: str) -> float:
    """Calculate revenue considering game phase.
    
    Args:
        base_revenue: Base revenue of the location
        phase: Current game phase
        
    Returns:
        Modified revenue value
    """
    multiplier = PHASE_REVENUES.get(phase, 1.0)
    return base_revenue * multiplier

def encode_special_abilities(hex: Hex) -> np.ndarray:
    """Encode special abilities and restrictions of a hex.
    
    Returns array with:
    [0]: Private company access only (0/1)
    [1]: Phase restriction (encoded phase number / max phase)
    [2]: Must be upgraded in order (0/1)
    [3]: Cannot be upgraded (0/1)
    [4]: Special revenue condition exists (0/1)
    """
    encoding = np.zeros(5)
    
    if hex.tile:
        # Check for private company restrictions
        encoding[0] = 1 if any(blocker.is_private() for blocker in hex.tile.blockers) else 0
        
        # Phase restrictions
        if hasattr(hex.tile, 'phase') and hex.tile.phase:
            try:
                phase_num = int(hex.tile.phase.replace('D', '6'))
                encoding[1] = phase_num / 6  # Normalize to [0,1]
            except ValueError:
                encoding[1] = 0
                
        # Upgrade restrictions
        encoding[2] = 1 if hasattr(hex.tile, 'must_upgrade_in_order') and hex.tile.must_upgrade_in_order else 0
        encoding[3] = 1 if hasattr(hex.tile, 'cannot_upgrade') and hex.tile.cannot_upgrade else 0
        
        # Special revenue conditions
        encoding[4] = 1 if any(hasattr(p, 'revenue_condition') for p in hex.tile.parts) else 0
        
    return encoding

def encode_offboard_location(offboard: Offboard) -> np.ndarray:
    """Encode an offboard location.
    
    Returns array with:
    [0]: Is offboard (1)
    [1]: Base revenue normalized
    [2]: Phase revenue multiplier applies (0/1)
    [3-8]: Connections array (6 directions)
    [9]: Location category encoded (normalized value)
    """
    encoding = np.zeros(10)
    
    encoding[0] = 1  # Is offboard
    encoding[1] = offboard.revenue / MAX_REVENUE if hasattr(offboard, 'revenue') else 0
    encoding[2] = 1 if hasattr(offboard, 'phase_revenue') else 0
    
    # Encode connections
    if hasattr(offboard, 'exits'):
        for exit_num in offboard.exits:
            if 0 <= exit_num < 6:  # Valid direction
                encoding[3 + exit_num] = 1
                
    # Encode category if available (e.g., "East", "West", etc.)
    if hasattr(offboard, 'category'):
        categories = ['East', 'West', 'North', 'South', 'Mountain', 'Port']
        if offboard.category in categories:
            encoding[9] = (categories.index(offboard.category) + 1) / len(categories)
    
    return encoding

def get_valid_upgrades(hex: Hex, game: BaseGame) -> Set[int]:
    """Get valid upgrade tile numbers for a hex.
    
    Args:
        hex: The hex to check
        game: The game instance
        
    Returns:
        Set of valid upgrade tile numbers
    """
    valid_upgrades = set()
    
    if not hex.tile:
        return valid_upgrades
        
    current_color = hex.tile.color
    if current_color not in TILE_COLORS:
        return valid_upgrades
        
    current_idx = TILE_COLORS.index(current_color)
    if current_idx >= len(TILE_COLORS) - 1:
        return valid_upgrades  # No more upgrades possible
        
    next_color = TILE_COLORS[current_idx + 1]
    
    # Get all tiles of the next color
    potential_upgrades = [t for t in game.tiles.values() 
                         if t.color == next_color and t.number > 0]
    
    for upgrade in potential_upgrades:
        # Check if upgrade is legal
        if (hasattr(hex.tile, 'upgrades') and 
            upgrade.name in hex.tile.upgrades and
            not any(blocker.blocks_lay() for blocker in upgrade.blockers)):
            valid_upgrades.add(upgrade.id)
            
    return valid_upgrades

def encode_hex_state(hex: Hex, game: BaseGame) -> np.ndarray:
    """Encode a single hex's state into a numpy array.
    
    Args:
        hex: The hex to encode
        game: The game instance for context
    
    Returns array with:
    [0-4]: One-hot encoding of tile color
    [5-7]: Track gauge encoding
    [8]: Has station
    [9]: Number of cities
    [10]: Number of towns
    [11-16]: Connections array (6 directions)
    [17]: Has token
    [18-27]: One-hot encoding of corporation token
    [28]: Base revenue (normalized)
    [29]: Phase-adjusted revenue (normalized)
    [30-34]: Special abilities encoding
    [35-44]: Valid upgrades indicator (10 most common upgrades)
    [45-54]: Off-board encoding
    """
    # Color encoding
    color_encoding = np.zeros(len(TILE_COLORS))
    if hex.tile and hex.tile.color in TILE_COLORS:
        color_encoding[TILE_COLORS.index(hex.tile.color)] = 1
    
    # Track gauge (combine all paths)
    gauge_encoding = np.zeros(len(TRACK_GAUGES))
    if hex.tile:
        for path in hex.tile.paths:
            gauge_encoding |= encode_track_gauge(path)
    
    # Cities and towns
    has_station = 1 if hex.tile and hex.tile.cities else 0
    num_cities = len(hex.tile.cities) if hex.tile else 0
    num_towns = len(hex.tile.towns) if hex.tile else 0
    
    # Connections with gauge information
    connections = np.zeros(6)
    if hex.tile:
        for path in hex.tile.paths:
            for edge in path.edges:
                if edge.num < 6:
                    connections[edge.num] = 1
    
    # Token information
    has_token = 0
    token_corporation = np.zeros(MAX_CORPORATIONS)
    if hex.tile:
        for city in hex.tile.cities:
            for token in city.tokens:
                if token and token.corporation:
                    has_token = 1
                    corp_idx = game.corporations.index(token.corporation)
                    if corp_idx < MAX_CORPORATIONS:
                        token_corporation[corp_idx] = 1
    
    # Revenue calculations
    base_revenue = 0
    if hex.tile:
        for revenue_location in hex.tile.cities + hex.tile.towns:
            base_revenue = max(base_revenue, revenue_location.revenue)
    normalized_base_revenue = base_revenue / MAX_REVENUE
    
    # Phase-adjusted revenue
    phase = game.phase.name if game.phase else '2'
    normalized_phase_revenue = calculate_phase_revenue(base_revenue, phase) / (MAX_REVENUE * 3)  # Max multiplier is 3
    
    # Special abilities
    special_abilities = encode_special_abilities(hex)
    
    # Valid upgrades
    valid_upgrades = get_valid_upgrades(hex, game)
    upgrade_encoding = np.zeros(10)  # Encode top 10 most common upgrades
    for i, upgrade_id in enumerate(sorted(valid_upgrades)[:10]):
        upgrade_encoding[i] = 1
    
    # Off-board encoding
    offboard_encoding = np.zeros(10)
    if hex.tile and hex.tile.offboards:
        offboard_encoding = encode_offboard_location(hex.tile.offboards[0])
    
    return np.concatenate([
        color_encoding,           # 5 features
        gauge_encoding,          # 3 features
        [has_station],          # 1 feature
        [num_cities],           # 1 feature
        [num_towns],            # 1 feature
        connections,            # 6 features
        [has_token],            # 1 feature
        token_corporation,      # 10 features
        [normalized_base_revenue],    # 1 feature
        [normalized_phase_revenue],   # 1 feature
        special_abilities,      # 5 features
        upgrade_encoding,       # 10 features
        offboard_encoding       # 10 features
    ])

def encode_route_value(route: List[Hex], game: BaseGame) -> np.ndarray:
    """Encode a potential route's value and characteristics.
    
    Returns array with:
    [0]: Total base revenue
    [1]: Phase-adjusted revenue
    [2]: Number of stops
    [3]: Number of different corporations' tokens used
    [4]: Requires specific train type (0/1)
    """
    encoding = np.zeros(5)
    
    if not route:
        return encoding
        
    # Calculate revenue
    base_revenue = 0
    corps_used = set()
    num_stops = 0
    
    for hex in route:
        if hex.tile:
            # Add revenue from cities and towns
            for revenue_loc in hex.tile.cities + hex.tile.towns:
                base_revenue += revenue_loc.revenue
                num_stops += 1
                
            # Track corporations whose tokens are used
            for city in hex.tile.cities:
                for token in city.tokens:
                    if token and token.corporation:
                        corps_used.add(token.corporation)
    
    # Normalize revenues
    encoding[0] = base_revenue / (MAX_REVENUE * len(route))  # Normalize by route length
    encoding[1] = calculate_phase_revenue(base_revenue, game.phase.name) / (MAX_REVENUE * 3 * len(route))
    
    # Other characteristics
    encoding[2] = num_stops / len(route)  # Normalize by route length
    encoding[3] = len(corps_used) / MAX_CORPORATIONS
    
    # Train type requirement (if applicable)
    encoding[4] = 1 if any(hex.tile and any(hasattr(p, 'train_requirement') for p in hex.tile.parts) 
                          for hex in route) else 0
    
    return encoding

def encode_board_state(game: BaseGame) -> Dict[str, np.ndarray]:
    """Encode the entire board state including route information.
    
    Args:
        game: The game instance
        
    Returns:
        Dictionary containing:
        - 'hexes': 3D numpy array with shape (rows, cols, features_per_hex)
        - 'routes': List of encoded potential routes
        - 'connectivity': Sparse matrix of hex connectivity
    """
    # Determine board dimensions
    all_hexes = game.hexes
    max_row = max(hex.row for hex in all_hexes)
    max_col = max(hex.column for hex in all_hexes)
    
    # Features per hex from encode_hex_state
    features_per_hex = 54  # This should match encode_hex_state output size
    
    # Initialize board tensor
    board = np.zeros((max_row + 1, max_col + 1, features_per_hex))
    
    # Fill in hex states
    for hex in all_hexes:
        if not hex.empty:
            board[hex.row, hex.column] = encode_hex_state(hex, game)
    
    # Create connectivity matrix (sparse representation)
    connectivity = {}
    for hex in all_hexes:
        if hex.tile and hex.tile.paths:
            connections = []
            for path in hex.tile.paths:
                for edge in path.edges:
                    neighbor = hex.neighbors.get(edge.num)
                    if neighbor:
                        connections.append((neighbor.row, neighbor.column))
            if connections:
                connectivity[(hex.row, hex.column)] = connections
    
    # Find and encode potential routes
    routes = []
    # This would need game-specific logic to identify valid routes
    # For now, we'll return an empty list
    
    return {
        'hexes': board,
        'connectivity': connectivity,
        'routes': routes
    }