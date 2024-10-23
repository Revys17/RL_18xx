"""State encoding utilities for 18XX games."""
from typing import Dict, List, Any, Optional
import numpy as np
from .board_utils import encode_board_state
from ...game.engine.entities import Player, Corporation, Share, Company
from ...game.engine.game.base import BaseGame

# Constants for encoding
MAX_COMPANIES = 10
MAX_SHARES_PER_COMPANY = 10
MAX_CASH = 10000  # Adjust based on game rules
MAX_TRAINS = 4

def encode_player_state(player: Player, game: BaseGame) -> np.ndarray:
    """Encode player state into numpy array.
    
    Args:
        player: Player object
        game: Game instance for context
        
    Returns array with:
    [0]: Cash (normalized)
    [1]: Total value (normalized)
    [2-11]: Share percentages for each corporation (0-100%, normalized)
    [12-21]: Presidency indicators for corporations (0/1)
    [22-31]: Private company ownership indicators
    [32-41]: Train ownership counts
    [42]: Certificate count (normalized)
    [43]: Can act indicator (0/1)
    [44]: Certificate limit headroom (normalized)
    """
    # Basic financial state
    cash = player.cash / MAX_CASH
    total_value = player.value / (MAX_CASH * 3)  # Assume max value is 3x max cash
    
    # Share holdings and presidencies
    share_holdings = np.zeros(MAX_COMPANIES)
    presidencies = np.zeros(MAX_COMPANIES)
    for i, corp in enumerate(game.corporations[:MAX_COMPANIES]):
        share_holdings[i] = player.percent_of(corp) / 100.0
        presidencies[i] = 1 if corp.owner == player else 0
    
    # Private company ownership
    private_companies = np.zeros(MAX_COMPANIES)
    for i, company in enumerate(player.companies[:MAX_COMPANIES]):
        private_companies[i] = 1
        
    # Train ownership
    train_counts = np.zeros(MAX_COMPANIES)
    for i, corp in enumerate(game.corporations[:MAX_COMPANIES]):
        if corp.owner == player:
            train_counts[i] = len(corp.trains) / MAX_TRAINS
            
    # Certificate count and limit
    cert_count = len(player.shares) / (MAX_COMPANIES * MAX_SHARES_PER_COMPANY)
    cert_limit = game.cert_limit() if hasattr(game, 'cert_limit') else MAX_COMPANIES * MAX_SHARES_PER_COMPANY
    cert_headroom = (cert_limit - len(player.shares)) / cert_limit  # Normalize to [0,1]
    
    # Action availability
    can_act = 1 if game.current_entity == player else 0
    
    return np.concatenate([
        [cash, total_value],
        share_holdings,
        presidencies,
        private_companies,
        train_counts,
        [cert_count, can_act, cert_headroom]
    ])

def encode_corporation_state(corporation: Corporation, game: BaseGame) -> np.ndarray:
    """Encode corporation state into numpy array.
    
    Returns array with:
    [0]: Share price (normalized)
    [1]: Treasury cash (normalized)
    [2]: Number of trains (normalized)
    [3]: Number of stations placed (normalized)
    [4]: Number of stations available (normalized)
    [5]: IPO status (0/1)
    [6]: Float status (0/1)
    [7-16]: Share distribution (10 slots for different holders)
    [17-21]: Recent operating history (revenue)
    [22]: Can operate (0/1)
    [23]: Has operated this round (0/1)
    [24]: President cash (normalized)
    [25-30]: Train types owned (one-hot)
    [31]: In receivership (0/1)
    [32]: Train limit headroom (normalized)
    [33]: Needs mandatory train (0/1)
    """
    # Basic financial state
    share_price = corporation.share_price.price / MAX_CASH if corporation.share_price else 0
    treasury = corporation.cash / MAX_CASH
    
    # Trains and stations
    num_trains = len(corporation.trains) / MAX_TRAINS
    num_stations_placed = len(corporation.placed_tokens()) / MAX_COMPANIES
    num_stations_available = len(corporation.unplaced_tokens()) / MAX_COMPANIES
    
    # Status flags
    ipo_status = 1 if corporation.ipoed else 0
    float_status = 1 if corporation.floated() else 0
    
    # Share distribution (top 10 holders)
    share_dist = np.zeros(10)
    for i, (holder, percent) in enumerate(sorted(
        corporation.share_holders.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]):
        share_dist[i] = percent / 100.0
        
    # Operating history (last 5 operations)
    history = np.zeros(5)
    for i, (_, revenue) in enumerate(list(corporation.operating_history.items())[-5:]):
        history[i] = revenue / MAX_CASH
    
    # Operation status
    can_operate = 1 if game.current_entity == corporation else 0
    has_operated = 1 if corporation.operated() else 0
    
    # President's cash
    president_cash = corporation.owner.cash / MAX_CASH if corporation.owner else 0
    
    # Train types
    TRAIN_TYPES = ['2', '3', '4', '5', '6', 'D']
    train_types = np.zeros(len(TRAIN_TYPES))
    for train in corporation.trains:
        if train.name in TRAIN_TYPES:
            train_types[TRAIN_TYPES.index(train.name)] = 1
            
    # Special status
    in_receivership = 1 if hasattr(corporation, 'receivership') and corporation.receivership else 0
    
    # Train limit info
    train_limit = game.train_limit() if hasattr(game, 'train_limit') else MAX_TRAINS
    train_headroom = (train_limit - len(corporation.trains)) / train_limit
    needs_train = 1 if not corporation.trains else 0
    
    return np.concatenate([
        [share_price, treasury, num_trains, num_stations_placed, num_stations_available],
        [ipo_status, float_status],
        share_dist,
        history,
        [can_operate, has_operated, president_cash],
        train_types,
        [in_receivership, train_headroom, needs_train]
    ])

def encode_market_state(game: BaseGame) -> np.ndarray:
    """Encode market state into numpy array.
    
    Returns array with:
    [0-9]: Market share prices for each corporation
    [10-19]: IPO share availability for each corporation
    [20-29]: Market pool share availability for each corporation
    [30-39]: Par prices available
    [40-49]: Corporation right to operate order
    [50]: Bank cash (normalized)
    [51]: Phase number (normalized)
    [52]: Operating round number
    [53]: Stock round number
    [54]: Train export pending (0/1)
    [55]: Phase change triggers (0/1)
    [56]: Emergency fund raising active (0/1)
    """
    # Share prices
    prices = np.zeros(MAX_COMPANIES)
    ipo_shares = np.zeros(MAX_COMPANIES)
    pool_shares = np.zeros(MAX_COMPANIES)
    par_prices = np.zeros(MAX_COMPANIES)
    operation_order = np.zeros(MAX_COMPANIES)
    
    for i, corp in enumerate(game.corporations[:MAX_COMPANIES]):
        # Market price
        prices[i] = corp.share_price.price / MAX_CASH if corp.share_price else 0
        
        # Share availability
        ipo_shares[i] = len([s for s in corp.shares if s.owner == corp.ipo_owner]) / MAX_SHARES_PER_COMPANY
        pool_shares[i] = len([s for s in corp.shares if s.owner == game.share_pool]) / MAX_SHARES_PER_COMPANY
        
        # Par price
        par_prices[i] = corp.par_price().price / MAX_CASH if corp.par_price() else 0
        
        # Operation order
        if hasattr(game, 'operation_order'):
            try:
                order_idx = game.operation_order.index(corp)
                operation_order[i] = (MAX_COMPANIES - order_idx) / MAX_COMPANIES
            except ValueError:
                pass
    
    # Game state indicators
    bank_cash = game.bank.cash / (MAX_CASH * 10)  # Bank typically has more money
    
    phase_num = 0
    if game.phase:
        try:
            phase_num = int(game.phase.name.replace('D', '6'))
        except ValueError:
            phase_num = 1
    phase_normalized = phase_num / 6  # Assuming max phase is 6
    
    operating_round = game.round_counter % 3 if hasattr(game, 'round_counter') else 0
    stock_round = game.stock_round_number if hasattr(game, 'stock_round_number') else 0
    
    # Special conditions
    train_export = 1 if hasattr(game, 'train_export_pending') and game.train_export_pending else 0
    phase_change_pending = 1 if hasattr(game, 'phase_change_triggered') and game.phase_change_triggered else 0
    emergency_active = 1 if hasattr(game, 'emergency_fund_raising') and game.emergency_fund_raising else 0
    
    return np.concatenate([
        prices,
        ipo_shares,
        pool_shares,
        par_prices,
        operation_order,
        [bank_cash, phase_normalized, operating_round/3, stock_round/3],
        [train_export, phase_change_pending, emergency_active]
    ])

def encode_game_phase(game: BaseGame) -> np.ndarray:
    """Encode game phase information.
    
    Returns array with:
    [0]: Round number (normalized)
    [1]: Phase number (normalized)
    [2-5]: One-hot encoding of round type
    [6]: Priority deal position (normalized)
    [7]: Current player number (normalized)
    [8]: Turns until end of round
    [9]: Emergency fund raising active (0/1)
    [10-13]: Operating step encoding
    """
    # Round info
    round_num = game.round_counter if hasattr(game, 'round_counter') else 0
    max_rounds = 30  # Typical game length
    normalized_round = round_num / max_rounds
    
    phase_num = 0
    if game.phase:
        try:
            phase_num = int(game.phase.name.replace('D', '6'))
        except ValueError:
            phase_num = 1
    phase_normalized = phase_num / 6
    
    # Round type one-hot
    round_types = ['Stock', 'Operating', 'Auction', 'Draft']
    round_type = np.zeros(len(round_types))
    current_type = game.round.__class__.__name__.replace('Round', '')
    if current_type in round_types:
        round_type[round_types.index(current_type)] = 1
    
    # Priority deal
    priority_pos = 0
    if hasattr(game, 'priority_deal_player'):
        try:
            priority_pos = game.players.index(game.priority_deal_player) / len(game.players)
        except (ValueError, AttributeError):
            pass
    
    # Current player
    current_player = 0
    if game.current_entity and game.current_entity in game.players:
        current_player = game.players.index(game.current_entity) / len(game.players)
    
    # Round progress
    turns_remaining = 0
    if hasattr(game.round, 'turns_remaining'):
        turns_remaining = game.round.turns_remaining / len(game.players)
    
    # Emergency status
    emergency_active = 1 if hasattr(game, 'emergency_fund_raising') and game.emergency_fund_raising else 0
    
    # Operating step
    operating_steps = ['Trains', 'Track', 'Tokens', 'Routes']
    step_encoding = np.zeros(len(operating_steps))
    if hasattr(game.round, 'current_step'):
        current_step = game.round.current_step.__class__.__name__.replace('Step', '')
        if current_step in operating_steps:
            step_encoding[operating_steps.index(current_step)] = 1
    
    return np.concatenate([
        [normalized_round, phase_normalized],
        round_type,
        [priority_pos, current_player, turns_remaining, emergency_active],
        step_encoding
    ])

def encode_full_state(game: BaseGame) -> Dict[str, np.ndarray]:
    """Encode complete game state.
    
    Args:
        game: Game instance
    
    Returns:
        Dictionary containing:
        - 'board': Complete board state encoding
        - 'players': Encoded states for all players
        - 'corporations': Encoded states for all corporations
        - 'market': Market state encoding
        - 'phase': Game phase encoding
    """
    # Get board state (includes hexes, routes, and connectivity)
    board_state = encode_board_state(game)
    
    # Encode player states
    player_states = np.stack([
        encode_player_state(player, game) 
        for player in game.players
    ])
    
    # Encode corporation states
    corporation_states = np.stack([
        encode_corporation_state(corp, game)
        for corp in game.corporations
    ])
    
    # Encode market state
    market_state = encode_market_state(game)
    
    # Encode game phase
    phase_state = encode_game_phase(game)
    
    return {
        'board': board_state,
        'players': player_states,
        'corporations': corporation_states,
        'market': market_state,
        'phase': phase_state
    }