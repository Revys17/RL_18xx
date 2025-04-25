from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game.engine.round import WaterfallAuction
from rl18xx.game.engine.entities import Player, Corporation, Company, Bank, Depot, Train  # Import entities
from rl18xx.game.engine.game.title.g1830 import Game as Game_1830, Entities as Entities_1830, Map as Map_1830


from torch import Tensor, from_numpy
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set
import logging
import time

LOGGER = logging.getLogger(__name__)

# --- Static Definitions based on g1830.py ---
# Use definitions from the game file for consistency
G1830_TRAINS = Game_1830.TRAINS
G1830_CORPORATIONS = Entities_1830.CORPORATIONS
G1830_COMPANIES = Entities_1830.COMPANIES
G1830_PHASES = Game_1830.PHASES  # Access phase definitions
G1830_MARKET = Game_1830.MARKET  # Access market layout
G1830_BANK_CASH = Game_1830.BANK_CASH
G1830_CERT_LIMIT = Game_1830.CERT_LIMIT  # Dict based on player count
G1830_STARTING_CASH = Game_1830.STARTING_CASH  # Dict based on player count

# Create ordered lists and mappings from static data
G1830_TRAIN_COUNT = {t["name"]: t["num"] for t in G1830_TRAINS}
TRAIN_TYPES_ORDERED = sorted([t["name"] for t in G1830_TRAINS])
CORPORATION_IDS_ORDERED = sorted([c["sym"] for c in G1830_CORPORATIONS])
PRIVATE_IDS_ORDERED = sorted([c["sym"] for c in G1830_COMPANIES])
PHASE_NAMES_ORDERED = sorted([p["name"] for p in G1830_PHASES])
TILE_IDS_ORDERED = sorted(list(Map_1830.TILES.keys()))
HEX_COORDS_ORDERED = sorted(
    hex_coord 
    for hex_type in Map_1830.HEXES.values() 
    for hex_coords in hex_type.keys() 
    for hex_coord in hex_coords
)
NUM_ROTATIONS = 6

# Max values for normalization/sizing
MAX_SHARE_PRICE = max(price for row in G1830_MARKET for price in row if price is not None)
MAX_PRIVATE_REVENUE = max(c["revenue"] for c in G1830_COMPANIES)
MAX_CORP_TOKENS = max(len(c.get("tokens", [0])) for c in G1830_CORPORATIONS)
MAX_TILE_SLOTS = # TODO: Calculate how many possible token slots there are on the map

# Round type mapping
ROUND_TYPE_MAP = {
    "Stock": 0,
    "Operating": 1,
    "Auction": 2,  # Assuming this is the class name used
    # Add other round types if necessary
}
MAX_ROUND_TYPE_IDX = max(ROUND_TYPE_MAP.values())


class Encoder_1830:
    # --- Dynamically set in __init__ based on player count ---
    # These will be filled based on the actual game instance
    num_players: int = 0
    starting_cash: int = 0
    cert_limit: int = 0
    # ---

    # --- Constants derived from static data ---
    NUM_CORPORATIONS = len(CORPORATION_IDS_ORDERED)
    NUM_PRIVATES = len(PRIVATE_IDS_ORDERED)
    NUM_TRAIN_TYPES = len(TRAIN_TYPES_ORDERED)
    NUM_HEXES = len(HEX_COORDS_ORDERED)
    NUM_TILE_IDS = len(TILE_IDS_ORDERED)
    NUM_PHASES = len(PHASE_NAMES_ORDERED)
    # ---

    # Pre-calculate ENCODING_SIZE structure (values depend on num_players)
    # This structure helps verify offsets during encoding
    ENCODING_STRUCTURE = {
        # --- Game State ---
        "active_entity": ("num_players + NUM_CORPORATIONS", 0),
        "active_president": ("num_players", 0),
        "round_type": (1, 0),
        "game_phase": (1, 0),
        "priority_deal_player": ("num_players", 0),
        "bank_cash": (1, 0),
        "player_certs_remaining": ("num_players", 0),
        # --- Player States ---
        "player_cash": ("num_players", 0),
        "player_shares": ("num_players * NUM_CORPORATIONS", 0),
        "player_privates": ("num_players * NUM_PRIVATES", 0),
        # --- Private Company States ---
        "private_revenue": (NUM_PRIVATES, 0),
        "private_corp_owner": ("NUM_PRIVATES * NUM_CORPORATIONS", 0),
        # --- Corporation States ---
        "corp_floated": (NUM_CORPORATIONS, 0),
        "corp_cash": (NUM_CORPORATIONS, 0),
        "corp_trains": (NUM_CORPORATIONS * NUM_TRAIN_TYPES, 0),
        "corp_tokens_remaining": (NUM_CORPORATIONS, 0),
        # --- Market States ---
        "corp_share_price": (NUM_CORPORATIONS, 0),
        "corp_ipo_shares": (NUM_CORPORATIONS, 0),
        "corp_market_shares": (NUM_CORPORATIONS, 0),
        "corp_market_zone": (NUM_CORPORATIONS, 0),
        "depot_trains": (NUM_TRAIN_TYPES, 0),
        "market_pool_trains": (NUM_TRAIN_TYPES, 0),
        "depot_tiles": (NUM_TILE_IDS, 0),
        # --- Map State ---
        "hex_tile_id": (NUM_HEXES * NUM_TILE_IDS, 0),
        "hex_rotation": (NUM_HEXES * NUM_ROTATIONS, 0),
        "hex_tokens": (MAX_TILE_SLOTS * NUM_CORPORATIONS, 0),
        # --- Auction State (Conditional) ---
        "auction_bids": ("NUM_PRIVATES * num_players", 0),
        "auction_min_bid": (NUM_PRIVATES, 0),
        "auction_available": (NUM_PRIVATES, 0),
        "auction_face_value": (NUM_PRIVATES, 0),
    }
    ENCODING_SIZE: int = 0

    def __init__(self):
        LOGGER.debug("Initializing Encoder_1830")
        # Mappings are initialized once per game instance in _initialize_entity_maps
        self.player_id_to_idx: Dict[str, int] = {}
        self.corp_id_to_idx: Dict[str, int] = {sym: i for i, sym in enumerate(CORPORATION_IDS_ORDERED)}
        self.private_id_to_idx: Dict[str, int] = {sym: i for i, sym in enumerate(PRIVATE_IDS_ORDERED)}
        self.train_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(TRAIN_TYPES_ORDERED)}
        self.hex_coord_to_idx: Dict[str, int] = {coord: i for i, coord in enumerate(HEX_COORDS_ORDERED)}
        self.tile_id_to_idx: Dict[str, int] = {name: i for i, name in enumerate(TILE_IDS_ORDERED)}
        self.tile_id_to_idx["plain"] = len(self.tile_id_to_idx)
        self.tile_id_to_idx["city"] = len(self.tile_id_to_idx)
        self.tile_id_to_idx["town"] = len(self.tile_id_to_idx)
        self.phase_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(PHASE_NAMES_ORDERED)}

        self.initialized_for_player_count = 0  # Track if size/settings are set

    def _calculate_encoding_size(self, num_players: int) -> int:
        """Calculates the total encoding size based on player count."""
        total_size = 0
        for key, (size_expr, _) in self.ENCODING_STRUCTURE.items():
            # Evaluate the size expression string with current constants
            size = eval(
                str(size_expr),
                {
                    "num_players": num_players,
                    "NUM_CORPORATIONS": self.NUM_CORPORATIONS,
                    "NUM_PRIVATES": self.NUM_PRIVATES,
                    "NUM_TRAIN_TYPES": self.NUM_TRAIN_TYPES,
                    "NUM_HEXES": self.NUM_HEXES,
                    "NUM_TILE_IDS": self.NUM_TILE_IDS,
                    "NUM_PHASES": self.NUM_PHASES,
                    "MAX_TILE_SLOTS": MAX_TILE_SLOTS,
                },
            )
            self.ENCODING_STRUCTURE[key] = (size_expr, size)
            total_size += size
        return total_size

    def _initialize_game_specifics(self, game: BaseGame):
        """Initializes settings based on the specific game instance (player count)."""
        player_count = len(game.players)
        if self.initialized_for_player_count == player_count:
            return

        LOGGER.info(f"Initializing encoder specifics for {player_count} players.")
        self.num_players = player_count
        # Use game methods/constants if available, otherwise fall back to static dicts
        self.starting_cash = getattr(game, "STARTING_CASH", G1830_STARTING_CASH).get(player_count, 0)
        self.cert_limit = getattr(game, "CERT_LIMIT", G1830_CERT_LIMIT).get(player_count, 0)

        # Calculate dynamic ENCODING_SIZE
        self.ENCODING_SIZE = self._calculate_encoding_size(player_count)
        self.initialized_for_player_count = player_count
        LOGGER.info(f"Calculated ENCODING_SIZE: {self.ENCODING_SIZE}")
        LOGGER.debug(f"Encoding Structure (Sizes): { {k: v[1] for k, v in self.ENCODING_STRUCTURE.items()} }")

    def _initialize_player_map(self, players: List[Player]):
        """Initializes or confirms the player ID to index mapping."""
        # Sort players by ID for consistent indexing
        sorted_players = sorted(players, key=lambda p: p.id)
        current_map = {p.id: i for i, p in enumerate(sorted_players)}

        if not self.player_id_to_idx:
            self.player_id_to_idx = current_map
            LOGGER.info(f"Initialized player map: {self.player_id_to_idx}")
        elif self.player_id_to_idx != current_map:
            LOGGER.error(
                "Player list mismatch or order change during encoding. Existing: %s, New: %s",
                self.player_id_to_idx,
                current_map,
            )
            raise ValueError("Player list mismatch during encoding.")
        # else: Player map is consistent

    def _get_section_size(self, section_name: str) -> int:
        """Helper to get the calculated size of a specific encoding section."""
        size = self.ENCODING_STRUCTURE[section_name][1]
        if size == 0 and section_name not in [
            "active_entity",
            "player_cash",
            "player_shares",
            "player_privates",
            "auction_bids",
        ]:
            # Only warn if size is 0 for sections not dependent on player count (which is set in _initialize_game_specifics)
            LOGGER.warning(f"Encoding section '{section_name}' has calculated size 0.")
        return size

    # --- Helper for offset checking ---
    def check_offset(self, section_name, offset, section_start_offset):
        expected_size = self._get_section_size(section_name)
        actual_size = offset - section_start_offset
        if actual_size != expected_size:
            LOGGER.error(
                f"Offset mismatch for section '{section_name}': Expected {expected_size}, Actual {actual_size}. Current total offset: {offset}"
            )
            # Decide whether to raise an error or just log
            # raise AssertionError(f"Offset mismatch for section '{section_name}'")
        else:
            LOGGER.debug(f"Section '{section_name}' OK (Size: {actual_size}, Offset: {offset})")
        return offset  # Reset for next section

    def encode(self, game: BaseGame) -> Tensor:
        start_time = time.perf_counter()
        LOGGER.debug("Starting game state encoding.")

        # --- Initialization ---
        try:
            # Initialize size/settings based on player count if not already done
            self._initialize_game_specifics(game)
            # Initialize or verify player mapping
            self._initialize_player_map(game.players)
        except Exception as e:
            LOGGER.exception("Failed during encoder initialization.")
            raise e

        if self.ENCODING_SIZE <= 0:
            raise ValueError("ENCODING_SIZE not calculated correctly.")

        encoding = np.zeros(self.ENCODING_SIZE, dtype=np.float32)
        offset = 0
        section_start_offset = 0  # For verification

        # --- Section 1: Active Entity Indicator (Player or Corporation) ---
        section_name = "active_entity"
        active_entities = game.round.active_entities  # Get current entity/entities
        if active_entities:
            active_entity = active_entities[0]
            if isinstance(active_entity, Player) and active_entity.id in self.player_id_to_idx:
                active_idx = self.player_id_to_idx[active_entity.id]
                encoding[offset + active_idx] = 1.0
            elif isinstance(active_entity, Corporation) and active_entity.id in self.corp_id_to_idx:
                # Offset index by num_players for corporations
                active_idx = self.num_players + self.corp_id_to_idx[active_entity.id]
                encoding[offset + active_idx] = 1.0
            else:
                LOGGER.warning(
                    f"Active entity '{active_entity.id}' ({type(active_entity)}) not found in expected maps."
                )
        else:
            LOGGER.warning("No active entity found in game round.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 2: Active President (if Corp active) ---
        section_name = "active_president"
        president = None
        if active_entities and isinstance(active_entities[0], Corporation):
            active_players = game.active_players()
            if active_players and isinstance(active_players[0], Player):
                president = active_players[0]

        if president and president.id in self.player_id_to_idx:
            president_idx = self.player_id_to_idx[president.id]
            encoding[offset + president_idx] = 1.0
            LOGGER.debug(
                f"Encoded active president: Player {president.id} (Index {president_idx}) for active corp {active_entities[0].id}"
            )
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 3: Current Round Type ---
        section_name = "round_type"
        round_name = game.round.__class__.__name__
        if round_name in ROUND_TYPE_MAP:
            encoding[offset] = ROUND_TYPE_MAP[round_name] / MAX_ROUND_TYPE_IDX  # Normalize
        else:
            raise ValueError(f"Unknown round type: {round_name}")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 4: Current Game Phase ---
        section_name = "game_phase"
        phase_name = game.phase.name
        if phase_name in self.phase_name_to_idx:
            encoding[offset] = self.phase_name_to_idx[phase_name] / (self.NUM_PHASES - 1)  # Normalize
        else:
            raise ValueError(f"Unknown phase name: {phase_name}")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 5: Priority Deal Player ---
        section_name = "priority_deal_player"
        priority_player = game.priority_deal_player()
        if priority_player.id in self.player_id_to_idx:
            priority_player_idx = self.player_id_to_idx[priority_player.id]
            encoding[offset + priority_player_idx] = 1.0
            LOGGER.debug(f"Encoded priority deal player: ID={priority_player.id}, Index={priority_player_idx}")
        else:
            raise ValueError(f"Priority deal player {priority_player.id} not found in player_id_to_idx.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 6: Bank Cash ---
        section_name = "bank_cash"
        encoding[offset] = game.bank.cash / G1830_BANK_CASH if G1830_BANK_CASH > 0 else 0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 7: Certificate Limit ---
        section_name = "player_certs_remaining"
        initial_limit = self.cert_limit
        if initial_limit > 0:
            for player in game.players:
                if player.id in self.player_id_to_idx:
                    player_idx = self.player_id_to_idx[player.id]
                    # Calculate remaining: initial_limit - num_certs_owned
                    # Assumes player.num_certs exists and is correct
                    num_certs = player.num_certs if hasattr(player, 'num_certs') else initial_limit # Fallback if attr missing
                    remaining = initial_limit - num_certs
                    encoding[offset + player_idx] = remaining / initial_limit
                    LOGGER.debug(f"Encoded certs remaining for Player {player.id}: {remaining}/{initial_limit} (Normalized: {encoding[offset + player_idx]:.3f})")
                else:
                     LOGGER.warning(f"Player {player.id} not in map for cert limit encoding.")
        else:
             LOGGER.warning("Initial cert limit is 0, cannot normalize remaining certs.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 8: Player Cash ---
        section_name = "player_cash"
        for player in sorted(game.players, key=lambda p: p.id):  # Ensure consistent order
            if player.id in self.player_id_to_idx:
                player_idx = self.player_id_to_idx[player.id]
                encoding[offset + player_idx] = player.cash / self.starting_cash if self.starting_cash > 0 else 0
            else:
                raise ValueError(f"Player {player.id} not found in player_id_to_idx.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 9: Player Share Ownership (%) ---
        section_name = "player_shares"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            for player in sorted(game.players, key=lambda p: p.id):
                if player.id in self.player_id_to_idx:
                    player_idx = self.player_id_to_idx[player.id]
                    player_corp_offset = offset + player_idx * self.NUM_CORPORATIONS + corp_idx
                    percent = player.percent_of(corporation)
                    encoding[player_corp_offset] = percent / 100.0
                else:
                    raise ValueError(f"Player {player.id} not found in player_id_to_idx.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 10: Player Private Ownership ---
        section_name = "player_privates"
        for priv_id, priv_idx in self.private_id_to_idx.items():
            company = game.company_by_id(priv_id)
            if company:
                if isinstance(company.owner, Player) and company.owner.id in self.player_id_to_idx:
                    owner_idx = self.player_id_to_idx[company.owner.id]
                    player_priv_offset = offset + owner_idx * self.NUM_PRIVATES + priv_idx
                    encoding[player_priv_offset] = 1.0
            else:
                raise ValueError(f"Private company {priv_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 11: Private Revenue ---
        section_name = "private_revenue"
        max_rev = MAX_PRIVATE_REVENUE  # Use constant defined above
        for priv_id, priv_idx in self.private_id_to_idx.items():
            company = game.company_by_id(priv_id)
            if company:
                encoding[offset + priv_idx] = company.revenue / max_rev if max_rev > 0 else 0
            else:
                raise ValueError(f"Private company {priv_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 12: Private Owning Corporation ---
        section_name = "private_corp_owner"
        for priv_id, priv_idx in self.private_id_to_idx.items():
            company = game.company_by_id(priv_id)
            if company and isinstance(company.owner, Corporation) and company.owner.id in self.corp_id_to_idx:
                owner_corp_idx = self.corp_id_to_idx[company.owner.id]
                # Calculate the flat index for the one-hot encoding
                flat_index = offset + priv_idx * self.NUM_CORPORATIONS + owner_corp_idx
                encoding[flat_index] = 1.0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 13: Corporation Floated Status ---
        section_name = "corp_floated"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if corporation:
                if corporation.floated():
                    encoding[offset + corp_idx] = 1.0
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 14: Corporation Cash ---
        section_name = "corp_cash"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if corporation:
                encoding[offset + corp_idx] = float(corporation.cash) / self.starting_cash
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 15: Corporation Trains ---
        section_name = "corp_trains"
        train_counts = {name: 0 for name in TRAIN_TYPES_ORDERED}  # Temp dict for counts
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if corporation:
                train_counts = {name: 0 for name in TRAIN_TYPES_ORDERED}
                for train in corporation.trains:
                    if train.name in train_counts:
                        train_counts[train.name] += 1
                    else:
                        LOGGER.warning(f"Corp {corp_id} has unknown train '{train.name}'")

                for train_name, train_idx in self.train_name_to_idx.items():
                    corp_train_offset = offset + corp_idx * self.NUM_TRAIN_TYPES + train_idx
                    initial_count = next((t["num"] for t in G1830_TRAINS if t["name"] == train_name), 0)
                    encoding[corp_train_offset] = train_counts[train_name] / initial_count if initial_count > 0 else 0
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 16: Corporation Tokens Remaining ---
        section_name = "corp_tokens_remaining"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if corporation:
                unused_token_count = len(corporation.unplaced_tokens())
                initial_tokens = len(corporation.tokens)
                encoding[offset + corp_idx] = unused_token_count / initial_tokens if initial_tokens > 0 else 0
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 17: Corporation Share Price ---
        section_name = "corp_share_price"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if corporation:
                if corporation.share_price:
                    price = corporation.share_price.price
                    encoding[offset + corp_idx] = price / MAX_SHARE_PRICE if MAX_SHARE_PRICE > 0 else 0
                else:
                    encoding[offset + corp_idx] = 0  # Not on market yet
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 18: Corporation IPO Shares Remaining ---
        section_name = "corp_ipo_shares"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if corporation:
                # Use game.share_pool.num_shares(corporation) - need to check if it distinguishes IPO/Market
                # Assuming corporation.num_ipo_shares exists:
                ipo_shares = corporation.num_ipo_shares()
                total_ipo = corporation.total_shares  # Usually 10
                encoding[offset + corp_idx] = ipo_shares / total_ipo
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 19: Corporation Market Shares Available ---
        section_name = "corp_market_shares"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if corporation:
                market_shares = corporation.num_market_shares()
                total_market = corporation.total_shares
                encoding[offset + corp_idx] = market_shares / total_market
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 20: Corporation Market Zone (Color) ---
        section_name = "corp_market_zone"
        # grey tiles are regular and have no type
        zone_map = {"no_cert_limit": 1, "unlimited": 2, "multiple_buy": 3}
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            zone_idx = 0
            if corporation:
                if corporation.share_price:
                    type = corporation.share_price.type
                    if type in zone_map:
                        zone_idx = zone_map[type]
            else:
                raise ValueError(f"Corporation {corp_id} not found in game.")

            corp_zone_offset = offset + corp_idx * 3 + zone_idx  # 3 zones: Y, G, B/Other
            encoding[corp_zone_offset] = 1.0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 21: Trains in Market/Depot ---
        section_name = "depot_trains"
        # Access available trains via game.depot
        depot_counts = {name: 0 for name in TRAIN_TYPES_ORDERED}
        if game.depot:
            for train in game.depot.trains:
                if train.name in depot_counts:
                    depot_counts[train.name] += 1

        for train_name, train_idx in self.train_name_to_idx.items():
            initial_count = G1830_TRAIN_COUNT[train_name]
            encoding[offset + train_idx] = depot_counts[train_name] / initial_count if initial_count > 0 else 0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 22: Market Pool Trains (Discarded) ---
        section_name = "market_pool_trains"
        market_train_counts = {name: 0 for name in self.train_name_to_idx}
        for train in game.depot.discarded:
            if hasattr(train, "name") and train.name in market_train_counts:
                market_train_counts[train.name] += 1
            else:
                train_name = getattr(train, "name", "UNKNOWN")
                LOGGER.warning(f"Discarded train '{train_name}' not in TRAIN_TYPES_ORDERED or has no name.")

        for train_name, train_idx in self.train_name_to_idx.items():
            initial_count = G1830_TRAIN_COUNT[train_name]
            encoding[offset + train_idx] = market_train_counts[train_name] / initial_count if initial_count > 0 else 0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 23: Tiles Available in Market ---
        section_name = "depot_tiles"
        used_tile_counts = {name: sum(1 for tile in game.tiles if tile.name == name) for name in TILE_IDS_ORDERED}
        for tile_name, tile_idx in self.tile_id_to_idx.items():
            encoding[offset + tile_idx] = float(used_tile_counts.get(tile_name, 0)) / Map_1830.TILES.get(tile_name, 1)
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # TODO: Figure out a better way to do the hex and tile encoding
        # --- Section 24: Hex Tile ID ---
        # One hot encoded
        section_name = "hex_tile_id"
        for hex_coord, hex_idx in self.hex_coord_to_idx.items():
            hex_obj = game.hex_by_id(hex_coord)
            tile_name = hex_obj.tile.name if hex_obj and hex_obj.tile else None
            if tile_name and tile_name in self.tile_id_to_idx:
                encoding[offset + hex_idx * len(self.tile_id_to_idx) + self.tile_id_to_idx[tile_name]] = 1.0
            elif tile_name and tile_name == hex_obj.name:
                encoding[offset + hex_idx * len(self.tile_id_to_idx) + self.tile_id_to_idx["plain"]] = 1.0
            else:
                raise ValueError(f"Tile '{tile_name}' on hex '{hex_coord}' not in TILE_IDS_ORDERED.")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 25: Hex Rotation ---
        section_name = "hex_rotation"
        for hex_coord, hex_idx in self.hex_coord_to_idx.items():
            hex_obj = game.hex_by_id(hex_coord)
            if hex_obj and hex_obj.tile:
                # Normalize rotation 0-5? Or just use value? Use value directly.
                encoding[offset + hex_idx] = hex_obj.tile.rotation
            # else: remains 0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 26: Hex Tokens ---
        section_name = "hex_tokens"
        # One-hot encoding: For each hex, for each slot, mark the owning corp's index
        for hex_coord, hex_idx in self.hex_coord_to_idx.items():
            hex_obj = game.hex_by_id(hex_coord)
            if hex_obj and hex_obj.tile:
                # Iterate through cities/towns on the tile
                slot_idx = 0
                for city in hex_obj.tile.cities:  # Assumes cities are the tokenable locations
                    # Check city.tokens (list of Token objects)
                    if city.tokens:
                        for token in city.tokens:
                            if slot_idx < MAX_TILE_SLOTS and token and token.corporation.id in self.corp_id_to_idx:
                                corp_idx = self.corp_id_to_idx[token.corporation.id]
                                # Calculate offset: hex_base + slot_offset + corp_offset
                                token_offset = (
                                    offset
                                    + hex_idx * MAX_TILE_SLOTS * self.NUM_CORPORATIONS
                                    + slot_idx * self.NUM_CORPORATIONS
                                    + corp_idx
                                )
                                encoding[token_offset] = 1.0
                            slot_idx += 1
                    else:
                        # If no tokens, advance slot index based on city.slots
                        slot_idx += city.slots if hasattr(city, "slots") else 1  # Assume 1 slot if attr missing

                    if slot_idx >= MAX_TILE_SLOTS:
                        break  # Stop if max slots reached
            # else: remains 0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Sections A1-A4: Auction-Specific Details ---
        is_auction = isinstance(game.round, WaterfallAuction) or (
            hasattr(game.round, "active_step") and isinstance(game.round.active_step(), WaterfallAuction)
        )

        if is_auction:
            auction_step = game.round if isinstance(game.round, WaterfallAuction) else game.round.active_step()
            LOGGER.debug("Encoding WaterfallAuction details.")

            # A1: Bids
            section_name = "auction_bids"
            if hasattr(auction_step, "bids"):  # Check if bids attribute exists
                for priv_id, priv_idx in self.private_id_to_idx.items():
                    company = game.company_by_id(priv_id)
                    if not company:
                        continue
                    # auction_step.bids is likely Dict[Company, List[Bid]]
                    for bid in auction_step.bids.get(company, []):
                        player = bid.entity
                        price = bid.price
                        if isinstance(player, Player) and player.id in self.player_id_to_idx:
                            player_idx = self.player_id_to_idx[player.id]
                            bid_offset = offset + priv_idx * self.num_players + player_idx
                            encoding[bid_offset] = price / self.starting_cash if self.starting_cash > 0 else 0
            else:
                LOGGER.warning("Auction step does not have 'bids' attribute.")
            offset += self._get_section_size(section_name)
            section_start_offset = self.check_offset(section_name, offset, section_start_offset)

            # A2: Min Bid/Price
            section_name = "auction_min_bid"
            for priv_id, priv_idx in self.private_id_to_idx.items():
                company = game.company_by_id(priv_id)
                if not company:
                    continue
                if company.owner:
                    encoding[offset + priv_idx] = -1.0  # Indicate owned
                elif hasattr(auction_step, "min_bid"):
                    min_bid_val = auction_step.min_bid(company)  # Call method
                    encoding[offset + priv_idx] = min_bid_val / self.starting_cash if self.starting_cash > 0 else 0
                else:
                    LOGGER.warning("Auction step does not have 'min_bid' method.")
                    encoding[offset + priv_idx] = -1.0  # Indicate unknown
            offset += self._get_section_size(section_name)
            section_start_offset = self.check_offset(section_name, offset, section_start_offset)

            # A3: Available for Purchase
            section_name = "auction_available"
            available_company = auction_step.companies[0]
            for priv_id, priv_idx in self.private_id_to_idx.items():
                company = game.company_by_id(priv_id)
                if company == available_company:
                    encoding[offset + priv_idx] = 1.0
            offset += self._get_section_size(section_name)
            section_start_offset = self.check_offset(section_name, offset, section_start_offset)

            # A4: Face Value
            section_name = "auction_face_value"
            for priv_id, priv_idx in self.private_id_to_idx.items():
                company = game.company_by_id(priv_id)
                if company:
                    encoding[offset + priv_idx] = company.value / self.starting_cash if self.starting_cash > 0 else 0
                else:
                    raise ValueError(f"Company {priv_id} not found in game.")
            offset += self._get_section_size(section_name)
            section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        else:
            LOGGER.debug("Not an Auction round/step, skipping auction-specific encoding sections.")
            # Advance offset by the size of all auction sections
            offset += self._get_section_size("auction_bids")
            offset += self._get_section_size("auction_min_bid")
            offset += self._get_section_size("auction_available")
            offset += self._get_section_size("auction_face_value")
            # No need for check_offset here as we skipped the sections

        # --- Final Offset Check ---
        LOGGER.debug(f"Final offset after encoding: {offset}")
        if offset != self.ENCODING_SIZE:
            LOGGER.error("CRITICAL: Final offset %d != calculated ENCODING_SIZE %d", offset, self.ENCODING_SIZE)
            # Log details about the mismatch
            raise AssertionError(f"Encoding size mismatch: Final offset {offset} != expected size {self.ENCODING_SIZE}")
        else:
            LOGGER.info(f"Encoding size check PASSED (Offset: {offset}, Expected: {self.ENCODING_SIZE})")

        # Convert numpy array to torch tensor and add batch dimension
        # Replace NaN with 0 before converting to tensor
        encoding = np.nan_to_num(encoding, nan=0.0, posinf=0.0, neginf=0.0)
        tensor_encoding = from_numpy(encoding).unsqueeze(0)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        LOGGER.debug(f"Encoding finished in {duration_ms:.3f} ms. Output shape: {tensor_encoding.shape}")
        return tensor_encoding
