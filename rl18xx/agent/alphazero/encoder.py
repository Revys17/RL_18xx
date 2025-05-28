from rl18xx.game.engine.game.base import BaseGame
from rl18xx.game.engine.round import WaterfallAuction
from rl18xx.game.engine.entities import Player, Corporation, Company, Bank, Depot, Train
from rl18xx.game.engine.game.title.g1830 import Game as Game_1830, Entities as Entities_1830, Map as Map_1830
from rl18xx.game.engine.graph import Hex, Tile, Edge, City
from rl18xx.agent.alphazero.singleton import Singleton
from torch import Tensor, from_numpy
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
import time
import re

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
CORPORATION_IDS_ORDERED = [c["sym"] for c in G1830_CORPORATIONS]
PRIVATE_IDS_ORDERED = [c["sym"] for c in G1830_COMPANIES]
PHASE_NAMES_ORDERED = [p["name"] for p in G1830_PHASES]
TILE_IDS_ORDERED = sorted(list(Map_1830.TILES.keys()), key=lambda x: int(x))
HEX_COORDS_ORDERED = sorted(
    hex_coord for hex_type in Map_1830.HEXES.values() for hex_coords in hex_type.keys() for hex_coord in hex_coords
)
NUM_ROTATIONS = 6

# Max values for normalization/sizing
MAX_SHARE_PRICE = 350
MAX_PRIVATE_REVENUE = max(c["revenue"] for c in G1830_COMPANIES)
MAX_CORP_TOKENS = max(len(c.get("tokens", [0])) for c in G1830_CORPORATIONS)
MAX_HEX_REVENUE = 80
NUM_TILE_EDGES = 6
NUM_PORT_PAIRS = 15
MAX_LAY_COST = 120

# Round type mapping
ROUND_TYPE_MAP = {
    "Stock": 0,
    "Operating": 1,
    "Auction": 2,  # Assuming this is the class name used
    # Add other round types if necessary
}
MAX_ROUND_TYPE_IDX = max(ROUND_TYPE_MAP.values())


class Encoder_1830(metaclass=Singleton):
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

    # Pre-calculate GAME_STATE_ENCODING_STRUCTURE (values depend on num_players)
    # This structure helps verify offsets during encoding
    GAME_STATE_ENCODING_STRUCTURE = {
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
        # --- Private Company States ---
        "private_ownership": ("NUM_PRIVATES * (num_players + NUM_CORPORATIONS)", 0),
        "private_revenue": (NUM_PRIVATES, 0),
        # --- Corporation States ---
        "corp_floated": (NUM_CORPORATIONS, 0),
        "corp_cash": (NUM_CORPORATIONS, 0),
        "corp_trains": (NUM_CORPORATIONS * NUM_TRAIN_TYPES, 0),
        "corp_tokens_remaining": (NUM_CORPORATIONS, 0),
        # --- Market States ---
        "corp_share_price": (NUM_CORPORATIONS * 2, 0),
        "corp_shares": (2 * NUM_CORPORATIONS, 0),
        "corp_market_zone": (4 * NUM_CORPORATIONS, 0),
        "depot_trains": (NUM_TRAIN_TYPES, 0),
        "market_pool_trains": (NUM_TRAIN_TYPES, 0),
        "depot_tiles": (NUM_TILE_IDS, 0),
        # --- Auction State (Conditional) ---
        "auction_bids": ("NUM_PRIVATES * num_players", 0),
        "auction_min_bid": (NUM_PRIVATES, 0),
        "auction_available": (NUM_PRIVATES, 0),
        "auction_face_value": (NUM_PRIVATES, 0),
    }
    GAME_STATE_ENCODING_SIZE: int = 0

    def __init__(self):
        LOGGER.debug("Initializing Encoder_1830")
        # Mappings
        self.player_id_to_idx: Dict[str, int] = {}
        self.corp_id_to_idx: Dict[str, int] = {sym: i for i, sym in enumerate(CORPORATION_IDS_ORDERED)}
        self.private_id_to_idx: Dict[str, int] = {sym: i for i, sym in enumerate(PRIVATE_IDS_ORDERED)}
        self.train_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(TRAIN_TYPES_ORDERED)}
        self.hex_coord_to_idx: Dict[str, int] = {coord: i for i, coord in enumerate(HEX_COORDS_ORDERED)}
        self.tile_id_to_idx: Dict[str, int] = {name: i for i, name in enumerate(TILE_IDS_ORDERED)}
        self.phase_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(PHASE_NAMES_ORDERED)}

        self.initialized_for_player_count = 0
        self._calculate_map_node_features()
        self.initialized = False

    def _calculate_map_node_features(self):
        self.map_node_features = [
            "revenue",  # Normalized revenue value
            "is_city",  # Boolean flag
            "is_oo",  # Boolean flag
            "is_town",  # Boolean flag
            "is_offboard",  # Boolean flag
            "upgrade_cost",  # Normalized upgrade cost
            "rotation",  # Raw tile rotation (0-5)
            # Token presence (NUM_CORPORATIONS features, 1 if corp has token here)
            *[f"token_{corp_id}_revenue_{i}" for corp_id in CORPORATION_IDS_ORDERED for i in range(2)],
            # Internal Connectivity (port-to-port, e.g., connects_0_3)
            *[f"connects_{i}_{j}" for i in range(NUM_TILE_EDGES) for j in range(i)],
            # Port-to-Revenue Connectivity (e.g., port_2_connects_revenue)
            *[
                f"port_{i}_connects_revenue_{j}" for i in range(NUM_TILE_EDGES) for j in range(2)
            ],  # Need to handle OO cities/towns
        ]

        self.num_map_node_features = len(self.map_node_features)
        LOGGER.info(f"Calculated number of map node features: {self.num_map_node_features}")

    def _precompute_adjacency(self, game: BaseGame):
        """Calculates the static adjacency list for the hex grid."""
        LOGGER.debug("Precomputing hex grid adjacency")
        edges = []
        for hex in game.hexes:
            for direction, neighbor in hex.all_neighbors.items():
                if hex.id not in self.hex_coord_to_idx or neighbor.id not in self.hex_coord_to_idx:
                    raise ValueError(f"Hex {hex.id} or neighbor {neighbor.id} not found in hex_coord_to_idx")

                i = self.hex_coord_to_idx[hex.id]
                j = self.hex_coord_to_idx[neighbor.id]
                edges.append([i, j, direction])
                edges.append([j, i, (direction + 3) % 6])

        if not edges:
            raise ValueError("Adjacency calculation resulted in no edges. Check neighbor logic.")

        edge_np = np.unique(np.array(edges), axis=0).T
        edge_index = from_numpy(edge_np)
        self.base_edge_index = edge_index[0:2, :].long()
        self.base_edge_attributes = edge_index[2, :].long()
        # LOGGER.debug(f"Precomputed edge_index with shape: {self.edge_index.shape}")

    def _calculate_encoding_size(self, num_players: int) -> int:
        """Calculates the total encoding size for the flat game state vector."""
        total_size = 0
        temp_structure = self.GAME_STATE_ENCODING_STRUCTURE.copy()
        eval_globals = {
            "num_players": num_players,
            "NUM_CORPORATIONS": self.NUM_CORPORATIONS,
            "NUM_PRIVATES": self.NUM_PRIVATES,
            "NUM_TRAIN_TYPES": self.NUM_TRAIN_TYPES,
            "NUM_TILE_IDS": self.NUM_TILE_IDS,
        }
        for key, (size_expr, _) in temp_structure.items():
            try:
                size = eval(str(size_expr), eval_globals)
                temp_structure[key] = (size_expr, size)
                total_size += size
            except Exception as e:
                LOGGER.error(f"Error evaluating size for key '{key}' with expression '{size_expr}': {e}")
                raise
        self.GAME_STATE_ENCODING_STRUCTURE = temp_structure
        self.GAME_STATE_ENCODING_SIZE = total_size
        return total_size

    def _initialize_game_specifics(self, game: BaseGame):
        """Initializes settings based on the specific game instance (player count)."""
        player_count = len(game.players)
        LOGGER.info(f"Initializing encoder specifics for {player_count} players.")
        self.num_players = player_count
        self.starting_cash = getattr(game, "STARTING_CASH", G1830_STARTING_CASH).get(player_count, 0)
        self.cert_limit = getattr(game, "CERT_LIMIT", G1830_CERT_LIMIT).get(player_count, 0)
        self.ENCODING_SIZE = self._calculate_encoding_size(player_count)
        self.initialized_for_player_count = player_count
        LOGGER.info(f"Calculated ENCODING_SIZE: {self.ENCODING_SIZE}")
        LOGGER.debug(
            f"Encoding Structure (Sizes): { {k: v[1] for k, v in self.GAME_STATE_ENCODING_STRUCTURE.items()} }"
        )

    def _initialize_player_map(self, players: List[Player]):
        """Initializes or confirms the player ID to index mapping."""
        # Sort players by ID for consistent indexing
        sorted_players = sorted(players, key=lambda p: p.id)
        self.player_id_to_idx = {p.id: i for i, p in enumerate(sorted_players)}

    def _get_section_size(self, section_name: str) -> int:
        """Helper to get the calculated size of a specific encoding section."""
        size = self.GAME_STATE_ENCODING_STRUCTURE[section_name][1]
        if size == 0:
            raise ValueError(f"Encoding section '{section_name}' has calculated size 0.")
        return size

    # --- Helper for offset checking ---
    def check_offset(self, section_name, offset, section_start_offset):
        expected_size = self._get_section_size(section_name)
        actual_size = offset - section_start_offset
        if actual_size != expected_size:
            LOGGER.error(
                f"Offset mismatch for section '{section_name}': Expected {expected_size}, Actual {actual_size}. Current total offset: {offset}"
            )
            raise ValueError(f"Offset mismatch for section '{section_name}'")

        # LOGGER.debug(f"Section '{section_name}' OK (Size: {actual_size}, Offset: {offset})")
        return offset

    def initialize(self, game: BaseGame):
        if not self.initialized:
            try:
                self._initialize_game_specifics(game)
                self._precompute_adjacency(game)
                self._initialize_player_map(game.players)
            except Exception as e:
                LOGGER.exception("Failed during encoder initialization.")
                raise e

            if self.ENCODING_SIZE <= 0:
                raise ValueError("ENCODING_SIZE not calculated correctly.")

            self.initialized = True

    def encode(self, game: BaseGame) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        start_time = time.perf_counter()
        # LOGGER.debug("Starting encoding.")

        self.initialize(game)

        game_state_tensor = self.encode_game_state(game)
        node_features_tensor = self.get_node_features(game)
        base_edge_index_tensor, base_edge_attributes_tensor = self.get_edge_index(game)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        LOGGER.debug(f"Encoding finished in {duration_ms:.3f} ms.")
        # LOGGER.debug(f"  Flat state shape: {game_state_tensor.shape}")
        # LOGGER.debug(f"  Node features shape: {node_features_tensor.shape}")
        # LOGGER.debug(f"  Edge index shape: {edge_index_tensor.shape}")
        return game_state_tensor, node_features_tensor, base_edge_index_tensor, base_edge_attributes_tensor

    def encode_game_state(self, game: BaseGame) -> Tensor:
        start_time = time.perf_counter()
        # LOGGER.debug("Starting game state encoding.")
        state_encoding = np.zeros(self.ENCODING_SIZE, dtype=np.float32)
        offset = 0
        section_start_offset = 0

        # --- Section 1: Active Entity Indicator (Player or Corporation) ---
        section_name = "active_entity"
        if not game.finished:
            active_entities = game.round.active_entities
            if not active_entities:
                raise ValueError("No active entity found in unfinished game.")

            active_entity = active_entities[0]
            if isinstance(active_entity, Player) and active_entity.id in self.player_id_to_idx:
                active_idx = self.player_id_to_idx[active_entity.id]
                state_encoding[offset + active_idx] = 1.0
            elif isinstance(active_entity, Corporation) and active_entity.id in self.corp_id_to_idx:
                active_idx = self.num_players + self.corp_id_to_idx[active_entity.id]
                state_encoding[offset + active_idx] = 1.0
            elif isinstance(active_entity, Company):
                owner_id = game.company_by_id(active_entity.id).owner.id
                if owner_id in self.player_id_to_idx:
                    active_idx = self.player_id_to_idx[owner_id]
                    state_encoding[offset + active_idx] = 1.0
                elif owner_id in self.corp_id_to_idx:
                    active_idx = self.num_players + self.corp_id_to_idx[owner_id]
                    state_encoding[offset + active_idx] = 1.0
            else:
                # TODO: Double check this new logic
                raise ValueError(
                    f"Active entity '{active_entity.id}' ({active_entity.__class__.__name__}) not found in expected maps."
                )
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 2: Active President (if Corp active) ---
        section_name = "active_president"
        if not game.finished:
            president = None
            if isinstance(active_entities[0], Corporation):
                active_players = game.active_players()
                if not active_players:
                    raise ValueError("No active players found in game.")
                president = active_players[0]

                if president.id not in self.player_id_to_idx:
                    raise ValueError(f"Active president '{president.id}' not found in player_id_to_idx.")

                president_idx = self.player_id_to_idx[president.id]
                state_encoding[offset + president_idx] = 1.0
                # LOGGER.debug(
                #    f"Encoded active president: Player {president.id} (Index {president_idx}) for active corp {active_entities[0].id}"
                # )
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 3: Current Round Type ---
        section_name = "round_type"
        round_name = game.round.__class__.__name__
        if round_name not in ROUND_TYPE_MAP:
            raise ValueError(f"Unknown round type: {round_name}")
        state_encoding[offset] = float(ROUND_TYPE_MAP[round_name]) / MAX_ROUND_TYPE_IDX
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 4: Current Game Phase ---
        section_name = "game_phase"
        phase_name = game.phase.name
        if phase_name not in self.phase_name_to_idx:
            raise ValueError(f"Unknown phase name: {phase_name}")
        state_encoding[offset] = self.phase_name_to_idx[phase_name] / (self.NUM_PHASES - 1)
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 5: Priority Deal Player ---
        section_name = "priority_deal_player"
        priority_player = game.priority_deal_player()
        if priority_player.id not in self.player_id_to_idx:
            raise ValueError(f"Priority deal player '{priority_player.id}' not found in player_id_to_idx.")
        priority_player_idx = self.player_id_to_idx[priority_player.id]
        state_encoding[offset + priority_player_idx] = 1.0
        # LOGGER.debug(f"Encoded priority deal player: ID={priority_player.id}, Index={priority_player_idx}")
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 6: Bank Cash ---
        section_name = "bank_cash"
        state_encoding[offset] = float(game.bank.cash) / G1830_BANK_CASH
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 7: Certificate Limit ---
        section_name = "player_certs_remaining"
        initial_limit = self.cert_limit
        if initial_limit <= 0:
            raise ValueError("Initial cert limit is 0, cannot normalize remaining certs.")
        for player in game.players:
            if player.id not in self.player_id_to_idx:
                raise ValueError(f"Player {player.id} not found in player_id_to_idx.")
            player_idx = self.player_id_to_idx[player.id]
            remaining = initial_limit - game.num_certs(player)
            state_encoding[offset + player_idx] = float(remaining) / initial_limit
            # LOGGER.debug(
            #    f"Encoded certs remaining for Player {player.id}: {remaining}/{initial_limit} (Normalized: {state_encoding[offset + player_idx]:.3f})"
            # )
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 8: Player Cash ---
        section_name = "player_cash"
        for player in sorted(game.players, key=lambda p: p.id):  # Ensure consistent order
            if player.id not in self.player_id_to_idx:
                raise ValueError(f"Player {player.id} not found in player_id_to_idx.")
            player_idx = self.player_id_to_idx[player.id]
            state_encoding[offset + player_idx] = float(player.cash) / self.starting_cash
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 9: Player Share Ownership (%) ---
        section_name = "player_shares"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            for player in sorted(game.players, key=lambda p: p.id):
                if player.id not in self.player_id_to_idx:
                    raise ValueError(f"Player {player.id} not found in player_id_to_idx.")
                player_idx = self.player_id_to_idx[player.id]
                player_corp_offset = offset + player_idx * self.NUM_CORPORATIONS + corp_idx
                percent = player.percent_of(corporation)
                state_encoding[player_corp_offset] = percent / 100.0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 10: Private Ownership ---
        section_name = "private_ownership"
        for priv_id, priv_idx in self.private_id_to_idx.items():
            company = game.company_by_id(priv_id)
            if not company:
                raise ValueError(f"Private company {priv_id} not found in game.")

            owner = company.owner
            owner_offset = -1  # Sentinel value

            if isinstance(owner, Player) and owner.id in self.player_id_to_idx:
                # Owner is a player
                player_idx = self.player_id_to_idx[owner.id]
                owner_offset = player_idx  # Player indices are 0 to num_players-1
            elif isinstance(owner, Corporation) and owner.id in self.corp_id_to_idx:
                # Owner is a corporation
                corp_idx = self.corp_id_to_idx[owner.id]
                # Corporation indices start after player indices
                owner_offset = self.num_players + corp_idx

            if owner_offset == -1:
                continue
            state_encoding[offset + priv_idx * (self.num_players + self.NUM_CORPORATIONS) + owner_offset] = 1.0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 11: Private Revenue ---
        section_name = "private_revenue"
        max_rev = MAX_PRIVATE_REVENUE
        for priv_id, priv_idx in self.private_id_to_idx.items():
            company = game.company_by_id(priv_id)
            if not company:
                raise ValueError(f"Private company {priv_id} not found in game.")
            state_encoding[offset + priv_idx] = float(company.revenue) / max_rev
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 12: Corporation Floated Status ---
        section_name = "corp_floated"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            state_encoding[offset + corp_idx] = float(corporation.floated())
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 13: Corporation Cash ---
        section_name = "corp_cash"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            state_encoding[offset + corp_idx] = float(corporation.cash) / self.starting_cash
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 14: Corporation Trains ---
        section_name = "corp_trains"
        train_counts = {name: 0 for name in TRAIN_TYPES_ORDERED}  # Temp dict for counts
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            train_counts = {name: 0 for name in TRAIN_TYPES_ORDERED}
            for train in corporation.trains:
                if train.name not in train_counts:
                    raise ValueError(f"Train {train.name} not found in TRAIN_TYPES_ORDERED")
                train_counts[train.name] += 1

            for train_name, train_idx in self.train_name_to_idx.items():
                corp_train_offset = offset + corp_idx * self.NUM_TRAIN_TYPES + train_idx
                initial_count = next((t["num"] for t in G1830_TRAINS if t["name"] == train_name), 0)
                state_encoding[corp_train_offset] = float(train_counts[train_name]) / initial_count
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 15: Corporation Tokens Remaining ---
        section_name = "corp_tokens_remaining"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            unused_token_count = len(corporation.unplaced_tokens())
            initial_tokens = len(corporation.tokens)
            state_encoding[offset + corp_idx] = float(unused_token_count) / initial_tokens
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 16: Corporation Share Price ---
        section_name = "corp_share_price"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            if not corporation.share_price:
                # Not on market yet
                continue
            ipo_price = corporation.par_price().price
            if ipo_price is None:
                ipo_price = 0
            price = corporation.share_price.price
            state_encoding[offset + corp_idx * 2] = float(ipo_price) / MAX_SHARE_PRICE
            state_encoding[offset + corp_idx * 2 + 1] = float(price) / MAX_SHARE_PRICE
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 17: Corporation Shares Remaining ---
        section_name = "corp_shares"
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            ipo_shares = corporation.num_ipo_shares()
            market_shares = corporation.num_market_shares()
            total_ipo = corporation.total_shares
            state_encoding[offset + 2 * corp_idx] = float(ipo_shares) / total_ipo
            state_encoding[offset + 2 * corp_idx + 1] = float(market_shares) / total_ipo
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 18: Corporation Market Zone (Color) ---
        section_name = "corp_market_zone"
        # grey tiles are regular and have no type
        zone_map = {"no_cert_limit": 1, "unlimited": 2, "multiple_buy": 3}
        for corp_id, corp_idx in self.corp_id_to_idx.items():
            corporation = game.corporation_by_id(corp_id)
            if not corporation:
                raise ValueError(f"Corporation {corp_id} not found in game.")
            if not corporation.share_price:
                continue

            zone_idx = 0
            type = corporation.share_price.type
            if type in zone_map:
                zone_idx = zone_map[type]

            corp_zone_offset = offset + corp_idx * 4 + zone_idx  # 4 zones: Regular, Yellow, Orange, Brown
            state_encoding[corp_zone_offset] = 1.0
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 19: Trains in Depot ---
        section_name = "depot_trains"
        depot_counts = {name: 0 for name in TRAIN_TYPES_ORDERED}
        if not game.depot:
            raise ValueError("No depot found in game.")
        for train in game.depot.trains:
            if train.name not in depot_counts:
                raise ValueError(f"Train {train.name} not found in TRAIN_TYPES_ORDERED")
            if train.owner == game.depot and train not in game.depot.discarded:
                depot_counts[train.name] += 1

        for train_name, train_idx in self.train_name_to_idx.items():
            initial_count = G1830_TRAIN_COUNT[train_name]
            state_encoding[offset + train_idx] = float(depot_counts[train_name]) / initial_count
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 20: Market Pool Trains (Discarded) ---
        section_name = "market_pool_trains"
        market_train_counts = {name: 0 for name in self.train_name_to_idx}
        for train in game.depot.discarded:
            if train.name not in market_train_counts:
                raise ValueError(f"Train {train.name} not found in TRAIN_TYPES_ORDERED")
            market_train_counts[train.name] += 1

        for train_name, train_idx in self.train_name_to_idx.items():
            initial_count = G1830_TRAIN_COUNT[train_name]
            state_encoding[offset + train_idx] = float(market_train_counts[train_name]) / initial_count
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Section 21: Tiles Available in Market ---
        section_name = "depot_tiles"
        available_tile_counts = {name: 0 for name in TILE_IDS_ORDERED}
        for tile in game.tiles:
            if tile.name not in available_tile_counts:
                raise ValueError(f"Tile '{tile.name}' found in depot but not in TILE_IDS_ORDERED.")
            available_tile_counts[tile.name] += 1

        for tile_name, tile_idx in self.tile_id_to_idx.items():
            initial_count = Map_1830.TILES.get(tile_name, 1)
            state_encoding[offset + tile_idx] = float(available_tile_counts.get(tile_name, 0)) / initial_count
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Sections A1-A4: Auction-Specific Details ---
        is_auction = isinstance(game.round.active_step(), WaterfallAuction)
        if is_auction:
            auction_step = game.round.active_step()
            # LOGGER.debug("Encoding WaterfallAuction details.")

            # A1: Bids
            section_name = "auction_bids"
            for priv_id, priv_idx in self.private_id_to_idx.items():
                company = game.company_by_id(priv_id)
                if not company:
                    raise ValueError(f"Company {priv_id} not found in game.")
                for bid in auction_step.bids.get(company, []):
                    player = bid.entity
                    price = bid.price
                    if player.id not in self.player_id_to_idx:
                        raise ValueError(f"Player {player.id} not found in player_id_to_idx.")
                    player_idx = self.player_id_to_idx[player.id]
                    bid_offset = offset + priv_idx * self.num_players + player_idx
                    state_encoding[bid_offset] = float(price) / self.starting_cash
            offset += self._get_section_size(section_name)
            section_start_offset = self.check_offset(section_name, offset, section_start_offset)

            # A2: Min Bid/Price
            section_name = "auction_min_bid"
            for priv_id, priv_idx in self.private_id_to_idx.items():
                company = game.company_by_id(priv_id)
                if not company:
                    raise ValueError(f"Company {priv_id} not found in game.")
                if company.owner:
                    state_encoding[offset + priv_idx] = -1.0  # Indicate owned
                elif hasattr(auction_step, "min_bid"):
                    min_bid_val = auction_step.min_bid(company)  # Call method
                    state_encoding[offset + priv_idx] = float(min_bid_val) / self.starting_cash
                else:
                    LOGGER.warning("Auction step does not have 'min_bid' method.")
                    state_encoding[offset + priv_idx] = -1.0  # Indicate unknown
            offset += self._get_section_size(section_name)
            section_start_offset = self.check_offset(section_name, offset, section_start_offset)

            # A3: Available for Purchase
            section_name = "auction_available"
            available_company = auction_step.companies[0]
            for priv_id, priv_idx in self.private_id_to_idx.items():
                company = game.company_by_id(priv_id)
                if not company:
                    raise ValueError(f"Company {priv_id} not found in game.")
                if company == available_company:
                    state_encoding[offset + priv_idx] = 1.0
            offset += self._get_section_size(section_name)
            section_start_offset = self.check_offset(section_name, offset, section_start_offset)
        else:
            # LOGGER.debug("Not an Auction round/step, skipping auction-specific encoding sections.")
            # Advance offset by the size of all auction sections
            offset += self._get_section_size("auction_bids")
            section_start_offset = self.check_offset("auction_bids", offset, section_start_offset)
            offset += self._get_section_size("auction_min_bid")
            section_start_offset = self.check_offset("auction_min_bid", offset, section_start_offset)
            offset += self._get_section_size("auction_available")
            section_start_offset = self.check_offset("auction_available", offset, section_start_offset)
            # No need for check_offset here as we skipped the sections

        # A4: Face Value - Always encoded
        section_name = "auction_face_value"
        for priv_id, priv_idx in self.private_id_to_idx.items():
            company = game.company_by_id(priv_id)
            if not company:
                raise ValueError(f"Company {priv_id} not found in game.")
            state_encoding[offset + priv_idx] = float(company.value) / self.starting_cash
        offset += self._get_section_size(section_name)
        section_start_offset = self.check_offset(section_name, offset, section_start_offset)

        # --- Final Offset Check ---
        # LOGGER.debug(f"Final offset after encoding: {offset}")
        if offset != self.ENCODING_SIZE:
            LOGGER.error("CRITICAL: Final offset %d != calculated ENCODING_SIZE %d", offset, self.ENCODING_SIZE)
            raise AssertionError(f"Encoding size mismatch: Final offset {offset} != expected size {self.ENCODING_SIZE}")

        # LOGGER.debug(f"Encoding size check PASSED (Offset: {offset}, Expected: {self.ENCODING_SIZE})")

        state_encoding = np.nan_to_num(state_encoding, nan=0.0, posinf=0.0, neginf=0.0)
        tensor_encoding = from_numpy(state_encoding).unsqueeze(0)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        LOGGER.debug(f"Game State Encoding finished in {duration_ms:.3f} ms.")
        return tensor_encoding

    def get_edge_index(self, game: BaseGame) -> Tuple[Tensor, Tensor]:
        self.initialize(game)
        return self.base_edge_index, self.base_edge_attributes

    def get_node_features(self, game: BaseGame) -> Tensor:
        start_time = time.perf_counter()

        self.initialize(game)

        """Encodes the map into node features and an adjacency list."""
        if self.num_map_node_features <= 0:
            raise ValueError("num_map_node_features not calculated correctly.")

        node_features = np.zeros((self.NUM_HEXES, self.num_map_node_features), dtype=np.float32)

        # --- Populate Node Features ---
        for hex_coord, hex_idx in self.hex_coord_to_idx.items():
            hex_obj = game.hex_by_id(hex_coord)
            tile: Tile = hex_obj.tile
            feature_offset = 0  # Track position within the feature vector for this node
            if not tile:
                raise ValueError(f"Tile not found for hex {hex_coord}.")

            # 1. Revenue
            revenue = 0
            if tile.cities:
                revenue = tile.cities[0].revenue
            elif tile.towns:
                revenue = tile.towns[0].revenue
            elif tile.offboards:
                revenue = tile.offboards[0].revenue

            if isinstance(revenue, dict):
                k = -1
                while game.phase.tiles[k] not in revenue:
                    k -= 1
                    if k < -len(game.phase.tiles):
                        raise ValueError(f"No revenue found for hex {hex_coord}.")
                revenue = revenue[game.phase.tiles[k]]

            node_features[hex_idx, feature_offset] = float(revenue) / MAX_HEX_REVENUE
            feature_offset += 1

            # 2-4. Type Flags
            node_features[hex_idx, feature_offset] = float(bool(tile.cities))
            node_features[hex_idx, feature_offset + 1] = float(len(tile.cities) > 1 or len(tile.towns) > 1)  # OO
            node_features[hex_idx, feature_offset + 2] = float(bool(tile.towns))
            node_features[hex_idx, feature_offset + 3] = float(bool(tile.offboards))
            feature_offset += 4

            # 4. Upgrade/Lay cost
            cost = 0
            if tile.upgrades:
                cost = tile.upgrades[0].cost
            node_features[hex_idx, feature_offset] = float(cost) / MAX_LAY_COST
            feature_offset += 1

            # 5. Rotation
            node_features[hex_idx, feature_offset] = tile.rotation
            feature_offset += 1

            # 6. Token Presence (Per Corporation)
            token_features_start = feature_offset
            for i, city in enumerate(tile.cities):
                for token in city.tokens:
                    if not token:
                        continue
                    if token.corporation.id not in self.corp_id_to_idx:
                        raise ValueError(f"Corporation {token.corporation.id} not found in corp_id_to_idx.")
                    corp_idx = self.corp_id_to_idx[token.corporation.id]
                    node_features[hex_idx, token_features_start + corp_idx * 2 + i] = 1.0
            feature_offset += self.NUM_CORPORATIONS * 2

            # 7. Internal Connectivity Features
            edge_connections = np.zeros((NUM_TILE_EDGES, NUM_TILE_EDGES), dtype=np.float32)
            revenue_connections = np.zeros((NUM_TILE_EDGES, 2), dtype=np.float32)
            for path in tile.paths:
                if isinstance(path.a, Edge) and isinstance(path.b, Edge):
                    edge_connections[path.a.num][path.b.num] = 1.0
                    edge_connections[path.b.num][path.a.num] = 1.0
                    continue

                if isinstance(path.a, Edge):
                    edge = path.a
                    revenue_location = path.b
                else:
                    edge = path.b
                    revenue_location = path.a

                ## OO cities
                if len(tile.cities) > 1 or len(tile.towns) > 1:
                    idx = (
                        tile.cities.index(revenue_location)
                        if isinstance(revenue_location, City)
                        else tile.towns.index(revenue_location)
                    )
                    revenue_connections[edge.num][idx] = 1.0
                    continue

                revenue_connections[edge.num][0] = 1.0

            for i in range(NUM_TILE_EDGES):
                for j in range(i):
                    node_features[hex_idx, feature_offset + (i * (i - 1) // 2) + j] = edge_connections[i][j]

            feature_offset += NUM_PORT_PAIRS

            for i in range(NUM_TILE_EDGES):
                for j in range(2):
                    node_features[hex_idx, feature_offset + i * 2 + j] = revenue_connections[i][j]

            feature_offset += NUM_TILE_EDGES * 2

            if feature_offset != self.num_map_node_features:
                raise ValueError(
                    f"Feature count mismatch for hex {hex_coord}: Expected {self.num_map_node_features}, Actual {feature_offset}"
                )

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        LOGGER.debug(f"Map encoding finished in {duration_ms:.3f} ms.")
        return from_numpy(node_features).float()
