"""Thin adapter that wraps the Rust BaseGame to match Python's BaseGame interface.

This allows the encoder, ActionHelper, and MCTS to use the Rust engine as a
drop-in replacement for the Python game engine. It bridges naming differences
and provides missing properties that the Rust PyO3 interface doesn't expose.

Usage:
    from engine_rs import BaseGame as RustGame
    from rl18xx.rust_adapter import RustGameAdapter

    rust_game = RustGame(names)
    adapted = RustGameAdapter(rust_game)
    # Now use `adapted` wherever Python's BaseGame is expected
"""

from rl18xx.game.engine.entities import Player as PyPlayer
from rl18xx.game.engine.entities import Corporation as PyCorporation
from rl18xx.game.engine.entities import Company as PyCompany
from rl18xx.game.engine.graph import Edge as PyEdge
from rl18xx.game.engine.graph import City as PyCity
from rl18xx.game.engine.graph import Town as PyTown
from rl18xx.game.engine.graph import Offboard as PyOffboard
from rl18xx.game.engine.round import WaterfallAuction as PyWaterfallAuction
from rl18xx.game.engine.round import Operating as PyOperating
from rl18xx.game.engine.round import Stock as PyStock
from rl18xx.game.engine.round import (
    BuyTrain as BuyTrainStep,
    Exchange as ExchangeStep,
    SpecialTrack as SpecialTrackStep,
    SpecialToken as SpecialTokenStep,
)
from rl18xx.game.engine.actions import (
    Bid, BuyCompany, BuyShares, BuyTrain as BuyTrainAction,
    Dividend, LayTile, Par, Pass, PlaceToken, RunRoutes,
    SellShares, DiscardTrain, Bankrupt,
)

# Map action type strings from Rust legal_action_types() to Python action classes
_ACTION_TYPE_MAP = {
    "bid": Bid,
    "buy_company": BuyCompany,
    "buy_shares": BuyShares,
    "buy_train": BuyTrainAction,
    "dividend": Dividend,
    "lay_tile": LayTile,
    "par": Par,
    "pass": Pass,
    "place_token": PlaceToken,
    "run_routes": RunRoutes,
    "sell_shares": SellShares,
    "discard_train": DiscardTrain,
    "bankrupt": Bankrupt,
}

_ROUND_TYPE_MAP = {
    "Auction": PyWaterfallAuction,
    "Stock": PyStock,
    "Operating": PyOperating,
}

# Map for __class__.__name__ used by encoder's ROUND_TYPE_MAP
_ROUND_NAME_MAP = {
    "Auction": "Auction",  # Encoder uses "Auction" not "WaterfallAuction"
    "Stock": "Stock",
    "Operating": "Operating",
}


class _ActionDesc:
    """Lightweight action wrapper with .description() for logging."""

    def __init__(self, raw):
        self._raw = raw

    def description(self):
        if isinstance(self._raw, dict):
            return f"{self._raw.get('type', '?')} by {self._raw.get('entity', '?')}"
        return str(self._raw)


class _NeighborRef:
    """Lightweight object with just .id to satisfy encoder's neighbor.id access."""

    def __init__(self, hex_id: str):
        self.id = hex_id


class _TokenableCityTileRef:
    """Fake tile back-reference for _TokenableCity with city→tile→hex→tile chain."""

    def __init__(self, tile_id, hex_proxy):
        self.id = tile_id
        self.hex = hex_proxy


class _TokenableCity:
    """City proxy returned by connected_nodes that supports tokenable() and get_slot()."""

    def __init__(self, city, city_index, hex_id, tile_id, slot, corp_sym, hex_proxy=None):
        self._city = city
        self._city_index = city_index
        self._hex_id = hex_id
        self._slot = slot
        self._corp_sym = corp_sym
        # Back-reference: city.tile.hex must return a hex with .tile.cities
        self.tile = _TokenableCityTileRef(tile_id, hex_proxy)

    def __getattr__(self, name):
        return getattr(self._city, name)

    @property
    def id(self):
        """Match Python's BasePart.id: tile_id-city_index."""
        return f"{self.tile.id}-{self._city_index}"

    def tokenable(self, entity):
        """Always tokenable — Rust already filtered for this."""
        return True

    def get_slot(self, entity):
        """Find the first empty token slot in this city."""
        for i, t in enumerate(self._city.tokens):
            if t is None:
                return i
        return 0

    @property
    def tokens(self):
        return [_TokenProxy(t) if t else None for t in self._city.tokens]

    @property
    def hex_id(self):
        return self._hex_id

    @property
    def __class__(self):
        return PyCity

    def __eq__(self, other):
        if isinstance(other, _TokenableCity):
            return self._hex_id == other._hex_id and self._city_index == other._city_index
        if isinstance(other, _CityRef):
            return self._city_index == other._index
        if isinstance(other, _EndpointProxy) and other._type_name == "City":
            return other._index == self._city_index
        return NotImplemented

    def __hash__(self):
        return hash(("City", self._hex_id, self._city_index))

    def __lt__(self, other):
        if isinstance(other, _TokenableCity):
            return (self._hex_id, self._city_index) < (other._hex_id, other._city_index)
        return NotImplemented


class _TokenProxy:
    """Proxy for Token that provides .corporation as an object with .id and .type alias."""

    def __init__(self, token):
        self._token = token

    def __getattr__(self, name):
        return getattr(self._token, name)

    @property
    def type(self):
        """Alias: Python uses .type, Rust uses .token_type."""
        return self._token.token_type

    @property
    def corporation(self):
        return _NeighborRef(self._token.corporation_id)


class _CityRef:
    """Wrapper for a city that supports == comparison with _EndpointProxy and list.index()."""

    def __init__(self, city, index, tile_id=None):
        self._city = city
        self._index = index
        self._tile_id = tile_id

    def __getattr__(self, name):
        return getattr(self._city, name)

    @property
    def id(self):
        """Match Python's BasePart.id: tile_id-city_index."""
        return f"{self._tile_id}-{self._index}" if self._tile_id else f"?-{self._index}"

    @property
    def tokens(self):
        """Wrap tokens so token.corporation.id works."""
        return [_TokenProxy(t) if t else None for t in self._city.tokens]

    def get_slot(self, entity):
        """Find the first empty token slot in this city."""
        for i, t in enumerate(self._city.tokens):
            if t is None:
                return i
        return 0

    def __eq__(self, other):
        if isinstance(other, _EndpointProxy) and other._type_name == "City":
            return other._index == self._index
        if isinstance(other, _CityRef):
            return other._index == self._index
        return NotImplemented

    def __hash__(self):
        return hash(("City", self._index))

    def get_slot(self, entity):
        """Find the first empty token slot in this city."""
        for i, t in enumerate(self._city.tokens):
            if t is None:
                return i
        return 0

    @property
    def __class__(self):
        return PyCity


class _TownRef:
    """Wrapper for a town that supports == comparison with _EndpointProxy."""

    def __init__(self, town, index):
        self._town = town
        self._index = index

    def __getattr__(self, name):
        return getattr(self._town, name)

    def __eq__(self, other):
        if isinstance(other, _EndpointProxy) and other._type_name == "Town":
            return other._index == self._index
        if isinstance(other, _TownRef):
            return other._index == self._index
        return NotImplemented

    def __hash__(self):
        return hash(("Town", self._index))

    @property
    def __class__(self):
        return PyTown


class _TileProxy:
    """Proxy for Tile that adds .paths and makes cities/towns endpoint-comparable."""

    def __init__(self, tile):
        self._tile = tile

    def __getattr__(self, name):
        return getattr(self._tile, name)

    @property
    def paths(self):
        """Alias for path_defs, returning objects with .a and .b attributes."""
        return [_PathProxy(a, b, t) for a, b, t in self._tile.path_defs]

    @property
    def cities(self):
        """Cities wrapped for endpoint comparison."""
        tile_id = self._tile.id
        return [_CityRef(c, i, tile_id) for i, c in enumerate(self._tile.cities)]

    @property
    def towns(self):
        """Towns wrapped for endpoint comparison."""
        return [_TownRef(t, i) for i, t in enumerate(self._tile.towns)]


class _PathProxy:
    """Proxy for a path definition with .a and .b endpoint objects."""

    def __init__(self, a_str: str, b_str: str, terminal: bool):
        self.a = _EndpointProxy(a_str)
        self.b = _EndpointProxy(b_str)
        self.terminal = terminal


_EP_TYPE_MAP = {
    "Edge": PyEdge,
    "City": PyCity,
    "Town": PyTown,
    "Offboard": PyOffboard,
}


class _EndpointProxy:
    """Proxy for a path endpoint like 'Edge(3)' or 'City(0)'.
    Passes isinstance checks against Python's Edge/City/Town/Offboard."""

    def __init__(self, ep_str: str):
        self._str = ep_str
        if '(' in ep_str:
            self._type_name = ep_str.split('(')[0]
            self._index = int(ep_str.split('(')[1].rstrip(')'))
        else:
            self._type_name = ep_str
            self._index = 0

    @property
    def num(self):
        """Edge number (for Edge endpoints)."""
        return self._index

    @property
    def index(self):
        """City/Town/Offboard index."""
        return self._index

    @property
    def __class__(self):
        return _EP_TYPE_MAP.get(self._type_name, type(self))

    def __repr__(self):
        return self._str


class _HexProxy:
    """Proxy for Hex that provides all_neighbors and tile.paths."""

    def __init__(self, hex_obj, adjacency=None):
        self._hex = hex_obj
        self._adjacency = adjacency or {}

    def __getattr__(self, name):
        return getattr(self._hex, name)

    @property
    def tile(self):
        return _TileProxy(self._hex.tile)

    @property
    def all_neighbors(self):
        """Returns {direction: _NeighborRef} using the game's hex_adjacency data."""
        adj = self._adjacency.get(self._hex.id, {})
        return {d: _NeighborRef(hid) for d, hid in adj.items()}

    @property
    def neighbors(self):
        """Same as all_neighbors."""
        return self.all_neighbors


class _RoundProxy:
    """Proxy for game.round that provides active_entities and __class__.__name__."""

    def __init__(self, game):
        self._game = game
        self._round = game.round

    @property
    def round_type(self):
        return self._round.round_type

    @property
    def round_num(self):
        return self._round.round_num

    def round_description(self):
        """Human-readable round description for logging."""
        rt = self._round.round_type
        rn = self._round.round_num
        return f"{rt} Round {rn}"

    @property
    def active_entities(self):
        """Returns a list of the currently active entity (Player or Corporation)."""
        player = self._game.current_player
        if player is not None:
            return [_PlayerProxy(player)]
        corp = self._game.current_corporation
        if corp is not None:
            return [_CorpProxy(corp)]
        return []

    @property
    def active_entity_id_str(self):
        return self._round.active_entity_id_str

    @property
    def active_entity_is_player(self):
        return self._round.active_entity_is_player

    @property
    def active_entity_is_corporation(self):
        return self._round.active_entity_is_corporation

    @property
    def __class__(self):
        """Returns the actual Python round class for isinstance checks."""
        return _ROUND_TYPE_MAP.get(self._round.round_type, type(self))

    def actions_for(self, entity):
        """Bridge for game.round.actions_for(entity).

        Uses Rust's legal_action_types() and maps type strings to Python action classes.
        For company entities, returns Pass if the company has relevant abilities.
        """
        # Company entities: check if this company has special abilities
        if isinstance(entity, _CompanyProxy) or isinstance(entity, PyCompany):
            sym = entity.sym if hasattr(entity, 'sym') else str(entity)
            # Companies can lay tiles (CS, DH) or place tokens (DH) or exchange shares (MH)
            step_type = self._game.active_step_type()
            co = self._game.company_by_id(sym) if isinstance(sym, str) else None
            if co and not co.closed:
                owner_str = co.owner
                round_type = self._game.round.round_type
                if round_type == "Operating":
                    # Check if this company has abilities usable in the current step
                    if sym in ("CS", "DH") and step_type in ("LayTile", "PlaceToken", "RunRoutes", "Dividend", "BuyTrain", "BuyCompany"):
                        return [LayTile]
                    if sym == "DH" and step_type == "PlaceToken" and co.ability_used:
                        return [PlaceToken]
                    if sym == "MH" and owner_str and owner_str.startswith("player:"):
                        return [BuyShares]
                elif round_type == "Stock":
                    if sym == "MH" and owner_str and owner_str.startswith("player:"):
                        return [BuyShares]
            return []

        # For player/corporation entities: use Rust's legal_action_types()
        type_strings = self._game.legal_action_types()
        return [_ACTION_TYPE_MAP[t] for t in type_strings if t in _ACTION_TYPE_MAP]

    @property
    def steps(self):
        """Return a list of step-like objects for isinstance checks.

        ActionHelper iterates round.steps to find ExchangeStep, SpecialTrackStep,
        SpecialTokenStep. We return proxy objects that pass these checks when
        the corresponding abilities are available.
        """
        result = []
        step_type = self._game.active_step_type()
        # Always include the active step
        if step_type == "WaterfallAuction":
            result.append(_AuctionStepProxy(self._game, PyWaterfallAuction))
        else:
            result.append(_StepProxy(self._game, step_type))
        # Add special step proxies for company abilities
        round_type = self._game.round.round_type
        if round_type == "Operating":
            result.append(_SpecialTrackStepProxy(self._game))
            result.append(_SpecialTokenStepProxy(self._game))
            result.append(_ExchangeStepProxy(self._game))
        elif round_type == "Stock":
            result.append(_ExchangeStepProxy(self._game))
        return result

    def active_step(self):
        """Return a proxy that passes isinstance checks for WaterfallAuction etc."""
        step_type = self._game.active_step_type()
        # Map step type to actual Python class for isinstance checks
        step_class_map = {
            "WaterfallAuction": PyWaterfallAuction,
        }
        base_class = step_class_map.get(step_type)
        if base_class:
            return _AuctionStepProxy(self._game, base_class)
        return _StepProxy(self._game, step_type)


class _AuctionStepProxy:
    """Step proxy that passes isinstance(x, WaterfallAuction) AND has auction methods."""

    def __init__(self, game, base_class):
        self._game = game
        self._base_class = base_class

    @property
    def __class__(self):
        return self._base_class

    @property
    def bids(self):
        result = {}
        for co_sym in self._game.auction_companies():
            bids = self._game.auction_bids(co_sym)
            if bids:
                co = self._game.company_by_id(co_sym)
                result[_CompanyProxy(co)] = [
                    type('Bid', (), {'entity': _OwnerProxy(f'player:{pid}'), 'price': price})()
                    for pid, price in bids
                ]
        return result

    @property
    def companies(self):
        return [_CompanyProxy(self._game.company_by_id(sym))
                for sym in self._game.auction_companies()]

    @property
    def companies_pending_par(self):
        """Return list of (company, corporation) pending par after auction win."""
        pending = self._game.auction_pending_par()
        if pending:
            corp_sym, player_id = pending
            co = self._game.company_by_id(corp_sym)
            if co:
                return [_CompanyProxy(co)]
        return []

    def min_bid(self, company):
        sym = company.sym if hasattr(company, 'sym') else str(company)
        return self._game.auction_min_bid(sym)

    def max_bid(self, entity, company):
        player_id = entity.id if hasattr(entity, 'id') else entity
        sym = company.sym if hasattr(company, 'sym') else str(company)
        return self._game.auction_max_bid(player_id, sym)

    def auctioning_company(self):
        sym = self._game.auctioning_company()
        if sym:
            co = self._game.company_by_id(sym)
            return _CompanyProxy(co) if co else None
        return None


class _StepProxy:
    """Proxy for an active step — bridges OR step methods to Rust BaseGame."""

    def __init__(self, game, step_type):
        self._game = game
        self._step_type = step_type

    # --- Auction methods (used when step is WaterfallAuction) ---
    @property
    def bids(self):
        result = {}
        for co_sym in self._game.auction_companies():
            bids = self._game.auction_bids(co_sym)
            if bids:
                co = self._game.company_by_id(co_sym)
                result[_CompanyProxy(co)] = [
                    type('Bid', (), {'entity': _OwnerProxy(f'player:{pid}'), 'price': price})()
                    for pid, price in bids
                ]
        return result

    @property
    def companies(self):
        return [_CompanyProxy(self._game.company_by_id(sym))
                for sym in self._game.auction_companies()]

    @property
    def companies_pending_par(self):
        """Return companies awaiting par after auction win.

        auction_pending_par() returns (corp_sym, player_id). We need to find
        the company that triggers parring of that corporation. In 1830:
        BO company → B&O corp, CA company → not a par trigger, etc.
        The company-to-corp mapping is: BO→B&O (the only par-triggering company).
        """
        pending = self._game.auction_pending_par()
        if pending:
            corp_sym, player_id = pending
            # Find the company that grants shares in this corporation
            # In 1830, BO company triggers B&O par
            _COMPANY_TO_CORP = {"BO": "B&O"}
            for co_sym, co_corp in _COMPANY_TO_CORP.items():
                if co_corp == corp_sym:
                    co = self._game.company_by_id(co_sym)
                    if co:
                        return [_CompanyProxy(co)]
        return []

    def min_bid(self, company):
        sym = company.sym if hasattr(company, 'sym') else str(company)
        return self._game.auction_min_bid(sym)

    def max_bid(self, entity, company):
        player_id = entity.id if hasattr(entity, 'id') else entity
        sym = company.sym if hasattr(company, 'sym') else str(company)
        return self._game.auction_max_bid(player_id, sym)

    def auctioning_company(self):
        sym = self._game.auctioning_company()
        if sym:
            co = self._game.company_by_id(sym)
            return _CompanyProxy(co) if co else None
        return None

    # --- Stock round methods ---
    def buyable_shares(self, entity):
        """Returns list of lists: each inner list is shares for a corp:source pair."""
        player_id = entity.id if hasattr(entity, 'id') else entity
        if isinstance(player_id, str):
            player_id = int(player_id) if player_id.isdigit() else 0
        tuples = self._game.buyable_shares(player_id)
        from collections import defaultdict
        groups = defaultdict(list)
        for corp_sym, source, idx, price in tuples:
            corp = self._game.corporation_by_id(corp_sym)
            share = _BuyableShare(corp, source, idx, price, self._game)
            groups[(corp_sym, source)].append(share)
        return list(groups.values())

    def sellable_shares(self, entity):
        """Returns list of share bundles from Rust sellable_bundles()."""
        player_id = entity.id if hasattr(entity, 'id') else entity
        if isinstance(player_id, str):
            player_id = int(player_id) if player_id.isdigit() else 0
        tuples = self._game.sellable_bundles(player_id)
        result = []
        for corp_sym, count, pct in tuples:
            corp = self._game.corporation_by_id(corp_sym)
            bundle = _SellableBundle(corp, count, pct, player_id)
            result.append(bundle)
        return result

    def sellable_bundle(self, share):
        """Check if a specific share is sellable. Returns the share if yes."""
        return share  # Simplified: if it came from sellable_shares, it's sellable

    @property
    def pending_token(self):
        """Return pending token info if any."""
        return None  # Handled by pending_tokens at the round level

    # --- Operating round methods ---
    def upgradeable_tiles(self, entity, hex_obj):
        """Returns tiles that can be placed on a hex."""
        hex_id = hex_obj.id if hasattr(hex_obj, 'id') else str(hex_obj)
        tuples = self._game.upgradeable_tiles_for(hex_id)
        result = []
        for tile_name, rotation in tuples:
            result.append(_UpgradeableTile(tile_name, self._game))
        return result

    def ability_blocking_hex(self, entity, hex_obj):
        """Check if a hex is blocked by a private company ability."""
        hex_id = hex_obj.id if hasattr(hex_obj, 'id') else str(hex_obj)
        # Hardcoded 1830 blocks: private companies block specific hexes
        for co in self._game.companies:
            if co.closed:
                continue
            owner_str = co.owner
            if not owner_str or not owner_str.startswith("player:"):
                continue
            blocked = {
                "SV": ["G15"], "CS": ["B20"], "DH": ["F16"],
                "MH": ["D18"], "CA": ["H18"], "BO": ["I13", "I15"],
            }.get(co.sym, [])
            if hex_id in blocked:
                return True  # Return truthy to indicate blocking
        return None

    def legal_tile_rotations(self, entity, hex_obj, tile):
        """Returns valid rotations for a tile on a hex."""
        hex_id = hex_obj.id if hasattr(hex_obj, 'id') else str(hex_obj)
        tile_name = tile.name if hasattr(tile, 'name') else str(tile)
        rotations = self._game.legal_tile_rotations(hex_id, tile_name)
        return list(rotations)

    def dividend_options(self, entity):
        """Returns dividend options as a dict."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        options = self._game.dividend_options(corp_sym)
        return {opt: True for opt in options}

    def president_may_contribute(self, entity):
        """Whether president must contribute to train purchase."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        return self._game.president_may_contribute(corp_sym)

    def buyable_trains(self, entity):
        """Returns list of buyable trains for a corporation."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        tuples = self._game.buyable_trains_for(corp_sym)
        depot_proxy = _DepotProxy.__new__(_DepotProxy)
        depot_proxy._update(self._game.depot)
        result = []
        for train_id, name, price, source in tuples:
            result.append(_BuyableTrain(train_id, name, price, source, depot_proxy, self._game))
        return result

    def spend_minmax(self, entity, train):
        """Returns (min_spend, max_spend) for buying a train."""
        # For cross-company trains: min=1, max=entity.cash (or +president if ebuy)
        if hasattr(train, '_source') and train._source != "depot":
            corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
            ci = None
            for i, c in enumerate(self._game.corporations):
                if c.sym == corp_sym:
                    ci = i
                    break
            corp_cash = self._game.corporations[ci].cash if ci is not None else 0
            if self.president_may_contribute(entity):
                pres = self._game.corporations[ci].owner_id_str if ci is not None else ""
                if pres.startswith("player:"):
                    pid = int(pres.split(":")[1])
                    pres_cash = next((p.cash for p in self._game.players if p.id == pid), 0)
                    return (1, corp_cash + pres_cash)
            return (1, corp_cash)
        # Depot train: fixed price
        price = train.price if hasattr(train, 'price') else train._price
        return (price, price)

    def names_of_cheapest_variants(self, train):
        """Return name variants of the cheapest depot train."""
        name = train.name if hasattr(train, 'name') else str(train)
        return [name]

    def can_ebuy_sell_shares(self, entity):
        """Whether the president can sell shares for emergency buy."""
        return self.president_may_contribute(entity)


class _SpecialTrackStepProxy:
    """Proxy that passes isinstance(x, SpecialTrackStep)."""

    def __init__(self, game):
        self._game = game

    @property
    def __class__(self):
        return SpecialTrackStep

    @property
    def active(self):
        return True

    @property
    def blocking(self):
        return False

    def actions(self, entity):
        return []

    def abilities(self, company):
        return []


class _SpecialTokenStepProxy:
    """Proxy that passes isinstance(x, SpecialTokenStep)."""

    def __init__(self, game):
        self._game = game

    @property
    def __class__(self):
        return SpecialTokenStep

    @property
    def active(self):
        return True

    @property
    def blocking(self):
        return False

    def actions(self, entity):
        return []


class _ExchangeStepProxy:
    """Proxy that passes isinstance(x, ExchangeStep)."""

    def __init__(self, game):
        self._game = game

    @property
    def __class__(self):
        return ExchangeStep

    @property
    def active(self):
        return True

    @property
    def blocking(self):
        return False

    def actions(self, entity):
        return []

    def exchangeable_shares(self, company):
        """MH exchange: return NYC share from IPO if MH is owned and NYC exists."""
        # In 1830, MH company can be exchanged for a NYC IPO share
        sym = company.sym if hasattr(company, 'sym') else str(company)
        if sym != "MH":
            return []
        nyc = self._game.corporation_by_id("NYC")
        if not nyc or nyc.ipo_price is None:
            return []
        # Check if NYC has IPO shares available
        ipo_shares = [s for s in nyc.shares if s.owner == "corp:NYC"]
        if not ipo_shares:
            return []
        # Return a buyable share proxy for the first IPO share
        return [_BuyableShare(nyc, "ipo", ipo_shares[0].index, 0, self._game)]


class _WrappedShare:
    """Wraps a Rust Share to add .corporation() method and typed .owner."""

    def __init__(self, share, owner_proxy):
        self._share = share
        self.owner = owner_proxy
        self.id = share.id if hasattr(share, 'id') else f"{share.corporation_id}_{share.index}"
        self.index = share.index
        self.percent = share.percent
        self.president = share.president

    def corporation(self):
        return _CorpRef(self._share.corporation_id)

    def __getattr__(self, name):
        return getattr(self._share, name)


class _BuyableShare:
    """Proxy for a buyable share returned by buyable_shares()."""

    def __init__(self, corp, source, idx, price, game):
        self._corp = _CorpProxy(corp) if corp else None
        self._source = source
        self.index = idx
        self.price = price
        self.percent = 10  # Standard 1830 share
        self._game = game

    @property
    def id(self):
        """Share ID for action serialization."""
        return f"{self._corp.sym}_{self.index}" if self._corp else f"unknown_{self.index}"

    def corporation(self):
        """Corporation this share belongs to (callable, matching Python Share)."""
        return self._corp

    @property
    def owner(self):
        """Owner of this share. ActionMapper checks owner.name for routing."""
        if self._source == "ipo":
            return _BankProxy()  # IPO shares owned by "The Bank"
        return _MarketProxy()  # Market shares owned by "Market"

    def __eq__(self, other):
        if isinstance(other, _BuyableShare):
            return (self._corp.sym == other._corp.sym and
                    self._source == other._source and self.index == other.index)
        return NotImplemented

    def __hash__(self):
        return hash((self._corp.sym if self._corp else "", self._source, self.index))

    def __lt__(self, other):
        if isinstance(other, _BuyableShare):
            return (self._corp.sym, self._source, self.index) < (other._corp.sym, other._source, other.index)
        return NotImplemented


from rl18xx.game.engine.entities import ShareBundle as _PyShareBundle


class _SellableBundle(_PyShareBundle):
    """Proxy for a sellable share bundle that IS a ShareBundle (skips validation)."""

    def __init__(self, corp, count, pct, player_id=0):
        # Skip ShareBundle.__init__ validation — we construct directly
        self._corp_ref = _CorpRef(corp.sym if corp else "")
        self.percent = pct
        self.share_price = None
        corp_sym = corp.sym if corp else ""
        owner = _OwnerProxy(f"player:{player_id}")
        self.shares = [
            _SellShareRef(corp_sym, owner, i) for i in range(count)
        ]

    @property
    def corporation(self):
        return self._corp_ref

    def num_shares(self, ceil=True):
        return len(self.shares)

    @property
    def partial(self):
        return False

    def __eq__(self, other):
        if isinstance(other, _SellableBundle):
            return (self._corp_ref.sym == other._corp_ref.sym and
                    len(self.shares) == len(other.shares))
        return NotImplemented

    def __hash__(self):
        return hash((self._corp_ref.sym, len(self.shares)))

    def __lt__(self, other):
        if isinstance(other, _SellableBundle):
            return (self._corp_ref.sym, len(self.shares)) < (other._corp_ref.sym, len(other.shares))
        return NotImplemented


class _SellShareRef:
    """Share reference for sellable bundles with all required attributes."""

    def __init__(self, corp_sym, owner, index):
        self.id = f"{corp_sym}_{index}"
        self.index = index
        self._corp_sym = corp_sym
        self.percent = 10
        self.owner = owner

    def corporation(self):
        return _CorpRef(self._corp_sym)


class _ShareRef:
    """Lightweight share reference with .id for action serialization."""

    def __init__(self, corp_sym, owner_type, index):
        self.id = f"{corp_sym}_{index}"
        self.index = index
        self._corp_sym = corp_sym
        self.percent = 10  # Standard 1830 share

    def corporation(self):
        return _CorpRef(self._corp_sym)


class _CorpRef:
    """Lightweight corporation reference for hashing in ShareBundle."""

    def __init__(self, sym):
        self.id = sym
        self.sym = sym
        self.name = sym

    def __eq__(self, other):
        if hasattr(other, 'sym'):
            return self.sym == other.sym
        return NotImplemented

    def __hash__(self):
        return hash(self.sym)


class _UpgradeableTile:
    """Proxy for an upgradeable tile."""

    def __init__(self, name, game):
        self.name = name
        self._game = game

    def __eq__(self, other):
        if hasattr(other, 'name'):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


class _BuyableTrain:
    """Proxy for a buyable train from buyable_trains_for()."""

    def __init__(self, train_id, name, price, source, depot_proxy, game):
        self._id = train_id
        self.id = train_id
        self.name = name
        self._price = price
        self.price = price
        self._source = source
        self._depot = depot_proxy
        self._game = game
        self.operated = False
        self.variant = name  # Default variant is the train name

    def from_depot(self):
        return self._source == "depot"

    def min_price(self):
        if self.from_depot():
            return self._price
        return 1

    @property
    def owner(self):
        if self.from_depot():
            return self._depot
        # Cross-company train: find owning corp
        for c in self._game.corporations:
            for t in c.trains:
                if t.name == self.name:
                    return _CorpProxy(c)
        return self._depot

    def __eq__(self, other):
        if hasattr(other, '_id'):
            return self._id == other._id and self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash((self._id, self.name))


class _SharePriceProxy:
    """Proxy for a share price with .price, .id, .coordinates attributes."""

    def __init__(self, price, row=0, col=0):
        self.price = price
        self.coordinates = (row, col)

    @property
    def id(self):
        return f"{self.price},{self.coordinates[0]},{self.coordinates[1]}"

    def __repr__(self):
        return f"SharePrice({self.price})"


class _GraphProxy:
    """Proxy for game.graph that provides connected_hexes and connected_nodes."""

    def __init__(self, game, hex_adjacency):
        self._game = game
        self._hex_adjacency = hex_adjacency

    def connected_hexes(self, entity):
        """Returns dict of {hex_proxy: set(edges)} for the entity's network."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        raw = self._game.connected_hexes(corp_sym)
        result = {}
        for hex_id, edges in raw.items():
            h = self._game.hex_by_id(hex_id)
            if h:
                hex_proxy = _HexProxy(h, self._hex_adjacency)
                result[hex_proxy] = set(edges)
        return result

    def connected_nodes(self, entity):
        """Returns dict of {city_like: True} matching Python graph API."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        tokenable = self._game.tokenable_cities_for(corp_sym)
        nodes = {}
        for hex_id, city_idx in tokenable:
            h = self._game.hex_by_id(hex_id)
            if h and city_idx < len(h.tile.cities):
                hex_proxy = _HexProxy(h, self._hex_adjacency)
                tile_id = h.tile.id
                city = _TokenableCity(h.tile.cities[city_idx], city_idx, hex_id, tile_id, 0, corp_sym, hex_proxy)
                nodes[city] = True
        return nodes

    def route_info(self, entity):
        """Returns route info dict matching Python's graph.route_info."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        # Can't easily get this from Rust without a method; delegate
        return {"route_available": True, "route_train_purchase": True}


class _StockMarketProxy:
    """Proxy for stock_market with par_prices."""

    def __init__(self, game):
        self._game = game

    @property
    def par_prices(self):
        try:
            tuples = self._game.par_prices_with_coords()
            return [_SharePriceProxy(p, r, c) for p, r, c in tuples]
        except AttributeError:
            return [_SharePriceProxy(p) for p in self._game.par_prices()]


class _SharePoolProxy:
    """Proxy for share_pool — used for ownership checks in action_mapper."""
    name = "Market"

    def __eq__(self, other):
        if isinstance(other, (_SharePoolProxy, _MarketProxy)):
            return True
        return False

    def __hash__(self):
        return hash("share_pool")


class _BankProxy:
    """Proxy for 'The Bank' — owner of IPO shares."""
    name = "The Bank"

    def __eq__(self, other):
        if isinstance(other, _BankProxy):
            return True
        return False

    def __hash__(self):
        return hash("bank")


class _MarketProxy:
    """Proxy for 'Market' — owner of market shares."""
    name = "Market"

    def __eq__(self, other):
        if isinstance(other, (_MarketProxy, _SharePoolProxy)):
            return True
        return False

    def __hash__(self):
        return hash("market")


class _PlayerProxy:
    """Proxy for Rust Player that passes isinstance(x, PyPlayer) checks."""

    def __init__(self, player):
        self._player = player

    def __getattr__(self, name):
        return getattr(self._player, name)

    def percent_of(self, corp):
        """Unwrap CorpProxy before passing to Rust."""
        real_corp = corp._corp if isinstance(corp, _CorpProxy) else corp
        return self._player.percent_of(real_corp)

    def is_player(self):
        return True

    def is_corporation(self):
        return False

    def is_company(self):
        return False

    def player(self):
        return self

    @property
    def __class__(self):
        return PyPlayer


class _OwnerProxy:
    """Proxy for entity ID strings that pass isinstance checks."""

    def __init__(self, owner_str: str):
        self._str = owner_str
        if owner_str.startswith("player:"):
            self.id = int(owner_str.split(":")[1])
            self._is_player = True
            self._is_corp = False
        elif owner_str.startswith("corp:"):
            self.id = owner_str.split(":")[1]
            self._is_player = False
            self._is_corp = True
        else:
            self.id = owner_str
            self._is_player = False
            self._is_corp = False

    @property
    def __class__(self):
        if self._is_player:
            return PyPlayer
        if self._is_corp:
            return PyCorporation
        return type(self)

    def __bool__(self):
        return bool(self._str)

    def is_player(self):
        return self._is_player

    def is_corporation(self):
        return self._is_corp

    def player(self):
        return self if self._is_player else None

    @property
    def companies(self):
        """Return empty list; OwnerProxy doesn't have game context."""
        return []


class _PresidentProxy(_OwnerProxy):
    """Player proxy for a corporation's president, with game-aware .companies."""

    def __init__(self, owner_str, game):
        super().__init__(owner_str)
        self._game = game

    @property
    def companies(self):
        """Return private companies owned by this player."""
        if not self._game or not self._is_player:
            return []
        pid_str = f"player:{self.id}"
        return [_CompanyProxy(co) for co in self._game.companies
                if not co.closed and co.owner == pid_str]

    @property
    def cash(self):
        if self._game and self._is_player:
            for p in self._game.players:
                if p.id == self.id:
                    return p.cash
        return 0


class _CompanyProxy:
    """Proxy for Company that wraps owner as a typed proxy."""

    def __init__(self, company):
        self._company = company

    def __getattr__(self, name):
        return getattr(self._company, name)

    def __eq__(self, other):
        if isinstance(other, _CompanyProxy):
            return self._company.sym == other._company.sym
        if hasattr(other, 'sym'):
            return self._company.sym == other.sym
        return NotImplemented

    def __hash__(self):
        return hash(self._company.sym)

    @property
    def owner(self):
        if self._company.closed:
            return None
        owner_str = self._company.owner
        if not owner_str:
            return None
        return _OwnerProxy(owner_str)

    def player(self):
        """Return the owning player proxy."""
        owner = self.owner
        if owner and owner.is_player():
            return owner
        return None

    def is_player(self):
        return False

    def is_corporation(self):
        return False

    def is_company(self):
        return True

    @property
    def min_bid(self):
        """Company's face value (used as minimum bid)."""
        return self._company.value

    @property
    def min_price(self):
        """Minimum price to buy this company from a player."""
        v = self._company.value
        return (v // 2) + (v % 2) if v else 0

    @property
    def max_price(self):
        """Maximum price to buy this company from a player."""
        v = self._company.value
        return v * 2 if v else 0

    @property
    def abilities(self):
        """Return abilities for this company. In 1830, BO has a SharesAbility."""
        if self._company.sym == "BO":
            return [_BOSharesAbility()]
        return []

    @property
    def __class__(self):
        return PyCompany


class _BOSharesAbility:
    """Fake SharesAbility for BO company → B&O corporation."""
    from rl18xx.game.engine.abilities import Shares as _SharesType
    type = "shares"

    @property
    def shares(self):
        return [_BOShareRef()]

    @property
    def __class__(self):
        return _BOSharesAbility._SharesType


class _BOShareRef:
    """Fake share reference for BO→B&O mapping."""
    def corporation(self):
        # Return an object with .id and .sym for B&O
        return type('Corp', (), {'id': 'B&O', 'sym': 'B&O', 'name': 'B&O'})()


class _DepotProxy:
    """Proxy for Depot where train.owner == depot for depot trains.
    Identity-stable: same object returned each time so train.owner == game.depot works."""

    def __init__(self, depot=None):
        if depot:
            self._update(depot)

    def _update(self, depot):
        self._depot = depot
        self._trains = [_DepotTrainProxy(t, self) for t in depot.trains]
        self._discarded = [_DepotTrainProxy(t, self) for t in depot.discarded]

    @property
    def trains(self):
        return self._trains

    @property
    def discarded(self):
        return self._discarded

    name = "The Depot"

    def __getattr__(self, name):
        return getattr(self._depot, name)

    @property
    def min_depot_train(self):
        """Cheapest depot train."""
        if self._trains:
            return min(self._trains, key=lambda t: t.price)
        return None

    def depot_trains(self, entity=None):
        """Return available depot trains."""
        return self._trains

    def available(self, entity=None):
        """Return available depot trains."""
        return self._trains

    def __bool__(self):
        return True


class _DepotTrainProxy:
    """Train proxy where .owner returns the depot instance."""

    def __init__(self, train, depot_proxy):
        self._train = train
        self._depot = depot_proxy

    def __getattr__(self, name):
        return getattr(self._train, name)

    @property
    def owner(self):
        """Depot trains are owned by the depot."""
        return self._depot


class _CorpProxy:
    """Proxy for Corporation that makes floated() callable and passes isinstance."""

    def __init__(self, corp, game=None):
        self._corp = corp
        self._game_ref = game

    def __getattr__(self, name):
        return getattr(self._corp, name)

    @property
    def name(self):
        return self._corp.name

    def floated(self):
        """Callable version of floated (encoder calls corp.floated())."""
        return self._corp.floated

    def is_player(self):
        return False

    def is_corporation(self):
        return True

    def is_company(self):
        return False

    def find_token_by_type(self, token_type=None):
        """Find the first unused token of the given type."""
        token_type = token_type or "normal"
        for t in self._corp.tokens:
            if not t.used and t.token_type == token_type:
                return _TokenProxy(t)
        return None

    @property
    def owner(self):
        """Return the president as a player-like proxy with .companies."""
        owner_str = self._corp.owner_id_str
        if owner_str and owner_str.startswith("player:"):
            return _PresidentProxy(owner_str, self._game_ref)
        return None

    def player(self):
        """Return the president player."""
        return self.owner

    @property
    def companies(self):
        """Companies owned by this corporation."""
        return []

    @property
    def ipo_shares(self):
        """Non-president shares in IPO, wrapped for ActionMapper compatibility."""
        ipo_eid = f"ipo:{self._corp.sym}"
        return [_WrappedShare(s, _BankProxy()) for s in self._corp.shares
                if s.owner == ipo_eid and not s.president]

    @property
    def market_shares(self):
        """Shares in the market pool, wrapped for ActionMapper compatibility."""
        return [_WrappedShare(s, _MarketProxy()) for s in self._corp.shares
                if s.owner == "market"]

    def __eq__(self, other):
        if isinstance(other, _CorpProxy):
            return self._corp.sym == other._corp.sym
        if hasattr(other, 'sym'):
            return self._corp.sym == other.sym
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, _CorpProxy):
            return self._corp.sym < other._corp.sym
        if hasattr(other, 'sym'):
            return self._corp.sym < other.sym
        return NotImplemented

    def __hash__(self):
        return hash(self._corp.sym)

    @property
    def __class__(self):
        return PyCorporation


class RustGameAdapter:
    """Wraps a Rust BaseGame to match Python's BaseGame interface for the encoder."""

    def __init__(self, rust_game):
        self._game = rust_game
        self._hex_adjacency = None
        self._depot_proxy = None
        self._move_number_override = None

    @property
    def move_number(self):
        # Check if raw_actions was monkey-patched (test compatibility)
        if 'raw_actions' in self.__dict__:
            return len(self.__dict__['raw_actions'])
        return self._game.move_number

    def _get_hex_adjacency(self):
        """Lazily load hex adjacency map."""
        if self._hex_adjacency is None:
            self._hex_adjacency = self._game.hex_adjacency_map
        return self._hex_adjacency

    def __getattr__(self, name):
        """Delegate everything to the underlying Rust game."""
        return getattr(self._game, name)

    def graph_for_entity(self, entity):
        """Returns the graph proxy (same for all entities in 1830)."""
        return self.graph

    def auto_routes_for(self, entity):
        """Compute optimal routes using Rust router. Returns (routes_dicts, revenue)."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        return self._game.calculate_routes(corp_sym)

    def token_graph_for_entity(self, entity):
        """Returns the graph proxy (same for all entities in 1830)."""
        return self.graph

    def route_trains(self, entity):
        """Return runnable trains for an entity."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        ci = None
        for i, c in enumerate(self._game.corporations):
            if c.sym == corp_sym:
                ci = i
                break
        if ci is None:
            return []
        corp = self._game.corporations[ci]
        return [t for t in corp.trains if not t.operated]

    def pickle_clone(self):
        """Clone the Rust game and wrap in a new adapter."""
        clone = self._game.pickle_clone()
        adapter = RustGameAdapter(clone)
        # Share the hex adjacency (immutable)
        adapter._hex_adjacency = self._hex_adjacency
        return adapter

    def process_action(self, action):
        """Process action on the underlying Rust game. Returns self (mutates in-place).

        Accepts either a dict (passed directly) or a Python Action object
        (converted via to_dict()).
        """
        if isinstance(action, dict):
            self._game.process_action(action)
        else:
            self._game.process_action(action.to_dict())
        # Invalidate depot proxy since trains may have changed
        self._depot_proxy = None
        return self

    @property
    def actions(self):
        """Return action-like objects with .description() for logging."""
        raw = self._game.raw_actions
        return [_ActionDesc(a) for a in raw] if raw else [_ActionDesc({})]

    def to_dict(self):
        """Minimal dict representation for showboard (non-critical)."""
        return {"move_number": self._game.move_number, "finished": self._game.finished}

    def end_game(self):
        """Trigger end-game scoring in the Rust engine."""
        if hasattr(self._game, 'end_game'):
            self._game.end_game()

    def active_players(self):
        """During OR, return the president of the active corporation."""
        # Try the default first
        result = self._game.active_players()
        if result:
            return [_PlayerProxy(p) for p in result]
        # During OR, find the president of the active corp
        corp = self._game.current_corporation
        if corp:
            pres_id = corp.owner_id_str
            if pres_id.startswith("player:"):
                pid = int(pres_id.split(":")[1])
                for p in self._game.players:
                    if p.id == pid:
                        return [_PlayerProxy(p)]
        return []

    @property
    def round(self):
        """Returns a proxy that provides active_entities and __class__.__name__."""
        return _RoundProxy(self._game)

    @property
    def current_entity(self):
        """Returns the active entity (Player or Corporation) with isinstance compat."""
        player = self._game.current_player
        if player is not None:
            return _PlayerProxy(player)
        corp = self._game.current_corporation
        if corp is not None:
            return _CorpProxy(corp, self._game)
        return None

    @property
    def players(self):
        """Returns players wrapped with proxy for percent_of unwrapping."""
        return [_PlayerProxy(p) for p in self._game.players]

    @property
    def hexes(self):
        """Returns hexes wrapped with neighbor object proxies."""
        adj = self._get_hex_adjacency()
        return [_HexProxy(h, adj) for h in self._game.hexes]

    @property
    def corporations(self):
        """Returns corporations wrapped with callable floated()."""
        return [_CorpProxy(c, self._game) for c in self._game.corporations]

    def corporation_by_id(self, sym):
        corp = self._game.corporation_by_id(sym)
        return _CorpProxy(corp, self._game) if corp else None

    @property
    def depot(self):
        """Depot proxy — recreated each access to reflect current train state.
        Uses identity-stable wrapper so train.owner == game.depot works."""
        # Recreate to reflect current trains, but keep same identity object
        if self._depot_proxy is None:
            self._depot_proxy = _DepotProxy.__new__(_DepotProxy)
        self._depot_proxy._update(self._game.depot)
        return self._depot_proxy

    def hex_by_id(self, coord):
        h = self._game.hex_by_id(coord)
        return _HexProxy(h, self._get_hex_adjacency()) if h else None

    @property
    def companies(self):
        """Companies wrapped with owner proxy."""
        return [_CompanyProxy(c) for c in self._game.companies]

    def company_by_id(self, sym):
        c = self._game.company_by_id(sym)
        return _CompanyProxy(c) if c else None

    @property
    def num_certs(self):
        """Bridge: encoder calls game.num_certs(player) with a Player object."""
        def _num_certs(player):
            if isinstance(player, int):
                return self._game.num_certs(player)
            return self._game.num_certs(player.id)
        return _num_certs

    def active_step(self):
        """Direct access to active step (ActionHelper calls game.active_step())."""
        return self.round.active_step()

    @property
    def share_prices(self):
        """Returns par prices as list of objects with .price, .id, .coordinates."""
        try:
            tuples = self._game.par_prices_with_coords()
            return [_SharePriceProxy(p, r, c) for p, r, c in tuples]
        except AttributeError:
            # Fallback if method not available
            prices = self._game.par_prices()
            return [_SharePriceProxy(p) for p in prices]

    def can_par(self, corp, entity):
        """Check if a corporation can be parred by the entity."""
        corp_obj = corp._corp if isinstance(corp, _CorpProxy) else corp
        if corp_obj.ipo_price is not None:
            return False
        # Check if player can afford minimum par (2 shares at lowest price)
        min_price = min(self._game.par_prices()) if self._game.par_prices() else 0
        player_cash = entity.cash if hasattr(entity, 'cash') else 0
        return player_cash >= min_price * 2

    def buying_power(self, entity):
        """Unified buying_power for both players and corporations."""
        if isinstance(entity, _PlayerProxy):
            return self._game.buying_power_player(entity.id)
        if isinstance(entity, _CorpProxy):
            return self._game.buying_power_corp(entity.sym)
        if hasattr(entity, 'id') and isinstance(entity.id, int):
            return self._game.buying_power_player(entity.id)
        if hasattr(entity, 'sym'):
            return self._game.buying_power_corp(entity.sym)
        return 0

    @property
    def graph(self):
        """Returns a graph proxy with connected_hexes/connected_nodes."""
        return _GraphProxy(self._game, self._get_hex_adjacency())

    def upgrade_cost(self, tile, hex_obj, entity, spender):
        """Returns terrain cost for upgrading a hex."""
        hex_id = hex_obj.id if hasattr(hex_obj, 'id') else str(hex_obj)
        h = self._game.hex_by_id(hex_id)
        if not h:
            return 0
        return sum(u.cost for u in h.tile.upgrades)

    def abilities(self, company, ability_type):
        """Check company abilities. Returns empty list for most cases."""
        if ability_type == "no_buy":
            # MH has no_buy ability when owned by a player
            if hasattr(company, 'sym') and company.sym == "MH":
                return []  # MH doesn't have no_buy
        return []

    def can_go_bankrupt(self, owner, corp):
        """Check if a corporation can go bankrupt."""
        # In 1830, bankruptcy happens when president can't afford any train
        return False  # Conservative: ActionHelper falls back to empty actions

    def discountable_trains_for(self, entity):
        """Returns discountable trains (4→D exchange)."""
        corp_sym = entity.sym if hasattr(entity, 'sym') else str(entity)
        ci = None
        for i, c in enumerate(self._game.corporations):
            if c.sym == corp_sym:
                ci = i
                break
        if ci is None:
            return []
        corp = self._game.corporations[ci]
        has_4 = any(t.name == "4" for t in corp.trains)
        if not has_4:
            return []
        d_train = next((t for t in self.depot.trains if t.name == "D"), None)
        if d_train is None:
            return []
        old_train = next(t for t in corp.trains if t.name == "4")
        # Wrap the D-train as a BuyableTrain so action.train.owner works
        wrapped_d = _BuyableTrain(d_train.id, d_train.name, 800, "depot", self.depot, self._game)
        return [(old_train, wrapped_d, "D", 800)]

    @property
    def stock_market(self):
        """Proxy for stock_market with par_prices property."""
        return _StockMarketProxy(self._game)

    @property
    def share_pool(self):
        """Proxy for share_pool — used for ownership checks."""
        return _SharePoolProxy()

    def get_available_tile_with_name(self, name):
        """Find an available tile by name from the tile catalog."""
        for tile in self._game.tiles:
            if tile.name == name:
                return _TileProxy(tile)
        return None

    # 1830 game constants
    EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST = True
    MUST_BUY_TRAIN = "route"
    ALLOW_TRAIN_BUY_FROM_OTHERS = True
    EBUY_OTHER_VALUE = False
