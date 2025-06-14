__all__ = [
    "Operator",
    "ShareBundle",
    "Share",
    "SharePool",
    "Depot",
    "Train",
    "Company",
    "Bank",
    "Corporation",
    "Loan",
    "Minor",
    "PlayerInfo",
    "Player",
]


from rl18xx.game.engine.core import (
    Entity,
    GameError,
    Ownable,
    Passer,
    ShareHolder,
    Spender,
)
from .graph import Token
from .abilities import Abilities

from collections import defaultdict


class Operator(Entity):
    def __init__(self, **kwargs):
        super().__init__()
        self.cash = 0
        self.trains = []
        self.operating_history = {}
        self.logo_filename = f"{kwargs.get('logo', '')}.svg"
        self.logo = f"/logos/{self.logo_filename}"
        self.simple_logo = f"/logos/{kwargs.get('simple_logo', '')}.svg" if kwargs.get("simple_logo") else self.logo
        self.coordinates = kwargs.get("coordinates")
        self.city = kwargs.get("city")
        self.tokens = [Token(self, price=price) for price in (kwargs.get("tokens", []) or [])]
        self.loans = []
        self.color = kwargs.get("color")
        self.text_color = kwargs.get("text_color", "#ffffff") or "#ffffff"
        self.destination_coordinates = kwargs.get("destination_coordinates")
        self.destination_exits = kwargs.get("destination_exits")
        self.destination_icon = f"/icons/{kwargs.get('destination_icon', '')}" if kwargs.get("destination_icon") else ""
        self.destination_loc = kwargs.get("destination_loc")
        self.destination_icon_in_city_slot = kwargs.get("destination_icon_in_city_slot")

    def is_operator(self):
        return True

    @property
    def runnable_trains(self):
        return [train for train in self.trains if not train.operated]

    def operated(self):
        return bool(self.operating_history)

    def next_token(self):
        return next((token for token in self.tokens if not token.used), None)

    def find_token_by_type(self, type=None):
        type = type or "normal"
        return next(
            (token for token in self.tokens if not token.used and token.type == type),
            None,
        )

    @property
    def tokens_by_type(self):
        seen_types = set()
        unique_tokens = []
        for token in self.tokens:
            if not token.used and token.type not in seen_types:
                unique_tokens.append(token)
                seen_types.add(token.type)
        return unique_tokens

    def unplaced_tokens(self):
        return [token for token in self.tokens if not token.city]

    def placed_tokens(self):
        return [token for token in self.tokens if token.city]


class ShareBundle:
    def __init__(self, shares, percent=None):
        if isinstance(shares, list):
            self.shares = shares
        else:
            self.shares = [shares]

        if not len({share.corporation() for share in self.shares}) == 1:
            raise ValueError("All shares must be from the same corporation")

        if not len({share.owner for share in self.shares}) == 1:
            raise ValueError("All shares must be owned by the same owner")

        self.percent = percent if percent is not None else sum(share.percent for share in self.shares)
        self.share_price = None

    def num_shares(self, ceil=True):
        num = self.percent / self.corporation.share_percent
        return math.ceil(num) if ceil else num

    @property
    def partial(self):
        return self.percent != sum(share.percent for share in self.shares)

    @property
    def corporation(self):
        return self.shares[0].corporation()

    @property
    def owner(self):
        return self.shares[0].owner

    @property
    def president(self):
        return self.corporation.owner

    @property
    def presidents_share(self):
        return next((share for share in self.shares if share.president), None)

    @property
    def preferred(self):
        return any(share.preferred for share in self.shares)

    def price_per_share(self):
        return self.share_price or self.shares[0].price_per_share()

    @property
    def price(self):
        return math.ceil(self.price_per_share() * self.num_shares(ceil=False))

    def can_dump(self, entity):
        if not self.presidents_share:
            return True

        sh = self.corporation.player_share_holders(corporate=True)
        return max((v for k, v in sh.items() if k != entity), default=0) >= self.presidents_share.percent

    def to_bundle(self):
        return self

    @property
    def buyable(self):
        return all(share.buyable for share in self.shares)

    def __eq__(self, other):
        return self.shares == other.shares and self.percent == other.percent and self.share_price == other.share_price

    @property
    def common_percent(self):
        return sum(share.percent for share in self.shares if not share.preferred)

    def __str__(self):
        return f"<Share Bundle> - shares: {self.shares}, corporation: {self.shares[0].corporation()}, owner: {self.shares[0].owner}, percent: {self.percent}, price: {self.share_price}"

    def __repr__(self):
        return self.__str__()


import math


class Share(Ownable):
    def __init__(self, corporation, owner=None, president=False, percent=10, index=0, cert_size=1):
        super().__init__()
        self.cert_size = cert_size
        self._corporation = corporation
        self.president = president
        self.percent = percent
        self.owner = owner or corporation
        self.index = index
        self.buyable = True
        self.counts_for_limit = True
        self.preferred = False
        self.last_cert = False
        self.double_cert = False

    @property
    def id(self):
        return f"{self._corporation.id}_{self.index}"

    # def to_dict(self):
    #     return {
    #         "corporation_id": self._corporation.id,
    #         "id": self.id,
    #         "percent": self.percent,
    #         "owner": self.owner.id,
    #         "index": self.index,
    #     }
    
    # @classmethod
    # def from_dict(cls, game, data):
    #     if not game.corporations:
    #         raise ValueError("Corporations must be loaded before shares")

    #     return cls(
    #         game.corporations_by_id[data["corporation_id"]],
    #         game.players_by_id[data["owner"]],
    #         data["index"]
    #     )

    def __eq__(self, other):
        return (
            isinstance(other, Share)
            and self.percent == other.percent
            and self._corporation == other._corporation
            and self.owner == other.owner
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self.percent, self._corporation, self.owner, self.index))

    def num_shares(self, ceil=True):
        num = self.percent / self._corporation.share_percent
        return math.ceil(num) if ceil else num

    def price_per_share(self):
        share_price = (
            self._corporation.par_price()
            if self.owner == self._corporation.ipo_owner
            else self._corporation.share_price
        )
        return share_price.price * self._corporation.price_multiplier if share_price else self._corporation.min_price

    def corporation(self):
        return self._corporation

    @property
    def price(self):
        return math.ceil(self.price_per_share() * self.num_shares(ceil=False))

    def to_s(self):
        return f"{self.__class__.__name__} - {self.id()}"

    def to_bundle(self, percent=None):
        return ShareBundle(self, percent)

    def __str__(self):
        return f"<Share: {self._corporation.id} {self.percent}%>"

    def __repr__(self):
        return self.__str__()

    def common_percent(self):
        return 0 if self.preferred else self.percent

    def transfer(self, new_entity):
        self.owner.shares_by_corporation[self._corporation].remove(self)
        self._corporation.share_holders[self.owner] -= self.percent
        self.owner = new_entity
        self._corporation.share_holders[new_entity] += self.percent
        new_entity.shares_by_corporation[self._corporation].append(self)


class SharePool(Entity, ShareHolder):
    def __init__(self, game, allow_president_sale=False, no_rebundle_president_buy=False):
        super().__init__()
        self.game = game
        self.bank = game.bank
        self.log = game.log
        self.allow_president_sale = allow_president_sale
        self.no_rebundle_president_buy = no_rebundle_president_buy

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        return "Market"
    
    @name.setter
    # WARNING: This is only present for pickling.
    def name(self, value):
        return

    def player(self):
        return None

    def owner(self):
        return None

    def is_share_pool(self):
        return True

    def buy_shares(
        self,
        entity,
        shares,
        exchange=None,
        exchange_price=None,
        swap=None,
        allow_president_change=True,
        silent=None,
        borrow_from=None,
    ):
        bundle = shares if isinstance(shares, ShareBundle) else ShareBundle(shares)

        if (
            self.allow_president_sale
            and not self.no_rebundle_president_buy
            and bundle.presidents_share
            and bundle.owner == self
        ):
            bundle = ShareBundle(bundle.shares, bundle.corporation.share_percent)

        if (
            bundle.owner.is_player()
            and not self.game.BUY_SHARE_FROM_OTHER_PLAYER
            and (not self.game.CORPORATE_BUY_SHARE_ALLOW_BUY_FROM_PRESIDENT or not entity.is_corporation())
        ):
            raise GameError("Cannot buy share from player")

        corporation = bundle.corporation
        ipoed = corporation.ipoed
        floated = corporation.floated()

        if bundle.presidents_share:
            corporation.ipoed = True

        price = bundle.price
        par_price = corporation.par_price().price if corporation.par_price() else None

        if ipoed != corporation.ipoed and not silent:
            self.log.append(
                f"{entity.name} {self.game.ipo_verb(corporation)} {corporation.name} at "
                f"{self.game.format_currency(par_price)}"
            )

        share_str = f"a {bundle.percent}% share "
        if entity != corporation:
            share_str += f"of {corporation.name}"

        from_ = ""
        if bundle.owner == corporation.ipo_owner:
            from_ = f"the {self.game.ipo_name(corporation)}"
        elif bundle.owner.is_corporation() and bundle.owner == corporation:
            from_ = "the Treasury"
        elif bundle.owner.is_corporation() or bundle.owner.is_player():
            from_ = bundle.owner.name
        else:
            from_ = "the market"

        if exchange:
            price = exchange_price or 0
            if exchange == "free":
                if not silent:
                    self.log.append(f"{entity.name} receives {share_str}")
            elif isinstance(exchange, Company):
                if not silent:
                    if exchange_price:
                        self.log.append(
                            f"{entity.name} exchanges {exchange.name} and "
                            f"{self.game.format_currency(price)} from {from_} for {share_str}"
                        )
                    else:
                        self.log.append(f"{entity.name} exchanges {exchange.name} from {from_} for {share_str}")
        else:
            if swap:
                price -= swap.price

            swap_text = f" + swap of a {swap.percent}% share" if swap else ""
            borrowed = price - entity.cash
            borrowed_text = (
                f" by borrowing {self.game.format_currency(borrowed)} from {borrow_from.name}" if borrowed > 0 else ""
            )

            verb = "redeems" if entity == corporation else "buys"
            if not silent:
                self.log.append(
                    f"{entity.name} {verb} {share_str} from {from_} "
                    f"for {self.game.format_currency(price)}{swap_text}{borrowed_text}"
                )

        if price == 0:
            self.transfer_shares(bundle, entity, allow_president_change=allow_president_change)
        else:
            receiver = None
            if (
                (corporation.capitalization in ("escrow", "incremental") and bundle.owner.is_corporation())
                or (bundle.owner.is_corporation() and not corporation.ipo_is_treasury())
                or (bundle.owner.is_corporation() and bundle.owner != corporation)
                or bundle.owner.is_player()
            ):
                receiver = bundle.owner
            else:
                receiver = self.bank

            self.transfer_shares(
                bundle,
                entity,
                spender=self.bank if entity == self else entity,
                receiver=receiver,
                price=price,
                swap=swap,
                swap_to_entity=self if swap else None,
                borrow_from=borrow_from,
                allow_president_change=allow_president_change,
            )

        if corporation.floatable and floated != corporation.floated():
            self.game.float_corporation(corporation)

    def sell_shares(self, bundle, allow_president_change=True, swap=None, silent=None):
        entity = bundle.owner

        verb = "issues" if entity.is_corporation() and entity == bundle.corporation else "sells"

        price = bundle.price
        if swap:
            price -= swap.price
        price -= self.additional_price_adjustments(bundle)
        swap_text = f" and a {swap.percent}% share" if swap else ""
        swap_to_entity = entity if swap else None

        if not silent:
            self.log_sell_shares(entity, verb, bundle, price, swap_text)

        transfer_to = (
            bundle.corporation if self.game.sold_shares_destination(bundle.corporation) == "corporation" else self
        )

        self.transfer_shares(
            bundle,
            transfer_to,
            spender=self.bank,
            receiver=entity,
            price=price,
            allow_president_change=allow_president_change,
            swap=swap,
            swap_to_entity=swap_to_entity,
        )

    def log_sell_shares(self, entity, verb, bundle, price, swap_text):
        self.log.append(
            f"{entity.name} {verb} {self.num_presentation(bundle)} "
            f"of {bundle.corporation.name} and receives {self.game.format_currency(price)}{swap_text}"
        )

    def additional_price_adjustments(self, bundle):
        return 0

    def fit_in_bank(self, bundle):
        return (bundle.percent + self.percent_of(bundle.corporation)) <= self.game.market_share_limit(
            bundle.corporation
        )

    def bank_at_limit(self, corporation):
        return self.common_percent_of(corporation) >= self.game.market_share_limit(corporation)

    def transfer_shares(
        self,
        bundle,
        to_entity,
        spender=None,
        receiver=None,
        price=None,
        allow_president_change=True,
        swap=None,
        borrow_from=None,
        swap_to_entity=None,
        corporate_transfer=None,
    ):
        corporation = bundle.corporation
        owner = bundle.owner
        previous_president = bundle.president
        price = price if price is not None else bundle.price

        corporation.share_holders[owner] -= bundle.percent
        corporation.share_holders[to_entity] += bundle.percent

        if swap:
            corporation.share_holders[swap.owner] -= swap.percent
            corporation.share_holders[swap_to_entity] += swap.percent
            self.move_share(swap, swap_to_entity)

        if corporation.capitalization == "escrow" and receiver == corporation:
            if corporation.percent_of(corporation) > 50 and spender and price > 0:
                if spender and receiver:
                    spender.spend(price, receiver)
            else:
                spender.spend(price, self.bank)
                corporation.escrow += price
        elif spender and receiver and price > 0:
            spender.spend(price, receiver, borrow_from=borrow_from)

        for share in bundle.shares:
            self.move_share(share, to_entity)

        if not allow_president_change:
            return

        max_shares = self.presidency_check_shares(corporation).values()
        max_shares = max(max_shares) if max_shares else 0

        if (
            self.allow_president_sale
            and max_shares < corporation.presidents_percent
            and bundle.presidents_share
            and to_entity == self
        ):
            corporation.owner = self
            self.log.append(f"President's share sold to pool. {corporation.name} enters receivership")
            if bundle.partial:
                self.handle_partial(bundle, self, owner)
            return

        # set_trace()
        if self.allow_president_sale and owner == self and bundle.presidents_share:
            corporation.owner = to_entity
            self.log.append(f"{to_entity.name} becomes the president of {corporation.name}")
            self.log.append(f"{corporation.name} exits receivership")
            self.handle_partial(bundle, to_entity, self)
            return

        if self.allow_president_sale and max_shares < corporation.presidents_percent:
            return

        majority_share_holders = {
            player: p for player, p in self.presidency_check_shares(corporation).items() if p == max_shares
        }
        if previous_president in majority_share_holders:
            return

        president_candidates = [
            p for p in majority_share_holders if p.percent_of(corporation) >= corporation.presidents_percent
        ]

        president = None
        if president_candidates:
            president = min(
                president_candidates,
                key=lambda p: 0
                if previous_president == self
                else (
                    self.game.player_distance_for_president(previous_president, p)
                    if hasattr(self.game, "player_distance_for_president")
                    else self.distance(previous_president, p)
                ),
            )

        if not president:
            return

        corporation.owner = president
        self.log.append(f"{president.name} becomes the president of {corporation.name}")

        if (
            owner == corporation
            and not bundle.presidents_share
            and self.game.can_swap_for_presidents_share_directly_from_corporation
        ):
            previous_president = previous_president if previous_president else corporation
        if owner == president or not previous_president:
            return

        presidents_share = bundle.presidents_share or next(
            share for share in previous_president.shares_of(corporation) if share.president
        )

        if not presidents_share:
            return

        if owner.is_player() and to_entity.is_player():
            transfer_to = to_entity
            swap_to = to_entity
        else:
            transfer_to = corporation if self.game.sold_shares_destination(corporation) == "corporation" else self
            swap_to = (
                previous_president
                if previous_president.percent_of(corporation) >= presidents_share.percent
                else transfer_to
            )

        self.change_president(presidents_share, swap_to, president, previous_president)

        if bundle.partial:
            self.handle_partial(bundle, transfer_to, owner)

    def handle_partial(self, bundle, from_entity, to_entity):
        corp = bundle.corporation
        difference = sum(share.percent for share in bundle.shares) - bundle.percent
        num_shares = difference / corp.share_percent
        for _ in range(int(num_shares)):
            self.move_share(from_entity.shares_of(corp)[0], to_entity)

    def change_president(self, presidents_share, swap_to, president, _previous_president=None):
        corporation = presidents_share.corporation()
        num_shares = int(presidents_share.percent / corporation.share_percent)

        for s in self.game.shares_for_presidency_swap(
            self.possible_reorder(president.shares_of(corporation)), num_shares
        ):
            self.move_share(s, swap_to)

        self.move_share(presidents_share, president)

    def presidency_check_shares(self, corporation):
        return corporation.player_share_holders()

    def possible_reorder(self, shares):
        return shares

    def distance(self, player_a, player_b):
        if not player_a or not player_b:
            return 0

        entities = self.game.possible_presidents
        a = entities.index(player_a)
        b = entities.index(player_b)
        return b - a if a < b else b - (a - len(entities))

    def num_presentation(self, bundle):
        num_shares = bundle.num_shares()
        return f"a {bundle.percent}% share" if num_shares == 1 else f"{num_shares} shares"

    def move_share(self, share, to_entity):
        corporation = share.corporation()
        share.owner.shares_by_corporation[corporation].remove(share)
        to_entity.shares_by_corporation[corporation].append(share)
        share.owner = to_entity


class Depot(Entity):
    def __init__(self, trains, game):
        super().__init__()

        self.game = game
        self.trains = trains
        for train in self.trains:
            train.owner = self
        self.upcoming = self.trains.copy()
        self.discarded = []
        self.bank = self.game.bank
        self.depot_trains_cache = None

    def export(self):
        train = self.upcoming[0]
        self.game.log.append(f"-- Event: A {train.name} train exports --")
        self.game.remove_train(train)
        self.game.phase.buying_train(None, train, self)

    def export_all(self, name, silent=False):
        if not silent:
            self.game.log.append(f"-- Event: All {name} trains are exported --")
        while self.upcoming and self.upcoming[0].name == name:
            train = self.upcoming[0]
            self.game.remove_train(train)
            self.game.phase.buying_train(None, train, self)

    def reclaim_all(self, name):
        self.game.log.append(f"-- Event: All {name} trains are discarded to the Bank Pool --")
        while self.upcoming and self.upcoming[0].name == name:
            train = self.upcoming[0]
            self.reclaim_train(train)
            self.game.phase.buying_train(None, train, self)

    def reclaim_train(self, train):
        if train.owner is None:
            return
        self.game.remove_train(train)
        train.owner = self
        if self.game.discarded_train_placement() == "discard" and not train.obsolete:
            self.discarded.append(train)
        self.depot_trains_cache = None

    def min_price(self, corporation, ability=None):
        return min(train.min_price(ability=ability) for train in self.available(corporation))

    @property
    def min_depot_train(self):
        return min(self.depot_trains(), key=lambda train: train.price)

    @property
    def min_depot_price(self):
        train = self.min_depot_train
        return min(variant["price"] for variant in train.variants.values()) if train else 0

    @property
    def max_depot_price(self):
        train = max(self.depot_trains(), key=lambda train: train.price, default=None)
        return max(variant["price"] for variant in train.variants.values()) if train else 0

    def unshift_train(self, train):
        train.owner = self
        self.upcoming.insert(0, train)
        self.depot_trains_cache = None

    def remove_train(self, train):
        try:
            self.upcoming.remove(train)
        except ValueError:
            pass
        try:
            self.discarded.remove(train)
        except ValueError:
            pass
        self.depot_trains_cache = None

    def forget_train(self, train):
        try:
            self.trains.remove(train)
        except ValueError:
            pass
        try:
            self.upcoming.remove(train)
        except ValueError:
            pass
        try:
            self.discarded.remove(train)
        except ValueError:
            pass
        self.depot_trains_cache = None

    def add_train(self, train):
        train.owner = self
        self.trains.append(train)
        self.upcoming.append(train)
        self.depot_trains_cache = None

    def insert_train(self, train, index=0):
        train.owner = self
        self.trains.append(train)
        self.upcoming.insert(index, train)
        self.depot_trains_cache = None

    def depot_trains(self, clear=False):
        if clear or self.depot_trains_cache is None:
            self.depot_trains_cache = [self.upcoming[0]] + [
                t for t in self.upcoming if self.game.phase.available(t.available_on)
            ]
            self.depot_trains_cache = list(dict.fromkeys(self.depot_trains_cache + self.discarded).keys())
        return self.depot_trains_cache

    def available(self, corporation):
        return self.depot_trains() + self.other_trains(corporation)

    def other_trains(self, corporation):
        all_others = [train for train in self.trains if train.buyable and train.owner not in [corporation, self, None]]
        if not self.game.ALLOW_TRAIN_BUY_FROM_OTHER_PLAYERS:
            all_others = [train for train in all_others if train.owner.owner == corporation.owner]
        return all_others

    @property
    def cash(self):
        return self.bank.cash

    @cash.setter
    def cash(self, new_cash):
        self.bank.cash = new_cash

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        return "The Depot"
    
    @name.setter
    # WARNING: This is only present for pickling.
    def name(self, value):
        return

    def empty(self):
        return not self.depot_trains()

    def player(self):
        return None


from typing import Union, List


class Train(Ownable):
    def __init__(
        self,
        name: str,
        distance: Union[int, List[dict]],
        price: int,
        index: int = 0,
        **opts,
    ):
        super().__init__()
        self.sym = name
        self.name = name
        self.distance = distance
        self.price = price
        self.index = index
        self.rusts_on = opts.get("rusts_on", None)
        self.obsolete_on = opts.get("obsolete_on", None)
        self.available_on = opts.get("available_on", None)
        self.discount = opts.get("discount", None)
        self.salvage = opts.get("salvage", None)
        self.multiplier = opts.get("multiplier", None)
        self.no_local = opts.get("no_local", None)
        self.buyable = True
        self.rusted = False
        self.obsolete = False
        self.operated = False
        self.ever_operated = False
        self.track_type = opts.get("track_type", "broad")
        self.events = [e for e in (opts.get("events", []) or []) if self.index == (e.get("when", 1) - 1)]
        self.reserved = opts.get("reserved", False)
        self.requires_token = opts.get("requires_token", True)
        self.init_variants(opts.get("variants", []))

    def set_operated(self, value):
        self.ever_operated = value
        self.operated = value

    @property
    def variant(self):
        return self._variant

    @variant.setter
    def variant(self, variant):
        if not variant:
            return

        self._variant = self.variants[variant]

        for key, value in self._variant.items():
            setattr(self, key, value)

        if hasattr(self, "local"):
            delattr(self, "local")

    def init_variants(self, variants):
        variants = variants or []
        self._variant = {
            "name": self.name,
            "distance": self.distance,
            "multiplier": self.multiplier,
            "price": self.price,
            "rusts_on": self.rusts_on,
            "obsolete_on": self.obsolete_on,
            "discount": self.discount,
            "salvage": self.salvage,
            "track_type": self.track_type,
        }
        variants.insert(0, self.variant)
        self.variants = {v["name"]: v for v in variants}

    def remove_variants(self):
        self.variants = {name: variant for name, variant in self.variants.items() if name == self.name}

    def names_to_prices(self):
        return {name: variant["price"] for name, variant in self.variants.items()}

    def get_price(self, exchange_train=None, variant=None):
        discount = variant["discount"] if variant else self.discount
        price = variant["price"] if variant else self.price
        return price - (discount.get(exchange_train.name, 0) if discount and exchange_train else 0)

    @property
    def id(self):
        return f"{self.name}-{self.index}"

    def min_price(self, ability=None):
        if not self.from_depot():
            return 1

        if not ability:
            return self.price

        return min(
            [a.discounted_price(self, self.price) for a in (ability if isinstance(ability, list) else [ability])]
        )

    def from_depot(self):
        return isinstance(self.owner, Depot)

    def is_buyable(self, allow_obsolete_buys=False):
        return self.buyable and (not self.obsolete or allow_obsolete_buys)

    def is_local(self):
        if self.no_local:
            return False
        if hasattr(self, "local"):
            return self.local

        self.local = (
            self.distance == 1
            if isinstance(self.distance, int)
            else any(n["visit"] == 1 for n in self.distance if n.get("nodes", []).count("city") > 0)
        )
        return self.local

    def __str__(self):
        return f"<Train: {self.id}, owner: {self.owner}>"

    def __repr__(self):
        return f"<Train: {self.id}, owner: {self.owner}>"


import copy


class Company(Entity, Ownable, Passer, Abilities):
    def __init__(self, sym=None, name=None, value=None, revenue=0, desc="", abilities=[], **opts):
        Entity.__init__(self)
        Ownable.__init__(self)
        Passer.__init__(self)
        Abilities.__init__(self, abilities)

        self.sym = sym
        self.name = name
        self.value = value
        self.treasury = opts.get("treasury", value)
        self.desc = desc
        self.revenue = revenue
        self.discount = opts.get("discount", 0)
        self.min_auction_price = -self.discount
        self.closed = False
        self.min_price = opts.get("min_price", None if value is None else (value // 2) + (value % 2))
        self.max_price = opts.get("max_price", None if value is None else value * 2)
        self.interval = opts.get("interval", None)  # Array of prices or None
        self.color = opts.get("color", "yellow")
        self.text_color = opts.get("text_color", "black")
        self.type = opts.get("type", None)
        if self.type is not None:
            self.type = self.type
        self.auction_row = opts.get("auction_row", None)
        self.opts = opts

    def __copy__(self):
        copied_abilities = [copy.copy(ability) for ability in self.abilities]
        return Company(
            sym=self.sym,
            name=self.name,
            value=self.value,
            revenue=self.revenue,
            desc=self.desc,
            abilities=copied_abilities,
            **self.opts,
        )

    def __lt__(self, other):
        return (self.min_bid, self.name) < (other.min_bid, other.name)

    def __eq__(self, other):
        return (
            isinstance(other, Company)
            and self.sym == other.sym
            and self.name == other.name
            and self.value == other.value
            and self.discount == other.discount
        )

    def __hash__(self):
        return hash((self.sym, self.name, self.value, self.discount))

    @property
    def id(self):
        return self.sym
    
    @id.setter
    # WARNING: This is only present for pickling.
    def id(self, value):
        self.sym = value

    @property
    def min_bid(self):
        return self.value - self.discount

    def close(self):
        # set_trace()
        self.closed = True
        for ability in self.all_abilities:
            self.remove_ability(ability)
        if self.owner:
            if hasattr(self.owner, "companies"):
                self.owner.companies.remove(self)
            self.owner = None

    def is_closed(self):
        return self.closed

    def is_company(self):
        return True

    def is_path(self):
        return False

    # Token handling would need more context about the Token class and related methods
    def find_token_by_type(self, token_type):
        token_ability = next((a for a in self.all_abilities if a.type == "token"), None)
        if token_ability is None:
            raise Exception(f"{self.name} does not have a token")
        if token_ability.from_owner:
            return self.owner.find_token_by_type(token_type)
        return Token(self.owner)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.id}>"

    def get_max_price(self, buyer=None):
        return self.max_price


class Bank(Entity, Spender, ShareHolder):
    def __init__(self, cash, log=None, check=True):
        Entity.__init__(self)
        Spender.__init__(self)
        ShareHolder.__init__(self)

        self.cash = cash
        self.log = log
        if self.log is None:
            self.log = []
        self.broken = False
        self.companies = []
        self.check = check

    def check_cash(self, amount, borrow_from=None):
        if not self.check:
            return
        if self.cash - amount < 0:
            self.break_bank()

    def break_bank(self):
        if not self.broken:
            self.log.append("-- The bank has broken --")
            self.broken = True

    def is_broken(self):
        return self.broken

    def player(self):
        return None

    @property
    def name(self):
        return "The Bank"
    
    @name.setter
    # WARNING: This is only present for pickling.
    def name(self, value):
        return

    def __str__(self):
        return f"<{self.__class__.__name__}>"


class Corporation(Abilities, Operator, Entity, Ownable, Passer, ShareHolder, Spender):
    SHARES = [20] + [10] * 8

    def __init__(self, sym=None, name=None, **kwargs):
        Entity.__init__(self)
        Ownable.__init__(self)
        Passer.__init__(self)
        ShareHolder.__init__(self)
        Spender.__init__(self)

        Abilities.__init__(self, kwargs.get("abilities", []))
        Operator.__init__(self, **kwargs)

        self.name = sym
        self.full_name = name

        self.ipo_owner = kwargs.get("ipo_owner", self)
        corp_shares = [
            Share(
                self,
                owner=self.ipo_owner,
                president=(index == 0),
                percent=percent,
                index=index,
            )
            for index, percent in enumerate(kwargs.get("shares", self.SHARES))
        ]
        self.corp_shares = corp_shares
        for share in corp_shares:
            self.ipo_owner.shares_by_corporation.setdefault(self, []).append(share)
        self.share_holders = defaultdict(int)
        self.share_holders[self.ipo_owner] = sum(share.percent for share in corp_shares)

        self.fraction_shares = kwargs.get(
            "fraction_shares",
            any(percent not in list(map(lambda x: x.percent, corp_shares[0:2])) for percent in corp_shares),
        )
        self.presidents_share = corp_shares[0]
        self.second_share = corp_shares[1] if len(corp_shares) > 1 else None

        self.share_price = None
        self._par_price = None
        self.original_par_price = None
        self.ipoed = False
        self.companies = []
        self.cash = 0
        self.capitalization = kwargs.get("capitalization", "full")
        self.closed = False
        self.float_percent = kwargs.get("float_percent", 60)
        self.float_excludes_market = kwargs.get("float_excludes_market", False)
        self.float_includes_reserved = kwargs.get("float_includes_reserved", False)
        self.floatable = kwargs.get("floatable", True)
        self._floated = False
        self.max_ownership_percent = kwargs.get("max_ownership_percent", 60)
        self.min_price = kwargs.get("min_price")
        self.always_market_price = kwargs.get("always_market_price", False)
        self.needs_token_to_par = kwargs.get("needs_token_to_par", False)
        self.par_via_exchange = None
        self.type = kwargs.get("type", None)
        self.hide_shares = kwargs.get("hide_shares", False)
        self.reservation_color = kwargs.get("reservation_color", None)
        self.price_percent = kwargs.get(
            "price_percent",
            self.second_share.percent if self.second_share else self.presidents_share.percent / 2,
        )
        self.price_multiplier = (
            self.second_share.percent if self.second_share else self.presidents_share.percent / 2
        ) / self.price_percent
        self.treasury_as_holding = kwargs.get("treasury_as_holding", False)
        self.corporation_can_ipo = kwargs.get("corporation_can_ipo", None)

    def can_buy(self):
        return True

    def __lt__(self, other):
        self_key = self.sort_order_key()
        other_key = other.sort_order_key()
        if self_key is None:
            return -1
        if other_key is None:
            return 1
        return self_key < other_key

    def sort_order_key(self):
        if self.share_price is None:
            return None
        return [
            -self.share_price.price,
            -self.share_price.coordinates[-1],
            self.share_price.coordinates[0],
            self.share_price.corporations.index(self) if self in self.share_price.corporations else 0,
            self.name,
        ]

    @property
    def id(self):
        return self.name
    
    @id.setter
    # WARNING: This is only present for pickling.
    def id(self, value):
        self.name = value

    def counts_for_limit(self):
        return True if self.share_price is None else self.share_price.counts_for_limit()

    def buy_multiple(self):
        return self.share_price.buy_multiple() if self.share_price else False

    def hide_shares(self):
        return self.hide_shares

    def par_price(self):
        if self.is_closed():
            return None
        return self.share_price if self.always_market_price else self._par_price

    @property
    def total_shares(self):
        return int(100 / self.share_percent)

    def num_ipo_shares(self):
        return self.ipo_owner.num_shares_of(self)

    def reserved_shares(self):
        return [share for share in self.ipo_owner.shares_by_corporation[self] if not share.buyable]

    def num_ipo_reserved_shares(self):
        return sum(share.percent for share in self.reserved_shares()) / self.share_percent

    def num_treasury_shares(self):
        return 0 if self.treasury_as_holding else self.num_shares_of(self)

    def num_player_shares(self):
        return sum(holder_value for holder, holder_value in self.player_share_holders().items()) / self.share_percent

    def num_corporate_shares(self):
        return sum(holder_value for holder, holder_value in self.corporate_share_holders().items()) / self.share_percent

    def num_market_shares(self):
        return int(
            sum(holder_value for holder, holder_value in self.share_holders.items() if holder.is_share_pool())
            / self.share_percent
        )

    def player_share_holders(self, corporate=False):
        if corporate:
            return {
                holder: holder_value
                for holder, holder_value in self.share_holders.items()
                if holder
                and holder.is_player()
                or (corporate and self.corporation_can_ipo and holder.corporation() and holder != self)
            }
        return {
            holder: holder_value for holder, holder_value in self.share_holders.items() if holder and holder.is_player()
        }

    def ipo_is_treasury(self):
        return self.ipo_owner == self

    def corporate_share_holders(self):
        return {
            holder: holder_value
            for holder, holder_value in self.share_holders().items()
            if holder.corporation() and (holder != self or self.treasury_as_holding)
        }

    def corporate_shares(self):
        return [share for share in self.shares if share.corporation() == self and not self.treasury_as_holding]

    @property
    def ipo_shares(self):
        return [share for share in self.ipo_owner.shares if share.corporation() == self]

    @property
    def market_shares(self):
        return [share for share in self.corp_shares if share.owner.is_share_pool()]

    def treasury_shares(self):
        return [share for share in self.shares if share.corporation() == self and not self.treasury_as_holding]

    def president(self, player):
        if player is None:
            return False
        return self.owner == player

    def floated(self):
        if not self.floatable:
            return False
        self._floated = self._floated or (
            self.ipo_owner.percent_of(self)
            <= (
                100
                - self.float_percent
                - (self.percent_in_market() if self.float_excludes_market else 0)
                + (self.percent_in_reserved() if self.float_includes_reserved else 0)
            )
        )
        return self._floated

    def percent_to_float(self):
        if self.floated():
            return 0
        return self.ipo_owner.percent_of(self) - (
            100
            - self.float_percent
            - (self.percent_in_market() if self.float_excludes_market else 0)
            + (self.percent_in_reserved() if self.float_includes_reserved else 0)
        )

    def percent_in_market(self):
        return self.num_market_shares() * self.share_percent

    def percent_in_reserved(self):
        return self.num_ipo_reserved_shares() * self.share_percent

    def unfloat(self):
        self._floated = False

    def is_corporation(self):
        return True

    def is_receivership(self):
        return self.owner().share_pool() if self.owner() else False

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.id}>"

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.id}>"

    def holding_ok(self, share_holder, extra_percent=0):
        common_percent = share_holder.common_percent_of(self) + extra_percent
        return (
            self.share_price and self.share_price.type in ("multiple_buy", "unlimited")
        ) or common_percent <= self.max_ownership_percent

    @property
    def all_abilities(self):
        all_abilities = sum((company.all_abilities for company in self.companies), []) + self.abilities
        if self.owner and hasattr(self.owner, "companies"):
            all_abilities += [
                ability
                for company in self.owner.companies
                for ability in company.all_abilities
                if "owning_player" in str(ability.when)
            ]
        return all_abilities

    def remove_ability(self, ability):
        if ability.owner == self:
            super().remove_ability(ability)
        else:
            for company in self.companies:
                company.remove_ability(ability)

    @property
    def available_share(self):
        return next(
            (share for share in self.shares_by_corporation[self] if not share.president),
            None,
        )

    @property
    def presidents_percent(self):
        return self.presidents_share.percent

    @property
    def share_percent(self):
        return (
            self.forced_share_percent
            if hasattr(self, "forced_share_percent")
            else (self.second_share.percent if self.second_share else self.presidents_share.percent / 2)
        )

    def player(self):
        chain = {self.owner: True}
        current = self.owner
        while current and current.is_corporation():
            if not current.owner:
                return None
            current = current.owner
            if current in chain:
                return None
            chain[current] = True
        return current.player() if current and current.is_player() else None

    def is_closed(self):
        return self.closed

    def close(self):
        if self.share_price:
            self.share_price.corporations().remove(self)
        self.closed = True
        self.ipoed = False
        self._floated = False
        self.owner = None

    def reopen(self):
        self.closed = False


class Loan(Ownable):
    def __init__(self, id, amount):
        self.id = id
        self.amount = amount


class Minor(Abilities, Operator, Entity, Ownable, Passer, Spender):
    def __init__(self, sym, name, abilities=None, **kwargs):
        Entity.__init__(self)
        Ownable.__init__(self)
        Passer.__init__(self)
        Spender.__init__(self)

        Abilities.__init__(self, kwargs.get("abilities", []))
        Operator.__init__(self, **kwargs)

        self.name = sym
        self.full_name = name
        self.floated = False
        self.closed = False
        self.type = kwargs.get("type")
        self.reservation_color = kwargs.get("reservation_color")

    def companies(self):
        return []

    @property
    def id(self):
        return self.name

    def is_minor(self):
        return True

    @property
    def total_shares(self):
        return 1

    def is_floated(self):
        return self.floated

    def float(self):
        self.floated = True

    def inspect(self):
        return f"<{self.__class__.__name__}: {self.id()}>"

    def is_closed(self):
        return self.closed

    def share_price(self):
        pass

    def par_price(self):
        pass

    def num_shares_of(self, corporation, ceil=True):
        return 0

    def share_percent(self):
        return 100

    def president(self, player):
        if player:
            return self.owner() == player
        return False

    def close(self):
        self.closed = True
        self.floated = False
        self.owner = None

    def reopen(self):
        self.closed = False


class PlayerInfo:
    def __init__(self, round_name, turn, round_no, player_value):
        self.round_name = round_name
        self.turn = turn
        self.round_no = round_no
        self.value = player_value

    def round(self):
        if self.round_name in ["AR", "MR", "OR", "DEV", "BUST"]:
            return f"{self.round_name} {self.turn}.{self.round_no}"
        else:
            return f"{self.round_name} {self.turn}"


class Player(Entity, Passer, ShareHolder, Spender):
    def __init__(self, id, name):
        Entity.__init__(self)
        Passer.__init__(self)
        ShareHolder.__init__(self)
        Spender.__init__(self)
        self.id = id
        self.name = name
        self.bankrupt = False
        self.companies = []
        self.history = []
        self.unsold_companies = []

    @property
    def value(self):
        return (
            self.cash
            + sum(s.price for s in self.shares if s.corporation().ipoed)
            + sum(c.value for c in self.companies)
        )

    @property
    def owner(self):
        return self

    def player(self):
        return self

    def corporation(self):
        return None

    def __eq__(self, other):
        return isinstance(other, Player) and self.name == other.name

    def is_player(self):
        return True

    def __str__(self):
        return f"{self.__class__.__name__} - {self.name}"

    def __repr__(self):
        return f"<{self.__class__.__name__} - {self.name}>"

    def __hash__(self):
        return hash((self.id, self.name))
    
    # def to_dict(self):
    #     return {
    #         "id": self.id,
    #         "name": self.name,
    #         "cash": self.cash,
    #         "shares": [share.to_dict() for share in self.shares],
    #     }

    # @classmethod
    # def from_dict(cls, game, data):
    #     player = cls(data["id"], data["name"])
    #     player.cash = data["cash"]
    #     player.shares = [Share.from_dict(game, share) for share in data["shares"]]
    #     return player
