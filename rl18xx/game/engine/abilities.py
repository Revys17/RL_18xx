__all__ = [
    "AbilityBase",
    "AcquireCompany",
    "AdditionalToken",
    "AssignCorporation",
    "AssignHexes",
    "BlocksHexes",
    "BlocksHexesConsent",
    "BlocksPartition",
    "BorrowTrain",
    "ChooseAbility",
    "Close",
    "Description",
    "Exchange",
    "Generic",
    "HexBonus",
    "ManualCloseCompany",
    "NoBuy",
    "PurchaseTrain",
    "Reservation",
    "ReturnToken",
    "RevenueChange",
    "SellCompany",
    "Shares",
    "Teleport",
    "TileDiscount",
    "TileIncome",
    "TileLay",
    "Token",
    "TrainBuy",
    "TrainDiscount",
    "TrainLimit",
    "TrainScrapper",
    "Abilities",
]


from .core import Ownable, snake_to_pascal


class AbilityBase(Ownable):
    def __init__(
        self,
        type=None,
        description=None,
        desc_detail=None,
        owner_type=None,
        count=None,
        remove=None,
        use_across_ors=None,
        count_per_or=None,
        passive=None,
        on_phase=None,
        after_phase=None,
        **opts,
    ):
        self.type = type
        self.description = description
        self.desc_detail = desc_detail
        self.owner_type = owner_type
        when = opts.get("when", [])
        if not isinstance(when, list):
            when = [when]
        self.when = when
        self.on_phase = on_phase
        self.after_phase = after_phase
        self.count = count
        self.count_per_or = count_per_or
        self.count_this_or = 0
        self.use_across_ors = True if use_across_ors is None else use_across_ors
        self.used = False
        self.remove = remove
        self.start_count = self.count
        self.passive = passive if passive is not None else len(self.when) == 0

    def used(self):
        return self.used

    def use(self, **kwargs):
        self.used = True

        if self.count_per_or:
            self.count_this_or += 1

        if self.count is not None:
            self.count -= 1
            if self.count <= 0:
                self.owner.remove_ability(self)

    def use_up(self):
        while self.count > 0:
            self.use()

    def teardown(self):
        pass

    def matches_when(self, *times):
        return bool(set(self.when) & set(times))


class AcquireCompany(AbilityBase):
    def __init__(self, company, **kwargs):
        super().__init__(**kwargs)
        self.company = company


class AdditionalToken(AbilityBase):
    pass


class AssignCorporation(AbilityBase):
    def __init__(self, closed_when_used_up, **kwargs):
        super().__init__(**kwargs)
        self.closed_when_used_up = closed_when_used_up


class AssignHexes(AbilityBase):
    def __init__(self, hexes, closed_when_used_up=None, cost=0, **kwargs):
        super().__init__(**kwargs)
        self.hexes = hexes
        self.closed_when_used_up = closed_when_used_up
        self.cost = cost


class BlocksHexes(AbilityBase):
    def __init__(self, hexes, hidden=False, **kwargs):
        super().__init__(**kwargs)
        self.hexes = hexes
        self.hidden = hidden


class BlocksHexesConsent(AbilityBase):
    def __init__(self, hexes, hidden=False, **kwargs):
        super().__init__(**kwargs)
        self.hexes = hexes
        self.hidden = hidden


class BlocksPartition(AbilityBase):
    def __init__(self, partition_type, **kwargs):
        super().__init__(**kwargs)
        self.partition_type = partition_type

    def blocks(self, partition_type):
        return self.partition_type == partition_type


class BorrowTrain(AbilityBase):
    def __init__(self, train_types, **kwargs):
        super().__init__(**kwargs)
        self.train_types = train_types


class ChooseAbility(AbilityBase):
    def __init__(self, choices=[], **kwargs):
        super().__init__(**kwargs)
        self.choices = choices


class Close(AbilityBase):
    def __init__(self, corporation=None, silent=False, **kwargs):
        super().__init__(**kwargs)
        self.corporation = corporation
        self.silent = silent


class Description(AbilityBase):
    pass


class Exchange(AbilityBase):
    def __init__(self, corporations, from_, **kwargs):
        super().__init__(**kwargs)
        self.corporations = corporations
        self.from_ = from_


class Generic(AbilityBase):
    def __init__(self, subtype, from_, **kwargs):
        super().__init__(**kwargs)
        self.type = subtype


class HexBonus(AbilityBase):
    def __init__(self, hexes, amount, **kwargs):
        super().__init__(**kwargs)
        self.hexes = hexes
        self.amount = amount


class ManualCloseCompany(AbilityBase):
    pass


class NoBuy(AbilityBase):
    pass


class PurchaseTrain(AbilityBase):
    def __init__(self, free=False, **kwargs):
        super().__init__(**kwargs)
        self.free = free


class Reservation(AbilityBase):
    def __init__(self, hex, city=0, slot=0, tile=None, icon=None, **kwargs):
        super().__init__(**kwargs)
        self.hex = hex
        self.city = city
        self.slot = slot
        self.tile = tile
        self.icon = f"/icons/{icon}.svg" if icon else None

    def teardown(self):
        if self.tile:
            self.tile.cities[self.city].remove_reservation(self.owner)


class ReturnToken(AbilityBase):
    def __init__(self, reimburse=False, **kwargs):
        super().__init__(**kwargs)
        self.reimburse = reimburse


class RevenueChange(AbilityBase):
    def __init__(self, revenue, **kwargs):
        super().__init__(**kwargs)
        self.revenue = revenue


class SellCompany(AbilityBase):
    pass


class Shares(AbilityBase):
    def __init__(self, shares, corporations=None, **kwargs):
        super().__init__(**kwargs)
        self.shares = list(shares) if isinstance(shares, (list, tuple)) else [shares]
        self.corporations = corporations


class Teleport(AbilityBase):
    def __init__(
        self,
        hexes,
        tiles,
        cost=None,
        free_tile_lay=False,
        from_owner=True,
        extra_action=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hexes = hexes
        self.tiles = tiles
        self.cost = cost
        self.free_tile_lay = free_tile_lay
        # If the 'when' list is empty, default it to ['track']
        if not self.when:
            self.when = ["track"]
        self.passive = False
        self.from_owner = from_owner
        self.extra_action = extra_action


class TileDiscount(AbilityBase):
    def __init__(self, discount, terrain=None, hexes=None, exact_match=True, **kwargs):
        super().__init__(**kwargs)
        self.discount = discount
        self.terrain = terrain
        self.hexes = hexes
        self.exact_match = exact_match

    def discounts_tile(self, tile):
        if self.exact_match:
            return tile.terrain == [self.terrain]
        return self.terrain in tile.terrain


class TileIncome(AbilityBase):
    def __init__(self, income, terrain=None, owner_only=False, **kwargs):
        super().__init__(**kwargs)
        self.income = income
        self.terrain = terrain
        self.owner_only = owner_only


class TileLay(AbilityBase):
    def __init__(
        self,
        tiles,
        hexes=None,
        free=False,
        discount=0,
        special=True,
        connect=True,
        blocks=False,
        reachable=False,
        must_lay_together=False,
        cost=0,
        closed_when_used_up=False,
        must_lay_all=False,
        consume_tile_lay=False,
        lay_count=None,
        upgrade_count=None,
        combo_entities=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hexes = hexes
        self.tiles = tiles
        self.free = free
        self.discount = discount
        self.special = special
        self.connect = connect
        self.blocks = blocks
        self.reachable = reachable
        self.must_lay_together = must_lay_together
        self.must_lay_all = must_lay_all and must_lay_together
        self.cost = cost
        self.closed_when_used_up = closed_when_used_up
        self.consume_tile_lay = consume_tile_lay
        self.laid_hexes = []
        self.lay_count = lay_count
        self.upgrade_count = upgrade_count
        if lay_count is not None:
            self.count = lay_count
        self.start_count = self.count
        self.combo_entities = combo_entities

    def use(self, upgrade=False):
        if self.count is not None and self.count <= 0:
            return

        super().use()

        if self.upgrade_count is not None and self.lay_count is not None:
            if upgrade:
                if self.upgrade_count <= 0:
                    raise ValueError("Cannot use this ability to upgrade a tile now")

                self.lay_count = 0
                self.upgrade_count -= 1
                if self.upgrade_count <= 0:
                    self.owner.remove_ability(self)
                    self.count = 0
            else:
                if self.lay_count <= 0:
                    raise ValueError("Cannot use this ability to lay a tile now")

                self.upgrade_count = 0
                self.lay_count -= 1
                if not self.lay_count > 0:
                    self.owner.remove_ability(self)
                    self.count = 0


class Token(AbilityBase):
    def __init__(
        self,
        hexes,
        price=None,
        teleport_price=None,
        extra_action=False,
        from_owner=False,
        discount=None,
        city=None,
        neutral=False,
        cheater=False,
        extra_slot=False,
        special_only=False,
        check_tokenable=True,
        closed_when_used_up=False,
        connected=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hexes = hexes
        self.price = price
        self.teleport_price = teleport_price
        self.extra_action = extra_action
        self.from_owner = from_owner
        self.discount = discount
        self.city = city
        self.neutral = neutral
        self.cheater = cheater
        self.extra_slot = extra_slot
        self.special_only = special_only
        self.check_tokenable = check_tokenable
        self.closed_when_used_up = closed_when_used_up
        self.connected = connected

    def get_price(self, token=None):
        if token is None or self.discount is None:
            return self.price
        return token.price - (token.price * self.discount)


class TrainBuy(AbilityBase):
    def __init__(self, face_value=None, **kwargs):
        super().__init__(**kwargs)
        self.face_value = bool(face_value)


class TrainDiscount(AbilityBase):
    def __init__(self, discount, trains, closed_when_used_up=None, **kwargs):
        super().__init__(**kwargs)
        self.discount = discount
        self.trains = trains
        self.closed_when_used_up = closed_when_used_up

    def discounted_price(self, train, price):
        # If trains are specified and the current train is not in the list, return the original price
        if self.trains and train.name not in self.trains:
            return price

        # Calculate discount value based on whether the discount is a flat value or a percentage
        discount_value = self.discount[train.name] if isinstance(self.discount, dict) else self.discount

        # Apply the discount to the price
        return price - (discount_value if discount_value > 1 else int(price * discount_value))


class TrainLimit(AbilityBase):
    def __init__(self, increase=None, constant=None, **kwargs):
        super().__init__(**kwargs)
        self.increase = increase
        self.constant = constant


class TrainScrapper(AbilityBase):
    def __init__(self, scrap_values={}, **kwargs):
        super().__init__(**kwargs)
        self.scrap_values = scrap_values

    def scrap_value(self, train):
        return self.scrap_values.get(train.name, 0)


class Abilities:
    def __init__(self, abilities=[]):
        self._abilities = []

        for ability in abilities:
            if not isinstance(ability, AbilityBase):
                class_name = snake_to_pascal(ability["type"])
                if ability.get("from"):
                    ability["from_"] = ability.pop("from")
                ability_instance = globals()[class_name](**ability)
                ability_instance.owner = self
            else:
                ability_instance = ability
            self._abilities.append(ability_instance)

        self._update_start_counter()

    @property
    def abilities(self):
        return self._abilities

    def set_owner(self, owner):
        self.owner = owner

    def add_ability(self, ability):
        ability.owner = self.owner
        self._abilities.append(ability)
        self._update_start_counter()

    def remove_ability(self, ability):
        ability.teardown()
        self._abilities = [a for a in self._abilities if a != ability]

    def remove_ability_when(self, time):
        for ability in self._abilities[:]:
            if ability.remove == str(time):
                self.remove_ability(ability)

    @property
    def all_abilities(self):
        return self._abilities

    def reset_ability_count_this_or(self):
        for ability in self._abilities:
            ability.count_this_or = 0
            if ability.used and not ability.use_across_ors:
                ability.use_up()

    def ability_uses(self):
        if self._start_count is None:
            return

        count = [0, self._start_count]
        for ability in self._abilities:
            if ability.start_count is not None:
                count = max(count, [ability.count, ability.start_count], key=lambda x: x[1])

        return count

    def _update_start_counter(self):
        start_counts = [ability.start_count for ability in self._abilities if ability.start_count is not None]
        self._start_count = max(start_counts, default=None)
