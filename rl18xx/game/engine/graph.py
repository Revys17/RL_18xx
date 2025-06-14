__all__ = [
    "copy_dict",
    "BasePart",
    "Node",
    "Path",
    "Edge",
    "Frame",
    "FutureLabel",
    "Icon",
    "Junction",
    "Label",
    "Partition",
    "Stripes",
    "Stub",
    "Upgrade",
    "RevenueCenter",
    "Border",
    "City",
    "Town",
    "Halt",
    "Offboard",
    "Pass",
    "Token",
    "Graph",
    "DistanceGraph",
    "Route",
    "TileConfig",
    "Tile",
    "Hex",
]

import copy
from .core import GameError, Ownable
import itertools
from collections import defaultdict


def copy_dict(dict):
    return {key: value for key, value in dict.items()}


class BasePart:
    def __init__(self, tile=None, index=None, loc=None, **kwargs):
        self._tile = tile
        self.index = index
        self.loc = loc

    @property
    def id(self):
        return f"{self.tile.id if self.tile else None}-{self.index}"

    @property
    def signature(self):
        return f"{self.hex.id}-{self.index}" if self.hex and self.index is not None else None

    @property
    def hex(self):
        return self.tile.hex if self.tile else None

    @property
    def tile(self):
        return self._tile

    @tile.setter
    def tile(self, value):
        self._tile = value

    def __le__(self, other):
        return isinstance(self, other.__class__)

    def __hash__(self):
        return hash((self.tile, self.loc, self.index))

    def rotate(self, ticks):
        return self

    def blocks(self, corporation):
        return False

    def tokened_by(self, corporation):
        return False

    def tokenable(self, corporation, **args):
        return False

    def is_city(self):
        return False

    def is_edge(self):
        return False

    def is_junction(self):
        return False

    def is_label(self):
        return False

    def is_future_label(self):
        return False

    def is_path(self):
        return False

    def is_town(self):
        return False

    def is_halt(self):
        return False

    def is_upgrade(self):
        return False

    def is_offboard(self):
        return False

    def is_border(self):
        return False

    def is_icon(self):
        return False

    def blocks_lay(self):
        return False

    def is_stub(self):
        return False

    def is_frame(self):
        return False

    def is_stripes(self):
        return False

    def is_partition(self):
        return False

    def is_pass(self):
        return False

    def visit_cost(self):
        return 0

    def inspect(self):
        return f"<{self.__class__.__name__}: hex: {self.hex.name if self.hex else None}>"

    def __str__(self):
        return self.inspect()

    def __repr__(self):
        return self.__str__()


class Node(BasePart):
    def __init__(self, lanes=None, paths=None, exits=None, **kwargs):
        super().__init__(**kwargs)
        self._paths = None
        self._exits = None
        self.lanes = None

    def clear(self):
        self._paths = None
        self._exits = None

    def solo(self):
        return len(self.tile.nodes) == 1

    @property
    def paths(self):
        if self._paths is None:
            self._paths = [p for p in self.tile.paths if self in p.nodes]
        return self._paths

    @property
    def exits(self):
        if self._exits is None:
            self._exits = [exit for path in self.paths for exit in path.exits]
        return self._exits

    def rect(self):
        return False

    def walk(
        self,
        visited=None,
        corporation=None,
        visited_paths=None,
        skip_paths=None,
        counter=None,
        skip_track=None,
        converging_path=True,
        all_paths=False,
        debug=False,
    ):
        # if debug:
        #    set_trace()
        if visited is None:
            visited = {}
        if visited_paths is None:
            visited_paths = {}
        if skip_paths is None:
            skip_paths = set()
        if counter is None:
            counter = {}

        # set_trace()
        if self in visited:
            return

        visited[self] = True
        sub_visited, sub_visited_paths, sub_counter = (
            (
                copy_dict(visited),
                copy_dict(visited_paths),
                copy_dict(counter),
            )
            if all_paths
            else (visited, visited_paths, counter)
        )

        for node_path in self.paths:
            if node_path.track == skip_track or node_path in skip_paths or node_path.ignore:
                continue

            for path, vp, ct, converging in node_path.walk(
                visited=sub_visited_paths,
                skip_paths=skip_paths,
                skip_track=skip_track,
                counter=sub_counter,
                converging=converging_path,
                debug=debug,
            ):
                yield path, vp, sub_visited

                # if debug:
                #    set_trace()

                # set_trace()
                if not path.terminal:
                    for next_node in path.nodes:
                        if next_node == self or (corporation and next_node.blocks(corporation)):
                            continue

                        yield from next_node.walk(
                            visited=sub_visited,
                            counter=sub_counter,
                            corporation=corporation,
                            visited_paths=sub_visited_paths,
                            skip_track=skip_track,
                            skip_paths=skip_paths,
                            converging_path=converging_path or converging,
                            all_paths=all_paths,
                            debug=debug,
                        )

        if converging_path:
            del visited[self]

    def __str__(self):
        return super().__str__() + f", paths: {self.paths}, exits: {self.exits}, lanes: {self.lanes}"

    def __repr__(self):
        return self.__str__()


class Path(BasePart):
    LANES = [[1, 0], [1, 0]]
    MATCHES_BROAD = {"broad", "dual"}
    MATCHES_NARROW = {"narrow", "dual"}
    LANE_INDEX = 1
    LANE_WIDTH = 0

    @staticmethod
    def decode_lane_spec(x_lane):
        if x_lane:
            return [int(x_lane), int((float(x_lane) - int(x_lane)) * 10)]
        else:
            return [1, 0]

    @staticmethod
    def make_lanes(
        a,
        b,
        terminal=None,
        lanes=None,
        a_lane=None,
        b_lane=None,
        track="broad",
        ignore=None,
        ignore_gauge_walk=None,
        ignore_gauge_compare=None,
    ):
        paths = []
        if lanes:
            for index in range(lanes):
                a_lanes = [lanes, index]
                b_lanes = [lanes, lanes - index - 1] if a.is_edge() and b.is_edge() else a_lanes
                paths.append(
                    Path(
                        a,
                        b,
                        terminal=terminal,
                        lanes=[a_lanes, b_lanes],
                        track=track,
                        ignore=ignore,
                        ignore_gauge_walk=ignore_gauge_walk,
                        ignore_gauge_compare=ignore_gauge_compare,
                    )
                )
        else:
            path = Path(
                a,
                b,
                terminal=terminal,
                lanes=[Path.decode_lane_spec(a_lane), Path.decode_lane_spec(b_lane)],
                track=track,
                ignore=ignore,
                ignore_gauge_walk=ignore_gauge_walk,
                ignore_gauge_compare=ignore_gauge_compare,
            )
            paths.append(path)
        return paths

    def __init__(
        self,
        a,
        b,
        terminal=None,
        lanes=None,
        track="broad",
        ignore=None,
        ignore_gauge_walk=None,
        ignore_gauge_compare=None,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self._terminal = terminal
        self._ends = []
        self._exits = None
        self.lanes = lanes if lanes else self.LANES
        self.edges = []
        self.stops = []
        self.nodes = []
        self.exit_lanes = {}
        self.track = track
        self._ignore = ignore
        self.ignore_gauge_walk = ignore_gauge_walk
        self.ignore_gauge_compare = ignore_gauge_compare
        self.offboard = None
        self.city = None
        self.junction = None
        self.town = None

        self.separate_parts()

    def separate_parts(self):
        for part in [self.a, self.b]:
            if isinstance(part, Edge):
                self.edges.append(part)
                self.exit_lanes[part.num] = self.lanes[0 if part == self.a else 1]
            elif isinstance(part, Offboard):
                self.offboard = part
                self.stops.append(part)
                self.nodes.append(part)
            elif isinstance(part, City):
                self.city = part
                self.stops.append(part)
                self.nodes.append(part)
            elif isinstance(part, Junction):
                self.junction = part
            elif isinstance(part, Town):
                self.town = part
                self.stops.append(part)
                self.nodes.append(part)
            # Set lanes for the part if applicable
            if hasattr(part, "lanes"):
                part.lanes = self.lanes[0 if part == self.a else 1]

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash((self.id, self.a, self.b))

    def __le__(self, other):
        other_ends = other.ends()
        return all(any(t <= o for o in other_ends) for t in self.ends()) and (
            self.ignore_gauge_compare or self.tracks_match(other)
        )

    def tracks_match(self, other_path, dual_ok=False):
        other_track = other_path.track
        if self.track == "broad":
            return other_track in self.MATCHES_BROAD
        elif self.track == "narrow":
            return other_track in self.MATCHES_NARROW
        elif self.track == "dual":
            return dual_ok or other_track == "dual"
        return False

    def ends(self):
        if not self._ends:
            self._ends = []
            for part in [self.a, self.b]:
                if part.is_junction():
                    for path in part.paths:
                        if path != self:
                            self._ends.extend([p for p in [path.a, path.b] if not p.is_junction()])
                else:
                    self._ends.append(part)
        return self._ends

    def connects_to(self, other, corporation):
        for part in [self.a, self.b]:
            if part.is_edge():
                edge = part.num
                neighbor = self.hex.neighbors.get(edge)
                np_edge = self.hex.invert(edge)
                if neighbor == other.hex and any(e.num == np_edge for e in other.edges):
                    return True
            elif part.is_junction():
                if other.junction == part:
                    return True
            elif part in other.nodes and not part.blocks(corporation):
                return True
        return False

    def lane_match(self, lanes0, lanes1):
        if not lanes0 or not lanes1:
            return False
        if lanes1[self.LANE_WIDTH] == lanes0[self.LANE_WIDTH]:
            return lanes1[self.LANE_INDEX] == self.lane_invert(lanes0)[self.LANE_INDEX]
        else:
            return self.lane_match_different_sizes(lanes0, lanes1)

    def lane_invert(self, lane):
        return [
            lane[self.LANE_WIDTH],
            lane[self.LANE_WIDTH] - lane[self.LANE_INDEX] - 1,
        ]

    def lane_match_different_sizes(self, lanes0, lanes1):
        lanes_a, lanes_b = sorted([lanes0, lanes1], key=lambda x: x[self.LANE_WIDTH])
        larger_width = lanes_b[self.LANE_WIDTH]
        delta = (larger_width - lanes_a[self.LANE_WIDTH]) // 2
        new_index = lanes_a[self.LANE_INDEX] + delta
        return [larger_width, new_index][self.LANE_INDEX] == self.lane_invert(lanes_b)[self.LANE_INDEX]

    def is_path(self):
        return True

    @property
    def node(self):
        return bool(self.nodes)

    @property
    def ignore(self):
        return bool(self._ignore)

    @property
    def terminal(self):
        return bool(self._terminal)

    @property
    def single(self):
        return self.lanes[0][self.LANE_WIDTH] == 1 and self.lanes[1][self.LANE_WIDTH] == 1

    @property
    def exits(self):
        if not self._exits:
            self._exits = [edge.num for edge in self.edges]
        return self._exits

    def rotate(self, ticks):
        rotated_path = Path(
            self.a.rotate(ticks),
            self.b.rotate(ticks),
            terminal=self.terminal,
            lanes=self.lanes,
            track=self.track,
            ignore=self.ignore,
            ignore_gauge_walk=self.ignore_gauge_walk,
            ignore_gauge_compare=self.ignore_gauge_compare,
        )
        rotated_path.index = self.index
        rotated_path.tile = self.tile
        return rotated_path

    def __str__(self):
        name = self.__class__.__name__
        if self.single:
            return f"<{name}: hex: {self.hex.name if self.hex else None}, exit: {self.exits}, track: {self.track}>"
        else:
            return f"<{name}: hex: {self.hex.name if self.hex else None}, exit: {self.exits}, lanes: {self.lanes[0]} {self.lanes[1]}>"

    def __repr__(self):
        return self.__str__()

    def walk(
        self,
        skip=None,
        jskip=None,
        visited=None,
        skip_paths=None,
        counter=None,
        skip_track=None,
        converging=True,
        debug=False,
    ):
        if visited is None:
            visited = {}
        if skip_paths is None:
            skip_paths = set()
        if counter is None:
            counter = {}

        if self in visited or self in skip_paths:
            return
        if self.junction and counter.get(self.junction) > 1:
            return
        if any(counter.get(edge.id) for edge in self.edges):
            return
        if self.track == skip_track:
            return
        if self.junction and self.terminal:
            return

        visited[self] = True
        if self.junction:
            counter[self.junction] += 1

        yield self, visited, counter, converging

        if debug and self.hex.name == "F12":
            set_trace()
        if self.junction and self.junction != jskip:
            for jp in self.junction.paths:
                yield from jp.walk(
                    jskip=self.junction,
                    visited=visited,
                    skip_paths=skip_paths,
                    counter=counter,
                    converging=converging,
                    debug=debug,
                )

        for edge in self.edges:
            edge_id = edge.id
            edge_num = edge.num
            if edge_num == skip:
                continue
            neighbor = self.hex.neighbors.get(edge_num)
            if not neighbor:
                continue

            counter[edge_id] = counter.get(edge_id, 0) + 1
            np_edge = self.hex.invert(edge_num)

            for np in neighbor.paths().get(np_edge, []):
                if not self.lane_match(self.exit_lanes[edge_num], np.exit_lanes[np_edge]):
                    continue
                if not self.ignore_gauge_walk and not self.tracks_match(np, dual_ok=True):
                    continue

                yield from np.walk(
                    skip=np_edge,
                    visited=visited,
                    skip_paths=skip_paths,
                    counter=counter,
                    skip_track=skip_track,
                    converging=converging or self.tile.converging_exit(edge_num),
                    debug=debug,
                )

            counter[edge_id] = counter.get(edge_id, 0) - 1

        if converging:
            del visited[self]
        if self.is_junction():
            counter[self.junction] -= 1


class Edge(BasePart):
    def __init__(self, num, lanes=None):
        super().__init__()
        self.num = int(num)
        self.lanes = lanes if lanes else [None, None]

    @property
    def id(self):
        return f"{self.hex.id if self.hex else None}_{self.num}_{self.lanes[1]}"

    def __hash__(self):
        return super().__hash__() ^ hash(self.num)

    def __le__(self, other):
        return isinstance(other, Edge) and (self.num == other.num)

    def __lt__(self, other):
        if self.is_edge() and other.is_edge():
            return self.num < other.num
        elif self.is_edge():
            return True
        return False

    def __eq__(self, other):
        if self.is_edge() and other.is_edge():
            return self.num == other.num
        elif self.is_edge() or other.is_edge():
            return False
        return True

    def __gt__(self, other):
        if self.is_edge() and other.is_edge():
            return self.num > other.num
        elif other.is_edge():
            return True
        return False

    def is_edge(self):
        return True

    def rotate(self, ticks):
        edge = Edge((self.num + ticks) % 6)
        edge.index = self.index
        edge.tile = self.tile
        edge.lanes = self.lanes
        return edge


class Frame(BasePart):
    def __init__(self, color, color2=None):
        super().__init__()
        self.color = color
        self.color2 = color2

    def is_frame(self):
        return True


class FutureLabel(BasePart):
    def __init__(self, label=None, color=None):
        super().__init__()
        self.label = label
        self.color = color
        self.sticker = None

    def is_future_label(self):
        return True


class Icon(BasePart, Ownable):
    def __init__(
        self,
        image,
        name=None,
        sticky=True,
        blocks_lay=None,
        preprinted=True,
        large=False,
        owner=None,
        loc=None,
    ):
        self.image = f"/icons/{image}.svg" if not image.startswith("/icons") else image
        self.name = name or image.split("/")[-1]
        self.sticky = bool(sticky)
        self.preprinted = bool(preprinted)
        self.blocks_lay = bool(blocks_lay) if blocks_lay is not None else None
        self.large = bool(large)
        self.owner = owner
        self.loc = loc

    def blocks_lay(self):
        return self.blocks_lay

    def is_icon(self):
        return True


class Junction(BasePart):
    def __init__(self, lanes=[], **kwargs):
        super().__init__(**kwargs)
        self.lanes = lanes

    def is_junction(self):
        return True

    def clear(self):
        del self._paths
        del self._exits

    def paths(self):
        if not hasattr(self, "_paths"):
            self._paths = [p for p in self.tile.paths if p.junction == self]
        return self._paths

    def exits(self):
        if not hasattr(self, "_exits"):
            self._exits = [exit for path in self.paths for exit in path.exits]
        return self._exits


class Label(BasePart):
    def __init__(self, label=None, **kwargs):
        super().__init__(**kwargs)
        self._label = label

    def __str__(self):
        return str(self._label)

    def __eq__(self, other):
        return isinstance(other, Label) and self._label == other._label

    def is_label(self):
        return True


class Partition(BasePart):
    SIGN = {
        "-": -1,
        None: 0,
        "+": 1,
    }

    def __init__(self, a, b, type, restrict):
        a, b = sorted([a, b])
        self.a = int(a[0])
        self.a_sign = self.SIGN.get(a[1], 0)
        self.b = int(b[0])
        self.b_sign = self.SIGN.get(b[1], 0)
        self.type = type
        self.restrict = restrict
        self.blockers = []
        self.inner = [] if restrict == "outer" else list(range(self.a, self.b))
        self.outer = [] if restrict == "inner" else list(set(range(6)) - set(range(self.a, self.b)))

    def add_blocker(self, private_company):
        self.blockers.append(private_company)

    def is_partition(self):
        return True


class Stripes(BasePart):
    def __init__(self, color, **kwargs):
        super().__init__(**kwargs)
        self.color = color

    def is_stripes(self):
        return True


class Stub(BasePart):
    def __init__(self, edge, track="broad", **kwargs):
        super().__init__(**kwargs)
        self.edge = edge
        self.track = track

    def is_stub(self):
        return True

    def __str__(self):
        return f"<{self.__class__.__name__} edge={self.edge}>"


class Upgrade(BasePart):
    def __init__(self, cost, terrains=[], size=0, loc=None, **kwargs):
        super().__init__(**kwargs)
        self.cost = int(cost)
        self.terrains = terrains
        self.size = size
        self.loc = loc

    def is_upgrade(self):
        return True

    def is_mountain(self):
        if not hasattr(self, "_mountain"):
            self._mountain = "mountain" in self.terrains
        return self._mountain

    def is_water(self):
        if not hasattr(self, "_water"):
            self._water = "water" in self.terrains
        return self._water


class RevenueCenter(Node):
    PHASES = ["yellow", "green", "brown", "gray", "diesel"]

    def __init__(self, revenue, **opts):
        super().__init__(**opts)
        self.revenue = self.parse_revenue(revenue, opts.get("format"))
        self.groups = opts.get("groups", "").split("|")
        self.hide = opts.get("hide")
        self.visit_cost = int(opts.get("visit_cost", 1))
        self.loc = opts.get("loc")
        self.route = opts.get("route", "mandatory")
        self.revenue_to_render = None

    def parse_revenue(self, revenue, format=None):
        if "|" in revenue:
            parts = revenue.split("|")
            revenue_dict = {color: int(r) for color, r in (part.split("_") for part in parts)}
            self.revenue_to_render = {phase: (format % rev if format else rev) for phase, rev in revenue_dict.items()}
            return revenue_dict
        else:
            revenue_val = int(revenue)
            self.revenue_to_render = format % revenue_val if format else revenue_val
            return {phase: revenue_val for phase in self.PHASES}

    def max_revenue(self):
        return max(self.revenue.values())

    def route_revenue(self, phase, train):
        return self.revenue_multiplier(train) * self.route_base_revenue(phase, train)

    def route_base_revenue(self, phase, train):
        if train.name.upper() == "D" and "diesel" in self.revenue:
            return self.revenue["diesel"]

        for color in reversed(phase.tiles):
            if color in self.revenue:
                return self.revenue[color]

        return 0

    def revenue_multiplier(self, train):
        distance = train.distance
        base_multiplier = train.multiplier or 1

        if isinstance(distance, (int, float)):
            return base_multiplier

        for h in distance:
            if self.type in h["nodes"]:
                return h.get("multiplier", base_multiplier)

        return base_multiplier

    def uniq_revenues(self):
        return list(set(self.revenue.values()))

    def __str__(self):
        return super().__str__() + f", revenue: {self.revenue}, groups: {self.groups}, loc: {self.loc}"

    def __repr__(self):
        return self.__str__()


class Border(BasePart):
    def __init__(self, edge, type=None, cost=None, color=None):
        self.edge = int(edge)
        self.type = type and type.lower()  # Converting to lowercase for symbol-like behavior
        self.cost = int(cost) if cost is not None else None
        self.color = color and color.lower()  # Converting to lowercase for symbol-like behavior

    def is_border(self):
        return True


class City(RevenueCenter):
    def __init__(self, revenue, **opts):
        super().__init__(revenue, **opts)
        self.slots = int(opts.get("slots", 1))
        self.tokens = [None] * int(self.slots)
        self.extra_tokens = []
        self.reservations = []
        self.boom = opts.get("boom")
        self.slot_icons = {}

    def slots(self, all=False):
        return len(self.tokens) + (len(self.extra_tokens) if all else 0)

    def normal_slots(self):
        return self.slots

    def remove_tokens(self):
        self.tokens = [None] * len(self.tokens)
        self.extra_tokens = []

    def blocks(self, corporation):
        if not corporation:
            return False
        if self.tokened_by(corporation):
            return False
        if None in self.tokens:
            return False
        if any(t is not None and t.type == "neutral" for t in self.tokens):
            return False
        return True

    def tokened(self):
        return any(t is not None for t in self.tokens)

    def tokened_by(self, corporation):
        return any(t is not None and t.corporation == corporation for t in self.tokens + self.extra_tokens)

    def find_reservation(self, corporation):
        for index, reservation in enumerate(self.reservations):
            if reservation and (reservation == corporation or reservation.owner == corporation):
                return index
        return None

    def reserved_by(self, corporation):
        return self.find_reservation(corporation) is not None

    def add_reservation(self, entity, slot=None):
        if slot is not None:
            self.reservations.insert(slot, entity)
        else:
            self.reservations.append(entity)

    def remove_reservation(self, entity):
        try:
            index = self.reservations.index(entity)
            self.reservations[index] = None
        except ValueError:
            pass

    def remove_all_reservations(self):
        self.reservations.clear()

    def is_city(self):
        return True

    def tokenable(
        self,
        corporation,
        free=False,
        tokens=None,
        cheater=False,
        extra_slot=False,
        spender=None,
        same_hex_allowed=False,
    ):
        # set_trace()
        if tokens is None:
            tokens = corporation.tokens_by_type
        self.error = "generic"
        if not extra_slot and not tokens:
            self.error = "no_tokens"
            return False

        return any(
            self._is_tokenable(t, corporation, free, cheater, extra_slot, spender, same_hex_allowed) for t in tokens
        )

    def _is_tokenable(self, token, corporation, free, cheater, extra_slot, spender, same_hex_allowed):
        # set_trace()
        if not extra_slot and self.get_slot(token.corporation, cheater) is None:
            self.error = "no_slots"
            return False
        if not free and token.price > (spender or corporation).cash:
            self.error = "no_money"
            return False
        if not same_hex_allowed and any(c.tokened_by(token.corporation) for c in self.tile.cities):
            self.error = "existing_token"
            return False
        if self.reserved_by(corporation):
            return True
        if self.tile.token_blocked_by_reservation(corporation) and not cheater:
            self.error = "blocked_reservation"
            return False
        return True

    @property
    def available_slots(self):
        reservations = self.reservations + ([None] * (4 - len(self.reservations)))
        return sum(1 for token, reservation in zip(self.tokens, reservations) if token is None and reservation is None)

    def get_slot(self, corporation, cheater=False):
        # set_trace()
        reservation = self.find_reservation(corporation)
        reservations = self.reservations + ([None] * (4 - len(self.reservations)))
        open_slot = next(
            (
                i
                for i, (token, reservation) in enumerate(zip(self.tokens, reservations))
                if token is None and reservation is None
            ),
            None,
        )
        if cheater:
            if open_slot is not None:
                return open_slot
            return len(self.tokens)

        if reservation is not None:
            return reservation
        return open_slot

    def place_token(
        self,
        corporation,
        token,
        free=False,
        check_tokenable=True,
        cheater=False,
        extra_slot=False,
        spender=None,
        same_hex_allowed=False,
    ):
        # set_trace()
        if check_tokenable and not self.tokenable(
            corporation, free, [token], cheater, extra_slot, spender, same_hex_allowed
        ):
            self._raise_token_error(corporation)
        self.exchange_token(token, cheater, extra_slot)
        if corporation in self.tile.reservations:
            self.tile.reservations.remove(corporation)
        self.remove_reservation(corporation)

    def exchange_token(self, token, cheater=False, extra_slot=False):
        token.place(self, extra=extra_slot, cheater=cheater)
        if extra_slot:
            self.extra_tokens.append(token)
            return
        slot = self.get_slot(token.corporation, cheater)
        self.tokens[slot] = token

    def _raise_token_error(self, corporation):
        error_messages = {
            "no_tokens": "cannot lay token - has no tokens left",
            "existing_token": "cannot lay token - already has a token",
            "blocked_reservation": "cannot lay token - remaining token slots are reserved",
            "no_money": "cannot lay token - cannot afford token",
            "no_slots": "cannot lay token - no token slots available",
        }
        error_msg = error_messages.get(self.error, "cannot lay token")
        raise GameError(f"{corporation.name} {error_msg} on {self.id} {self.tile.hex.id if self.tile.hex else 'N/A'}")

    def reset(self):
        self.remove_tokens()
        self.tokens = [None] * int(self.slots)

    def delete_token(self, token, remove_slot=False):
        if remove_slot:
            position = self.tokens.index(token) if token in self.tokens else None
            if position is not None:
                del self.tokens[position]
                del self.reservations[position]
                self.reservations.append(None)
        else:
            self.tokens = [None if t == token else t for t in self.tokens]

    def __str__(self):
        return super().__str__() + f", slots: {self.slots}, tokens: {self.tokens}, reservations: {self.reservations}"

    def __repr__(self):
        return self.__str__()


class Town(RevenueCenter):
    def __init__(self, revenue, to_city=None, boom=None, style=None, **kwargs):
        super().__init__(revenue)
        self.to_city = to_city
        self.boom = boom
        self.style = style

    def __le__(self, other):
        if self.to_city and other.city:
            return True
        return super().__le__(other)

    def is_town(self):
        return True

    def rect(self):
        if self.style:
            return self.style == "rect"
        return bool(self.paths and len(self.paths) < 3)

    def hidden(self):
        return self.style == "hidden"


class Halt(Town):
    def __init__(self, symbol, revenue="0", route="optional", **opts):
        super().__init__(revenue, **opts)
        self.symbol = symbol
        self.route = route if route is None else route.to_sym()

    def __le__(self, other):
        if other.is_town():
            return True
        return super().__le__(other)

    def is_halt(self):
        return True


class Offboard(RevenueCenter):
    def blocks(self, corporation):
        return True

    def is_offboard(self):
        return True


class Pass(City):
    def __init__(self, revenue, **opts):
        super().__init__(revenue, **opts)
        self.color = opts.get("color", "gray").lower()
        self.size = int(opts.get("size", 1))
        self.route = opts.get("route", "never").lower()

    def is_pass(self):
        return True


class Token:
    def __init__(self, corporation, price=0, logo=None, simple_logo=None, type="normal"):
        self.corporation = corporation
        self.price = price
        self.logo = logo or corporation.logo
        self.simple_logo = simple_logo or (corporation.simple_logo if corporation else self.logo)
        self.used = False
        self.extra = None  # Is this in an extra slot? (bull token)
        self.cheater = None
        self.type = type
        self.city = None
        self.hex = None
        self.status = None
        self.location_type = None

    def __str__(self):
        location_string = f", hex: {self.hex}, city: {self.city}" if self.used else ""
        return f"Token - corporation: {self.corporation}, used: {self.used}, type: {self.type}{location_string}"

    def __repr__(self):
        return self.__str__()

    def destroy(self):
        if self.corporation:
            self.corporation.tokens.remove(self)
        self.remove()

    def remove(self):
        if self.location_type == "city":
            if self.city:
                self.city.tokens = [None if t == self else t for t in self.city.tokens]
                if self.city.extra_tokens and self in self.city.extra_tokens:
                    self.city.extra_tokens.remove(self)
        elif self.location_type == "hex":
            if self.hex:
                self.hex.remove_token(self)

        self.city = None
        self.hex = None
        self.used = False
        self.extra = False
        self.cheater = False
        self.location_type = None

    def swap(self, other_token, check_tokenable=True, free=True):
        city = self.city
        hex = self.hex
        extra = self.extra
        location_type = self.location_type
        self.remove()
        corporation = other_token.corporation

        if (
            not extra
            and check_tokenable
            and location_type == "city"
            and not city.tokenable(corporation, free=free, tokens=[other_token])
        ):
            return

        if location_type == "city":
            city.place_token(
                corporation,
                other_token,
                free=free,
                check_tokenable=check_tokenable,
                extra_slot=extra,
            )
        elif location_type == "hex":
            hex.place_token(other_token)

    def move(self, new_location):
        self.remove()

        if isinstance(new_location, City):
            new_location.place_token(self.corporation, self, free=True)
        elif isinstance(new_location, Hex):
            new_location.place_token(self)

    def place(self, location, extra=None, cheater=None):
        self.used = True

        if isinstance(location, City):
            self.location_type = "city"
            self.city = location
            self.hex = location.hex
        elif isinstance(location, Hex):
            self.location_type = "hex"
            self.hex = location

        self.extra = extra
        self.cheater = cheater


class Graph:
    def __init__(self, game, **opts):
        self.game = game
        self._connected_hexes = {}
        self._connected_nodes = {}
        self._connected_paths = {}
        self._connected_hexes_by_token = {}
        self._connected_paths_by_token = {}
        self._connected_nodes_by_token = {}
        self._reachable_hexes = {}
        self._tokenable_cities = {}
        self.routes = {}
        self.tokens = {}
        self.cheater_tokens = {}
        self.home_as_token = opts.get("home_as_token", False)
        self.no_blocking = opts.get("no_blocking", False)
        self.skip_track = opts.get("skip_track")
        self.check_tokens = opts.get("check_tokens")
        self.check_regions = opts.get("check_regions")

    def clear(self):
        self._connected_hexes.clear()
        self._connected_nodes.clear()
        self._connected_paths.clear()
        self._connected_hexes_by_token.clear()
        self._connected_nodes_by_token.clear()
        self._connected_paths_by_token.clear()
        self._reachable_hexes.clear()
        self._tokenable_cities.clear()
        self.tokens.clear()
        self.cheater_tokens.clear()
        to_delete = [key for key, route in self.routes.items() if not route.get("route_train_purchase")]
        for key in to_delete:
            del self.routes[key]

    def clear_graph_for(self, corporation):
        self.clear()
        self.routes.pop(corporation, None)

    def clear_graph_for_all(self):
        self.clear()
        self.routes.clear()

    def route_info(self, corporation):
        if corporation not in self.routes:
            list(self.compute(corporation, routes_only=True))
        return self.routes.get(corporation)

    def can_token(self, corporation, cheater=False, same_hex_allowed=False, tokens=None):
        if tokens is None:
            tokens = corporation.tokens_by_type
        tokeners = self.cheater_tokens if cheater else self.tokens
        if corporation in tokeners:
            return tokeners[corporation]

        for node in self.compute(corporation):
            if node.tokenable(
                corporation,
                free=True,
                cheater=cheater,
                tokens=tokens,
                same_hex_allowed=same_hex_allowed,
            ):
                tokeners[corporation] = True
                break
        else:
            tokeners[corporation] = False

        return tokeners[corporation]

    def tokenable_cities(self, corporation):
        if corporation in self._tokenable_cities:
            return self._tokenable_cities[corporation]

        cities = []
        for node in self.compute(corporation):
            if node.tokenable(corporation, free=True):
                cities.append(node)

        if cities:
            self._tokenable_cities[corporation] = cities
        return cities

    def connected_hexes(self, corporation):
        if corporation not in self._connected_hexes:
            list(self.compute(corporation))
        return self._connected_hexes[corporation]

    def connected_nodes(self, corporation):
        if corporation not in self._connected_nodes:
            list(self.compute(corporation))
        return self._connected_nodes[corporation]

    def connected_paths(self, corporation):
        if corporation not in self._connected_paths:
            list(self.compute(corporation))
        return self._connected_paths[corporation]

    def connected_hexes_by_token(self, corporation, token):
        if token not in self._connected_hexes_by_token[corporation]:
            self.compute_by_token(corporation)
        return self._connected_hexes_by_token[corporation][token]

    def connected_nodes_by_token(self, corporation, token):
        if token not in self._connected_nodes_by_token[corporation]:
            self.compute_by_token(corporation)
        return self._connected_nodes_by_token[corporation][token]

    def connected_paths_by_token(self, corporation, token):
        if token not in self._connected_paths_by_token[corporation]:
            self.compute_by_token(corporation)
        return self._connected_paths_by_token[corporation][token]

    def reachable_hexes(self, corporation):
        if corporation not in self._reachable_hexes:
            list(self.compute(corporation))
        return self._reachable_hexes[corporation]

    def compute_by_token(self, corporation):
        list(self.compute(corporation))
        for hex in self.game.hexes():
            for city in hex.tile.cities:
                if self.game.city_tokened_by(city, corporation) and not (
                    self.check_tokens and self.game.skip_token(self, corporation, city)
                ):
                    list(self.compute(corporation, one_token=city))

    def home_hexes(self, corporation):
        home_hexes = {}
        coordinates = corporation.coordinates
        if not isinstance(coordinates, list):
            coordinates = [coordinates]
        hexes = [self.game.hex_by_id(h) for h in coordinates]
        for hex in hexes:
            for edge, _ in hex.neighbors.items():
                home_hexes.setdefault(hex, {})[edge] = True
        return home_hexes

    def home_hex_nodes(self, corporation):
        nodes = {}
        coordinates = corporation.coordinates
        if not isinstance(coordinates, list):
            coordinates = [coordinates]
        hexes = [self.game.hex_by_id(h) for h in coordinates]
        for hex in hexes:
            if corporation.city is not None:
                # If corporation.city is a single value, this makes it a list
                city_indices = [corporation.city] if not isinstance(corporation.city, list) else corporation.city
                for c_idx in city_indices:
                    if 0 <= c_idx < len(hex.tile.cities):
                        city = hex.tile.cities[c_idx]
                        nodes[city] = True
            else:
                # Include all city and town parts if corporation.city is None
                for ct in hex.tile.city_towns:
                    nodes[ct] = True
        return nodes

    def compute(self, corporation, routes_only=False, one_token=None):
        hexes = defaultdict(dict)
        nodes = {}
        paths = {}

        for hex in self.game.hexes:
            for city in hex.tile.cities:
                if one_token and city != one_token:
                    continue

                if not self.game.city_tokened_by(city, corporation):
                    continue
                if self.check_tokens and self.game.skip_token(self, corporation, city):
                    continue

                for e in hex.neighbors:
                    hexes[hex][e] = True
                nodes[city] = True

        if self.home_as_token and corporation.coordinates:
            hexes.update(self.home_hexes(corporation))
            nodes.update(self.home_hex_nodes(corporation))

        tokens = nodes.copy()

        for ability_type in ["token", "teleport"]:
            for ability in self.game.abilities(corporation, ability_type):
                owner = ability.owner
                if owner != corporation:
                    continue
                if ability_type == "token" and not ability.teleport_price:
                    continue

                for hex_id in ability.hexes:
                    hex = self.game.hex_by_id(hex_id)
                    for node in hex.tile.cities:
                        nodes[node] = True
                        if ability_type == "teleport" and ability.used:
                            yield node

        routes = self.routes.get(corporation, {})
        walk_corporation = None if self.no_blocking else corporation
        skip_paths = (
            self.game.graph_border_paths(corporation) if self.check_regions else self.game.graph_skip_paths(corporation)
        )
        if skip_paths is None:
            skip_paths = set()

        for node in tokens:
            if routes.get("route_train_purchase") and routes_only:
                return None

            visited = {k: v for k, v in tokens.items() if k != node}
            local_nodes = {}
            visited_paths = {}
            counter = {}

            for path, _, _ in node.walk(
                visited=visited,
                corporation=walk_corporation,
                skip_track=self.skip_track,
                visited_paths=visited_paths,
                skip_paths=skip_paths,
                converging_path=False,
                counter=counter,
            ):
                if path in paths:
                    continue

                paths[path] = True

                for p_node in path.nodes:
                    nodes[p_node] = True
                    local_nodes[p_node] = True
                    yield p_node

                hex = path.hex

                for edge in path.exits:
                    hexes[hex][edge] = True
                    if not self.check_regions or not self.game.region_border(hex, edge):
                        hexes[hex.neighbors[edge]][hex.invert(edge)] = True

            mandatory_nodes = sum(1 for p_node in local_nodes if p_node.route == "mandatory")
            optional_nodes = sum(1 for p_node in local_nodes if p_node.route == "optional")

            if mandatory_nodes > 1:
                routes["route_available"] = True
                routes["route_train_purchase"] = True
            elif mandatory_nodes == 1 and optional_nodes > 0:
                routes["route_available"] = True

        if one_token:
            self._connected_hexes_by_token[corporation][one_token] = hexes
            self._connected_nodes_by_token[corporation][one_token] = nodes
            self._connected_paths_by_token[corporation][one_token] = paths
        else:
            self.routes[corporation] = routes
            self._connected_hexes[corporation] = hexes
            self._connected_nodes[corporation] = nodes
            self._connected_paths[corporation] = paths
            self._reachable_hexes[corporation] = {path.hex: True for path in paths}


class DistanceGraph:
    def __init__(self, game, separate_node_types=False):
        self.game = game
        self.node_distances = {}
        self.path_distances = {}
        self.hex_distances = {}
        self.separate_node_types = separate_node_types

    def clear(self):
        self.node_distances.clear()
        self.path_distances.clear()
        self.hex_distances.clear()

    def get_token_cities(self, corporation):
        tokens = []
        for hex in self.game.hexes:
            for city in hex.tile.cities:
                if city.tokened_by(corporation):
                    tokens.append(city)
        return tokens

    def smaller_or_equal_distance(self, a, b):
        return all(a.get(k, float("inf")) <= v for k, v in b.items())

    def merge_distance(self, dict, key, b):
        a = dict.get(key, {})
        if a:
            for k, v in b.items():
                a[k] = min(a.get(k, float("inf")), v)
        else:
            dict[key] = b.copy()
        return dict

    def node_walk(
        self,
        node,
        distance,
        node_distances,
        path_distances,
        a_distances,
        b_distances,
        corporation,
        counter=None,
    ):
        if counter is None:
            counter = {}

        self.merge_distance(node_distances, node, distance)
        if corporation and node.blocks(corporation):
            return

        count = 1 if node.visit_cost > 0 else 0
        distance_key = "node"
        if self.separate_node_types:
            if node.is_city():
                distance_key = "city"
            elif node.is_town() and not node.is_halt():
                distance_key = "town"
        distance[distance_key] += count

        for node_path in node.paths:
            for path, _, ct in node_path.walk(counter=counter):
                self.merge_distance(path_distances, path, distance)
                yield path, distance

                if not path.terminal:
                    for next_node in path.nodes:
                        if next_node == node:
                            continue
                        a_or_b = "a" if path.a == next_node else "b"
                        next_distance = {**distance}  # Copy to avoid mutating the original distance

                        if a_or_b == "a":
                            if not self.smaller_or_equal_distance(a_distances.get(path, {}), next_distance):
                                self.merge_distance(a_distances, path, next_distance)
                            else:
                                continue
                        else:
                            if not self.smaller_or_equal_distance(b_distances.get(path, {}), next_distance):
                                self.merge_distance(b_distances, path, next_distance)
                            else:
                                continue

                        yield from self.node_walk(
                            next_node,
                            next_distance,
                            node_distances,
                            path_distances,
                            a_distances,
                            b_distances,
                            corporation,
                            counter=ct,
                        )

        distance[distance_key] -= count

    def compute(self, corporation):
        tokens = self.get_token_cities(corporation)
        n_distances, p_distances, h_distances = {}, {}, {}

        for node in tokens:
            start_distance = {"city": 0, "town": 0} if self.separate_node_types else {"node": 0}
            for path, dist in self.node_walk(node, start_distance, n_distances, p_distances, {}, {}, corporation):
                self.merge_distance(h_distances, path.hex, dist)

        self.node_distances[corporation] = n_distances
        self.path_distances[corporation] = p_distances
        self.hex_distances[corporation] = h_distances


class Route:
    def __init__(self, game, phase, train, **opts):
        self.game = game
        self.phase = phase
        self._train = train
        self.routes = opts.get("routes", [])
        self._connection_hexes = opts.get("connection_hexes", None)
        self._hexes = opts.get("hexes", None)
        self._revenue = opts.get("revenue", None)
        self._revenue_str = opts.get("revenue_str", None)
        self._subsidy = opts.get("subsidy", None)
        self.halts = opts.get("halts", None)
        self.abilities = opts.get("abilities", [])
        self._node_signatures = opts.get("nodes", None)
        self.local_length = game.local_length()

        self.node_chains = {}
        self._connection_data = opts.get("connection_data")
        self.last_node = None
        self.last_offboard = []

        self._ordered_paths = None
        self._ordered_hexes = None
        self._distance_str = None
        self._distance = None
        self._hexes = None
        self._paths = None
        self._stops = None
        self._subsidy = None
        self._visited_stops = None
        self._check_connected = None
        self._check_distance = None

        self.bitfield = opts.get("bitfield")  # array of ints used only by auto-routing algorithm

    def clear_cache(self, all=False, only_routes=False):
        if all:
            self._connection_hexes = None
            self._node_signatures = None
        self._revenue = None
        self._revenue_str = None

        if not all and only_routes:
            return

        self._ordered_paths = None
        self._ordered_hexes = None
        self._distance_str = None
        self._distance = None
        self._hexes = None
        self._paths = None
        self._stops = None
        self._subsidy = None
        self._visited_stops = None
        self._check_connected = None
        self._check_distance = None

    def reset(self):
        self.clear_cache(all=True)
        self.halts = None
        self._connection_data = None
        self.last_node = None
        self.last_offboard = []

    def __str__(self):
        return f"<Route> {self.paths}"

    def __repr__(self):
        return self.__str__()

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, new_train):
        self._train = new_train
        self.clear_cache()

    def cycle_halts(self):
        if self.halts is not None:
            self.halts += 1
            max_halts = self.game.max_halts(self)
            if self.halts > max_halts:
                self.halts = 0
            self.clear_cache()

    @property
    def head(self):
        return self.connection_data[0] if self.connection_data else None

    @property
    def tail(self):
        return self.connection_data[-1] if self.connection_data else None

    @property
    def chains(self):
        return [c["chain"] for c in self.connection_data] if self.connection_data else []

    @property
    def node_signatures(self):
        if not self._node_signatures:
            chains = self.chains
            self._node_signatures = list({node.signature for chain in chains for node in chain["nodes"] if node})
        return self._node_signatures

    def next_chain(self, node, chain, other):
        chains = self.select(node, other, chain)
        index = chains.index(chain) if chain in chains else len(chains)
        return chains[index + 1] if index + 1 < len(chains) else None

    def select(self, node, other, keep=None):
        other_paths = self.compute_other_paths()
        for c in self.connection_data:
            if c["chain"] != keep:
                other_paths.extend(c["chain"]["paths"])

        return [c for c in self.get_node_chains(node, other) if not set(c["paths"]).intersection(other_paths)]

    def get_node_chains(self, start_node, end_node):
        skip_track = self.game.skip_route_track_type(self._train)
        key = (start_node, end_node)
        if key not in self.node_chains:
            new_chains = []
            for start_path in start_node.paths:
                for current, visited, _, _ in start_path.walk(skip_track=skip_track):
                    if end_node in current.nodes:
                        paths = list(visited.keys())
                        new_chains.append(
                            {
                                "nodes": [start_node, end_node],
                                "paths": paths,
                                "hexes": [path.hex for path in paths],
                                "id": self.chain_id(paths),
                            }
                        )
            self.node_chains[key] = new_chains
        return self.node_chains[key]

    def segment(self, chain, left=None, right=None):
        nodes = chain["nodes"]
        if left is not None:
            right = next((node for node in nodes if node != left), None)
        else:
            left = next((node for node in nodes if node != right), None)
        return {"left": left, "right": right, "chain": chain}

    def touch_node(self, node):
        if self.connection_data and not self.local_connection():
            if node == self.head["left"]:
                chain = self.next_chain(self.head["right"], self.head["chain"], node)
                if chain:
                    self._connection_data[0] = self.segment(chain, right=self.head["right"])
                else:
                    self._connection_data.pop(0)
            elif node == self.tail["right"]:
                chain = self.next_chain(self.tail["left"], self.tail["chain"], node)
                if chain:
                    self._connection_data[-1] = self.segment(chain, left=self.tail["left"])
                else:
                    self._connection_data.pop()
            elif node == self.head["right"]:
                self._connection_data.pop(0)
            elif node == self.tail["left"]:
                self._connection_data.pop()
            else:
                chain = self.select(self.head["left"], node)[0]
                if chain:
                    self._connection_data.insert(0, self.segment(chain, right=self.head["left"]))
                else:
                    chain = self.select(self.tail["right"], node)[0]
                    if chain:
                        self._connection_data.append(self.segment(chain, left=self.tail["right"]))

            if self._train.local and len(self._connection_data) == self.local_length:
                self._connection_data.pop()
        elif self.last_node == node:
            self.last_node = None
            self._connection_data.clear()
        elif self.last_node:
            self._connection_data.clear()
            chain = self.select(self.last_node, node)[0]
            if chain:
                a, b = chain["nodes"]
                if self.last_node == a:
                    a, b = b, a
                self._connection_data.append({"left": a, "right": b, "chain": chain})
        else:
            self.last_node = node
            if self._train.local and not self.connection_data:
                self.add_single_node_connection(node)

        self.halts = None
        for route in self.routes:
            route.clear_cache(all=True)

    def disambiguate_node(self, nodes):
        onodes = [node for node in nodes if node.is_offboard()]

        # Determine the relevant nodes based on the current route's state
        list = [self.last_node] if self.connection_data == [] else [self.head["left"], self.tail["right"]]
        list = [node for node in list if node]  # Remove None values

        if not list:
            return

        # Find a matching offboard node already connected to the route
        match = next((node for node in onodes if node in list), None)
        if match:
            self.touch_node(match)
            self.last_offboard = [match]
            return

        # Select a candidate node that connects to the current route, excluding the most recently used one
        candidates = [node for node in onodes if any(select_node in list for select_node in self.select(node))]
        candidates = [candidate for candidate in candidates if candidate not in self.last_offboard]

        if len(candidates) > 1:
            self.touch_node(candidates[0])
            self.last_offboard = []
        elif len(candidates) == 1:
            self.touch_node(candidates[0])
            self.last_offboard = []

    @property
    def paths(self):
        if not self._paths:
            self._paths = [path for chain in self.chains for path in chain["paths"]]
        return self._paths

    def paths_for(self, other_paths):
        return set(self.paths) & set(other_paths)

    @property
    def visited_stops(self):
        if not self._visited_stops:
            self._visited_stops = self.game.visited_stops(self)
        return self._visited_stops

    @property
    def stops(self):
        if not self._stops:
            self._stops = self.game.compute_stops(self)
        return self._stops

    @property
    def hexes(self):
        if not self._hexes:
            self._hexes = {node.hex for c in self.connection_data for node in [c["left"], c["right"]] if node}
        return self._hexes

    @property
    def all_hexes(self):
        return set(path.hex for path in self.paths)

    def check_cycles(self):
        if self._train.local:
            return

        cycles = {}
        for c in self.connection_data:
            right = c["right"]
            if right in cycles:
                raise GameError(f"Cannot use {right.hex.name} twice")
            cycles[c["left"]] = True
            cycles[right] = True

    def check_overlap(self):
        self.game.check_overlap(self.routes)

    def check_connected(self):
        if not self._check_connected:
            self._check_connected = self.game.check_connected(self, self.corporation) or True
        return self._check_connected

    @property
    def ordered_paths(self):
        if not self._ordered_paths:
            self._ordered_paths = []
            for c in self.connection_data:
                if c["chain"]["paths"]:
                    paths = c["chain"]["paths"]
                    if c["left"] not in paths[0].nodes:
                        paths = paths[::-1]
                    self._ordered_paths.extend(paths)
        return self._ordered_paths

    @property
    def ordered_hexes(self):
        if not self._ordered_hexes:
            self._ordered_hexes = list(dict.fromkeys(path.hex for path in self.ordered_paths))
        return self._ordered_hexes

    def check_terminals(self):
        if len(self.paths) < 3:
            return
        if any(path.terminal for path in self.ordered_paths[1:-1]):
            raise GameError("Route cannot pass through terminal")

    @property
    def distance_str(self):
        if not self._distance_str:
            self._distance_str = self.game.route_distance_str(self)
        return self._distance_str

    @property
    def distance(self):
        if not self._distance:
            self._distance = self.game.route_distance(self)
        return self._distance

    def check_distance(self, visits):
        if not self._check_distance:
            self._check_distance = self.game.check_distance(self, visits) or True
        return self._check_distance

    def check_other(self):
        self.game.check_other(self)

    def revenue(self, suppress_check_other=False, suppress_route_token_check=False):
        if not self._revenue:
            visited = self.visited_stops
            if self.connection_data and len(visited) < 2 and not self._train.is_local():
                raise GameError("Route must have at least 2 stops")

            token = next(
                (stop for stop in visited if self.game.city_tokened_by(stop, self.corporation)),
                None,
            )
            if not suppress_route_token_check:
                self.game.check_route_token(self, token)
            # set_trace()
            flattened_groups = [group for stop in visited for group in stop.groups if group != ""] or []
            flattened_groups.sort()
            for key, group in itertools.groupby(flattened_groups, key=lambda x: x):
                grouped_list = list(group)
                if len(grouped_list) > 1:
                    raise GameError(f"Cannot use group {key} more than once")

            self.check_terminals()
            if not suppress_check_other:
                self.check_other()
            self.check_cycles()
            self.check_distance(visited)
            self.check_overlap()
            self.check_connected()

            self._revenue = self.game.revenue_for(self, self.stops)
        return self._revenue

    @property
    def subsidy(self):
        if not hasattr(self.game, "subsidy_for"):
            return None
        if not self._subsidy:
            self._subsidy = self.game.subsidy_for(self, self.stops)
        return self._subsidy

    @property
    def revenue_str(self):
        if not self._revenue_str:
            self._revenue_str = self.game.revenue_str(self)
        return self._revenue_str

    @property
    def corporation(self):
        return self.game.train_owner(self._train)

    @property
    def connection_hexes(self):
        if not self._connection_hexes:
            if self._train.local and len(self.connection_data) == 1 and not self.connection_data[0]["chain"]["paths"]:
                self._connection_hexes = [["local", self.connection_data[0]["left"].hex.id]]
            else:
                self._connection_hexes = [self.chain_id(chain["paths"]) for chain in self.chains if chain]
        return self._connection_hexes

    @property
    def connection_data(self):
        if self._connection_data is not None:
            return self._connection_data

        self._connection_data = []
        if not self.connection_hexes:
            return self._connection_data

        if len(self.connection_hexes) == 1 and "local" in self.connection_hexes[0]:
            if self._train.local:
                city_node = next(
                    (
                        n
                        for n in self.game.hex_by_id(self.connection_hexes[0][1]).tile.nodes
                        if self.game.city_tokened_by(n, self.corporation)
                    ),
                    None,
                )
                if city_node:
                    self.add_single_node_connection(city_node)
                    return self._connection_data
            self.connection_hexes.clear()

        possibilities = [self.find_matching_chains(hex_ids) for hex_ids in self.connection_hexes]
        other_paths = self.compute_other_paths()

        if len(possibilities) == 1:
            chain = next(
                (
                    ch
                    for ch in possibilities[0]
                    if any(node for node in ch["nodes"] if self.game.city_tokened_by(node, self.corporation))
                    and not set(ch["paths"]).intersection(other_paths)
                ),
                None,
            )
            if not chain:
                return self._connection_data

            left, right = chain["nodes"]
            if not left or not right:
                return self._connection_data

            self._connection_data.append({"left": left, "right": right, "chain": chain})
        else:
            for index, pair in enumerate(zip(possibilities, possibilities[1:])):
                a, b = pair
                a, b, left, right, middle = self.find_pairwise_chain(a, b, other_paths)
                if not left.hex or not right.hex or not middle.hex:
                    return self._connection_data.clear()

                self._connection_data.append(
                    {"left": left, "right": middle, "chain": a}
                    if index == 0
                    else {"left": middle, "right": right, "chain": b}
                )
                other_paths.extend(a["paths"])

        return self._connection_data

    def chain_id(self, paths):
        if len(paths) == 1 and paths[0].tile.ambiguous_connection():
            node0, node1 = sorted([paths[0].nodes[0].index, paths[0].nodes[1].index])
            return [f"{paths[0].hex.id} {node0}.{node1}"]
        else:
            junction_map = {}
            hex_ids = []

            for path in paths:
                if not junction_map.get(path.a) and not junction_map.get(path.b):
                    hex_ids.append(path.hex.id)
                junction_map[path.a] = path.a.is_junction()
                junction_map[path.b] = path.b.is_junction()

            return hex_ids

    def add_single_node_connection(self, node):
        self._connection_data.append(
            {
                "left": node,
                "right": node,
                "chain": {"nodes": None, "paths": [], "hexes": None, "id": None},
            }
        )

    def find_pairwise_chain(self, chains_a, chains_b, other_paths):
        chains_a = [a for a in chains_a if not set(a["paths"]).intersection(other_paths)]
        chains_b = [b for b in chains_b if not set(b["paths"]).intersection(other_paths)]
        candidates = []

        for a in chains_a:
            for b in chains_b:
                middle = set(a["nodes"]).intersection(set(b["nodes"]))
                if not middle or set(b["paths"]).intersection(a["paths"]) or len(middle) != 1:
                    continue
                left = (set(a["nodes"]) - middle).pop()
                right = (set(b["nodes"]) - middle).pop()
                candidates.append((a, b, left, right, middle.pop()))

        if not candidates:
            return None, None, None, None, None
        if len(candidates) == 1:
            return candidates[0]

        if self._node_signatures:
            for a, b, left, right, middle in candidates:
                if all(n.signature in self._node_signatures for n in [left, right, middle]):
                    return a, b, left, right, middle

        return candidates[0]

    def find_matching_chains(self, hex_ids):
        if not hex_ids:
            return []

        start_hex = self.game.hex_by_id(hex_ids[0].split()[0])
        end_hex = self.game.hex_by_id(hex_ids[-1].split()[0])
        matching = []

        for start_node in start_hex.tile.nodes:
            for end_node in end_hex.tile.nodes:
                if start_node == end_node:
                    continue
                for ch in self.get_node_chains(start_node, end_node):
                    if ch["id"] == hex_ids:
                        matching.append(ch)

        return matching

    def compute_other_paths(self):
        other_paths = self.game.compute_other_paths(self.routes, self)
        for route in self.routes:
            route._paths = None
        return other_paths

    def local_connection(self):
        return (
            self._train.local
            and self.connection_data
            and self.connection_data[0]["left"] == self.connection_data[0]["right"]
        )


class TileConfig:
    WHITE = {
        "blank": "",
        "city": "city=revenue:0",
        "town": "town=revenue:0",
    }

    BLUE = {}
    RED = {}

    YELLOW = {
        "1": "town=revenue:10;town=revenue:10;path=a:1,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:4",
        "2": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2",
        "3": "town=revenue:10;path=a:0,b:_0;path=a:_0,b:1",
        "4": "town=revenue:10;path=a:0,b:_0;path=a:_0,b:3",
        "5": "city=revenue:20;path=a:0,b:_0;path=a:1,b:_0",
        "6": "city=revenue:20;path=a:0,b:_0;path=a:2,b:_0",
        "7": "path=a:0,b:1",
        "8": "path=a:0,b:2",
        "9": "path=a:0,b:3",
        "55": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4",
        "56": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:2;path=a:1,b:_1;path=a:_1,b:3",
        "57": "city=revenue:20;path=a:0,b:_0;path=a:_0,b:3",
        "58": "town=revenue:10;path=a:0,b:_0;path=a:_0,b:2",
        "69": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:4",
        "71": "town=revenue:10;town=revenue:10;path=a:0,b:_0,track:narrow;path=a:_0,b:_1,track:narrow;path=a:_1,b:4,track:narrow",
        "72": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:_0,b:1,track:narrow",
        "73": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:_0,b:2,track:narrow",
        "74": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:_0,b:3,track:narrow",
        "75": "city=revenue:20;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow",
        "76": "city=revenue:20;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow",
        "77": "path=a:0,b:1,track:narrow",
        "78": "path=a:0,b:2,track:narrow",
        "79": "path=a:0,b:3,track:narrow",
        "113": "city=revenue:20;path=a:0,b:_0,track:narrow",
        "115": "city=revenue:20;path=a:0,b:_0",
        "128": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:2,b:_1;label=C",
        "201": "city=revenue:30;path=a:0,b:_0;path=a:1,b:_0;label=Y",
        "202": "city=revenue:30;path=a:0,b:_0;path=a:2,b:_0;label=Y",
        "235": "city=revenue:30;city=revenue:30;path=a:0,b:_0;label=OO",
        # '403': 'city=revenue:30;town=revenue:30;path=a:0,b:_0;label=B;upgrade=cost:40',
        "437": "town=revenue:30;path=a:0,b:_0;path=a:_0,b:2;icon=image:port,blocks_lay:1",
        "438": "city=revenue:40;path=a:0,b:_0;path=a:2,b:_0;label=H;upgrade=cost:80",
        "445": "town=revenue:20;path=a:0,b:_0;path=a:_0,b:2;icon=image:18_al/tree,blocks_lay:1",
        "451a": "city=revenue:30;city=revenue:30;city=revenue:30;path=a:0,b:_0;path=a:2,b:_1;path=a:4,b:_2;label=ATL",
        "471": "city=revenue:20,loc:center;town=revenue:10,loc:4.5;path=a:0,b:_0;path=a:_0,b:3;path=a:_1,b:_0;label=M",
        "472": "city=revenue:20,loc:center;town=revenue:10,loc:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:_1,b:_0;label=T",
        "473": "city=revenue:20,loc:center;town=revenue:10,loc:4;path=a:0,b:_0;path=a:_0,b:2;path=a:_1,b:_0;label=V",
        # '601': 'city=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:2,b:_1;label=V',
        "621": "city=revenue:30;path=a:0,b:_0;path=a:_0,b:3;label=Y",
        "630": "town=revenue:10;town=revenue:10;path=a:2,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:4",
        "631": "town=revenue:10;town=revenue:10;path=a:3,b:_0;path=a:_0,b:4;path=a:0,b:_1;path=a:_1,b:2",
        "632": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3",
        "633": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:1;path=a:3,b:_1;path=a:_1,b:4",
        "644": "city=revenue:20;path=a:0,b:_0,track:narrow;path=a:1,b:_0",
        "645": "city=revenue:20;path=a:0,b:_0,track:narrow;path=a:2,b:_0",
        "657": "city=revenue:20;path=a:0,b:_0,track:narrow;path=a:3,b:_0",
        "658": "city=revenue:20;path=a:0,b:_0;path=a:2,b:_0,track:narrow",
        "659": "city=revenue:20;path=a:0,b:_0;path=a:1,b:_0,track:narrow",
        "679": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:_0,b:1,track:narrow",
        "790": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_0;path=a:_0,b:4;label=N",
        "956": "city=revenue:20;path=a:0,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "441a": "city=revenue:10;path=a:0,b:_0;label=B",
    }

    GREEN = {
        "10": "city=revenue:30;city=revenue:30;path=a:0,b:_0;path=a:3,b:_1",
        "11": "town=revenue:10;path=a:0,b:2;path=a:2,b:_0;path=a:_0,b:4;path=a:0,b:4;label=HALT",
        "12": "city=revenue:30;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0",
        "13": "city=revenue:30;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0",
        "14": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "15": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
        "16": "path=a:0,b:2;path=a:1,b:3",
        "17": "path=a:1,b:3;path=a:0,b:4",
        "18": "path=a:0,b:3;path=a:1,b:2",
        "19": "path=a:0,b:3;path=a:2,b:4",
        "20": "path=a:0,b:3;path=a:1,b:4",
        "21": "path=a:0,b:2;path=a:3,b:4",
        "22": "path=a:0,b:4;path=a:2,b:3",
        "23": "path=a:0,b:3;path=a:0,b:4",
        "24": "path=a:0,b:3;path=a:0,b:2",
        "25": "path=a:0,b:2;path=a:0,b:4",
        "26": "path=a:0,b:3;path=a:0,b:5",
        "27": "path=a:0,b:3;path=a:0,b:1",
        "28": "path=a:0,b:4;path=a:0,b:5",
        "29": "path=a:0,b:2;path=a:0,b:1",
        "30": "path=a:0,b:4;path=a:0,b:1",
        "31": "path=a:0,b:2;path=a:0,b:5",
        "52": "city=revenue:40,loc:5;city=revenue:40,loc:3;path=a:0,b:_0;path=a:2,b:_1;label=OO",
        "53": "city=revenue:50;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=B",
        "54": "city=revenue:60,loc:0.5;city=revenue:60,loc:2.5;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=NY",
        "59": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:2,b:_1;label=OO",
        "80": "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0",
        "81": "junction;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0",
        "82": "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0",
        "83": "junction;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0",
        "84": "path=a:0,b:1,track:narrow;path=a:1,b:2,track:narrow;path=a:0,b:2,track:narrow",
        "85": "path=a:3,b:5,track:narrow;path=a:0,b:5,track:narrow;path=a:0,b:3,track:narrow",
        "86": "path=a:0,b:1,track:narrow;path=a:1,b:3,track:narrow;path=a:0,b:3,track:narrow",
        "87": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
        "88": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "89": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "90": "city=revenue:20,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "91": "city=revenue:20,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow;path=a:5,b:_0",
        "92": "city=revenue:20,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0;path=a:5,b:_0",
        "93": "city=revenue:20,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0,track:narrow;path=a:4,b:_0;path=a:5,b:_0",
        "94": "city=revenue:20,slots:2;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:2,b:_0;path=a:3,b:_0,track:narrow",
        "95": "city=revenue:20,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0;path=a:2,b:_0,track:narrow;path=a:3,b:_0",
        "96": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "97": "city=revenue:30,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0;path=a:3,b:_0",
        "98": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0,track:narrow;path=a:4,b:_0",
        "99": "city=revenue:30,slots:2;path=a:0,b:_0,track:narrow;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "100": "city=revenue:30;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow",
        "101": "city=revenue:30;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "116": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "117": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow",
        "120": "city=revenue:60;city=revenue:60;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=T",
        "121": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=B-L",
        "129": "city=revenue:60,slots:2;city=revenue:60;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=C",
        "141": "town=revenue:10;path=a:0,b:_0;path=a:3,b:_0;path=a:1,b:_0",
        "142": "town=revenue:10;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0",
        "143": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0",
        "144": "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0",
        "190": "city=revenue:40;city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_0;path=a:_0,b:4;path=a:2,b:_0;path=a:_0,b:5;label=ATL",
        "203": "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0",
        "204": "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "205": "city=revenue:30;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0",
        "206": "city=revenue:30;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0",
        "207": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=Y",
        "208": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y",
        "209": "city=revenue:40,slots:3;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=B",
        "210": "city=revenue:30;city=revenue:30;path=a:0,b:_0;path=a:3,b:_0;path=a:5,b:_1;path=a:4,b:_1;label=XX",
        "211": "city=revenue:30;city=revenue:30;path=a:2,b:_0;path=a:3,b:_0;path=a:0,b:_1;path=a:1,b:_1;label=XX",
        "212": "city=revenue:30;city=revenue:30;path=a:2,b:_0;path=a:3,b:_0;path=a:0,b:_1;path=a:5,b:_1;label=XX",
        "213": "city=revenue:30;city=revenue:30;path=a:2,b:_0;path=a:3,b:_0;path=a:0,b:_1;path=a:4,b:_1;label=XX",
        "214": "city=revenue:30;city=revenue:30;path=a:4,b:_0;path=a:3,b:_0;path=a:0,b:_1;path=a:2,b:_1;label=XX",
        "215": "city=revenue:30;city=revenue:30;path=a:1,b:_0;path=a:3,b:_0;path=a:0,b:_1;path=a:4,b:_1;label=XX",
        "233": "path=a:0,b:3,track:dual",
        "234": "path=a:0,b:1,track:dual",
        "236": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;label=K",
        "237": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;label=K",
        "238": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0;label=K",
        "298LA": "city=revenue:40;city=revenue:40;city=revenue:40;city=revenue:40;label=LB;path=a:1,b:_0;path=a:2,b:_1;path=a:3,b:_2;path=a:4,b:_3;path=a:0,b:_0;path=a:0,b:_1;path=a:0,b:_2;path=a:0,b:_3",
        "439": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=H;upgrade=cost:80",
        "440": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;label=T",
        "441": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0",
        "442": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0",
        "443": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0",
        "444": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0",
        "452a": "city=revenue:20;city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:5;path=a:4,b:_2;path=a:_2,b:1;label=ATL",
        "453a": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:5,b:_0;label=Aug",
        "454a": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:5,b:_0;label=S",
        "457": "city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:_0,b:1;path=a:1,b:_1;path=a:_1,b:3;path=a:_0,b:2;path=a:2,b:_1",
        "458": "city=revenue:20;city=revenue:20;path=a:1,b:_0;path=a:_0,b:4;path=a:0,b:_1;path=a:_1,b:3;path=a:0,b:_0;path=a:_1,b:4",
        "459": "city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4;path=a:_0,b:1;path=a:3,b:_1",
        "460": "city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4;path=a:0,b:_0;path=a:_0,b:4;path=a:1,b:_1;path=a:_1,b:3",
        "461": "city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:2;path=a:1,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2",
        "462": "city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:3;path=a:0,b:_0;path=a:_0,b:2;path=a:1,b:_1;path=a:_1,b:2",
        "463": "city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:4;path=a:_0,b:4;path=a:_1,b:3",
        "464": "city=revenue:20;city=revenue:20;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:2,b:_1;path=a:3,b:_1;path=a:4,b:_1",
        "474": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "475": "city=revenue:30,loc:center;path=a:0,b:_0;path=a:1,b:_0;path=a:5,b:_0;label=L",
        "476": "city=revenue:30,loc:center;town=revenue:10,loc:0;path=a:_0,b:2;path=a:4,b:_0;path=a:5,b:_0;path=a:_1,b:_0;label=M",
        "477": "city=revenue:30,loc:center;town=revenue:10,loc:3.5;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:5,b:_0;path=a:_1,b:_0;label=T",
        "478": "city=revenue:30,slots:2,loc:center;town=revenue:20,loc:0;path=a:2,b:_0;path=a:_0,b:4;path=a:_1,b:_0;label=V",
        "514": "city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=P",
        "576": "city=revenue:40;path=a:0,b:_0;path=a:3,b:_0;path=a:1,b:_0;label=Y",
        "577": "city=revenue:40;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0;label=Y",
        "578": "city=revenue:40;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;label=Y",
        "579": "city=revenue:40;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=Y",
        "580": "city=revenue:60,loc:0.5;city=revenue:60,loc:2.5;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=P",
        "581": "city=revenue:50;city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;path=a:4,b:_2;path=a:_2,b:5;label=B-V",
        "590": "city=revenue:60;city=revenue:60;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;path=a:3,b:_0;path=a:_0,b:4;label=Chi",
        "592": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=B",
        "602": "city=revenue:30;town=revenue:30;path=a:0,b:_0;path=a:2,b:_1;path=a:3,b:_0;path=a:4,b:_1;label=V",
        "604": "city=revenue:100,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=M",
        "606": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=T",
        "612": "city=revenue:40;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=G",
        "613": "city=revenue:0;path=a:0,b:_0;path=a:3,b:_0",
        "614": "city=revenue:0;path=a:0,b:_0;path=a:2,b:_0",
        "615": "city=revenue:0;path=a:0,b:_0;path=a:1,b:_0",
        "619": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "622": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y",
        "624": "path=a:0,b:1;path=a:1,b:2",
        "625": "path=a:0,b:1;path=a:2,b:3",
        "626": "path=a:0,b:1;path=a:3,b:4",
        "637": "city=revenue:50,loc:0.5;city=revenue:50,loc:2.5;city=revenue:50,loc:4.5;path=a:0,b:_0;path=a:_0,b:1;path=a:4,b:_2;path=a:_2,b:5;path=a:2,b:_1;path=a:_1,b:3;label=M",
        "650": "path=a:0,b:1,track:narrow;path=a:1,b:2,track:narrow",
        "651": "city=revenue:90,slots:2;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0;path=a:4,b:_0;path=a:0,b:_0;label=P",
        "653": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:3,b:_0;label=C",
        "655": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=M",
        "660": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:3,b:_0;path=a:4,b:_0,track:narrow",
        "661": "city=revenue:30,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0;path=a:3,b:_0,track:narrow;path=a:4,b:_0",
        "662": "city=revenue:30,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:3,b:_0;path=a:4,b:_0",
        "663": "city=revenue:30,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0,track:narrow",
        "664": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0",
        "665": "city=revenue:30,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0,track:narrow",
        "666": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:2,b:_0;path=a:3,b:_0,track:narrow",
        "667": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "668": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:3,b:_0;path=a:2,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "669": "city=revenue:30,slots:2;path=a:2,b:_0;path=a:4,b:_0;path=a:0,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "670": "city=revenue:30,slots:2;path=a:2,b:_0;path=a:3,b:_0;path=a:0,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "671": "city=revenue:30,slots:2;path=a:3,b:_0;path=a:4,b:_0;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow",
        "675": "city=revenue:20;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:2,b:_0;label=S",
        "677": "path=a:0,b:3,track:narrow;path=a:0,b:4,track:narrow",
        "678": "path=a:0,b:3,track:narrow;path=a:0,b:2,track:narrow",
        "680": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0;path=a:3,b:_0",
        "681": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:1,b:_0;path=a:2,b:_0,track:narrow;path=a:3,b:_0",
        "682": "town=revenue:10;path=a:0,b:_0,track:narrow;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0,track:narrow",
        "683": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0",
        "684": "town=revenue:10;path=a:3,b:_0;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:4,b:_0",
        "685": "town=revenue:10;path=a:2,b:_0;path=a:0,b:_0,track:narrow;path=a:4,b:_0,track:narrow;path=a:3,b:_0",
        "686": "town=revenue:10;path=a:2,b:_0;path=a:0,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0",
        "687": "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0,track:narrow;path=a:4,b:_0,track:narrow;path=a:3,b:_0",
        "688": "town=revenue:10;path=a:1,b:_0;path=a:0,b:_0,track:narrow;path=a:4,b:_0,track:narrow;path=a:3,b:_0",
        "689": "town=revenue:10;path=a:4,b:_0;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:3,b:_0",
        "690": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:4,b:_0,track:narrow;path=a:3,b:_0",
        "691": "town=revenue:10;path=a:1,b:_0;path=a:0,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0",
        "692": "path=a:0,b:3,track:narrow;path=a:0,b:5,track:narrow",
        "693": "path=a:0,b:3,track:narrow;path=a:0,b:1,track:narrow",
        "694": "path=a:0,b:4,track:narrow;path=a:0,b:5,track:narrow",
        "695": "path=a:0,b:2,track:narrow;path=a:0,b:1,track:narrow",
        "699": "path=a:0,b:2,track:narrow;path=a:0,b:4,track:narrow",
        "700": "town=revenue:10;path=a:1,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:0,b:_0;path=a:2,b:_0",
        "701": "town=revenue:10;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:0,b:_0;path=a:1,b:_0",
        "702": "town=revenue:10;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:0,b:_0;path=a:4,b:_0",
        "703": "town=revenue:10;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow;path=a:0,b:_0;path=a:2,b:_0",
        "704": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "705": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:4,b:_0;path=a:3,b:_0,track:narrow;path=a:2,b:_0,track:narrow",
        "706": "city=revenue:30,slots:2;path=a:2,b:_0;path=a:3,b:_0;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow",
        "707": "city=revenue:30,slots:2;path=a:1,b:_0;path=a:3,b:_0;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow",
        "708": "path=a:0,b:4,track:narrow;path=a:0,b:1,track:narrow",
        "709": "path=a:0,b:2,track:narrow;path=a:0,b:5,track:narrow",
        "710": "path=a:0,b:2,track:narrow;path=a:1,b:3",
        "711": "path=a:0,b:3,track:narrow;path=a:2,b:4",
        "712": "path=a:0,b:2;path=a:1,b:3,track:narrow",
        "713": "path=a:0,b:3;path=a:2,b:4,track:narrow",
        "714": "path=a:0,b:3;path=a:1,b:4,track:narrow",
        "715": "path=a:0,b:3,track:narrow;path=a:1,b:4",
        "791": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=N",
        "792": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0;label=Y",
        "793": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;label=Y",
        "800": "town=revenue:30;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=D&SL",
        "802": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=D",
        "887": "town=revenue:20;path=a:1,b:_0;path=a:3,b:_0;path=a:0,b:_0;path=a:2,b:_0",
        "888": "town=revenue:20;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "901": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_0;path=a:_0,b:3",
        "904": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=B",
        "907": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0;label=Z",
        "908": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;label=Z",
        "957": "path=a:0,b:2,track:narrow;path=a:1,b:3,track:narrow",
        "958": "path=a:1,b:3,track:narrow;path=a:0,b:4,track:narrow",
        "959": "path=a:0,b:3,track:narrow;path=a:1,b:2,track:narrow",
        "960": "path=a:0,b:3,track:narrow;path=a:2,b:4,track:narrow",
        "961": "path=a:0,b:3,track:narrow;path=a:1,b:4,track:narrow",
        "962": "city=revenue:30;path=a:0,b:_0,track:narrow;path=a:5,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "963": "city=revenue:30;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "964": "city=revenue:20;city=revenue:20;path=a:0,b:_0,track:narrow;path=a:_0,b:1,track:narrow;path=a:3,b:_1,track:narrow;path=a:_1,b:4,track:narrow",
        "965": "city=revenue:20;city=revenue:20;path=a:1,b:_0,track:narrow;path=a:_0,b:3,track:narrow;path=a:0,b:_1,track:narrow;path=a:_1,b:4,track:narrow",
        "966": "city=revenue:20;city=revenue:20;path=a:0,b:_0,track:narrow;path=a:_0,b:2,track:narrow;path=a:1,b:_1,track:narrow;path=a:_1,b:3,track:narrow",
        "967": "city=revenue:20;city=revenue:20;path=a:0,b:_0,track:narrow;path=a:_0,b:2,track:narrow;path=a:3,b:_1,track:narrow;path=a:_1,b:4,track:narrow",
        "968": "city=revenue:20;city=revenue:20;path=a:0,b:_0,track:narrow;path=a:_0,b:3,track:narrow;path=a:1,b:_1,track:narrow;path=a:_1,b:2,track:narrow",
        "969": "city=revenue:20;city=revenue:20;path=a:0,b:_0,track:narrow;path=a:_0,b:3,track:narrow;path=a:2,b:_1,track:narrow;path=a:_1,b:4,track:narrow",
        "970": "city=revenue:20;city=revenue:20;path=a:0,b:_0,track:narrow;path=a:_0,b:3,track:narrow;path=a:1,b:_1,track:narrow;path=a:_1,b:4,track:narrow",
        "971": "city=revenue:40;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;frame=color:#800080",
        "972": "city=revenue:40;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow;frame=color:#800080",
        "973": "city=revenue:40;path=a:0,b:_0,track:narrow;path=a:3,b:_0,track:narrow;frame=color:#800080",
        "974": "city=revenue:60,slots:2;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow;label=B;frame=color:#800080",
        "981": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:2",
        "991": "town=revenue:10;town=revenue:10;path=a:0,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:4",
        "53Y": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:4,b:_0;label=Y",
        "442a": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=B",
        "443a": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=M",
        # '479|a': 'city=revenue:40,slots:2;town=revenue:40;path=a:3,b:_0;path=a:5,b:_0',
        # '479|b': 'town=revenue:10;path=a:2,b:_0;path=a:_0,b:5;upgrade=cost:40,terrain:mountain',
        # '802|3': 'city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=D',
        "8858": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:2;path=a:1,b:_1;path=a:_1,b:3;label=OO",
        "8859": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:5;label=OO",
        "8860": "city=revenue:40;city=revenue:40;path=a:1,b:_0;path=a:_0,b:5;path=a:2,b:_1;path=a:_1,b:4;label=OO",
        "8863": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:5;label=OO",
        "8864": "city=revenue:40;city=revenue:40;path=a:1,b:_0;path=a:_0,b:5;path=a:2,b:_1;path=a:_1,b:3;label=OO",
        "8865": "city=revenue:40;city=revenue:40;path=a:1,b:_0;path=a:_0,b:5;path=a:3,b:_1;path=a:_1,b:4;label=OO",
    }

    BROWN = {
        "32": "city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;path=a:0,b:_0;path=a:1,b:_1;path=a:2,b:_2;path=a:3,b:_3;path=a:4,b:_4;path=a:5,b:_5;label=L",
        "33": "city=revenue:50,loc:0;city=revenue:50,loc:2;city=revenue:50,loc:4;path=a:5,b:_0;path=a:3,b:_1;path=a:4,b:_2;label=L",
        "34": "city=revenue:50,loc:1.5;city=revenue:50,loc:4.5;city=revenue:50,loc:3;path=a:0,b:_2;path=a:_2,b:3;path=a:2,b:_0;path=a:4,b:_1;label=BGM",
        "35": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:2;path=a:1,b:_1;path=a:_1,b:3",
        "36": "city=revenue:40;city=revenue:40;path=a:1,b:_0;path=a:_0,b:3;path=a:0,b:_1;path=a:_1,b:4",
        "37": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:3;path=a:3,b:_1;path=a:0,b:_0",
        "38": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "39": "path=a:0,b:2;path=a:0,b:1;path=a:1,b:2",
        "40": "path=a:0,b:2;path=a:2,b:4;path=a:0,b:4",
        "41": "path=a:0,b:3;path=a:0,b:1;path=a:1,b:3",
        "42": "path=a:0,b:3;path=a:3,b:5;path=a:0,b:5",
        "43": "path=a:0,b:3;path=a:0,b:2;path=a:1,b:3;path=a:1,b:2",
        "44": "path=a:0,b:3;path=a:1,b:4;path=a:0,b:1;path=a:3,b:4",
        "45": "path=a:0,b:3;path=a:2,b:4;path=a:0,b:4;path=a:2,b:3",
        "46": "path=a:0,b:3;path=a:2,b:4;path=a:3,b:4;path=a:0,b:2",
        "47": "path=a:0,b:3;path=a:1,b:4;path=a:1,b:3;path=a:0,b:4",
        "61": "city=revenue:60;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=B",
        "62": "city=revenue:80,slots:2;city=revenue:80,slots:2;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=NY",
        "63": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "64": "city=revenue:50;city=revenue:50,loc:3.5;path=a:0,b:_0;path=a:_0,b:2;path=a:3,b:_1;path=a:_1,b:4;label=OO",
        "65": "city=revenue:50;city=revenue:50,loc:2.5;path=a:0,b:_0;path=a:_0,b:4;path=a:2,b:_1;path=a:_1,b:3;label=OO",
        "66": "city=revenue:50;city=revenue:50,loc:1.5;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2;label=OO",
        "67": "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:4;label=OO",
        "68": "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4;label=OO",
        "70": "path=a:0,b:1;path=a:0,b:2;path=a:1,b:3;path=a:2,b:3",
        "102": "city=revenue:30,slots:2;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual;path=a:5,b:_0,track:dual",
        "103": "city=revenue:40,slots:2;path=a:0,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "104": "city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual;path=a:5,b:_0,track:dual;label=CP",
        "105": "city=revenue:40,slots:3;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;label=BM",
        "106": "junction;path=a:0,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:4,b:_0,track:dual;path=a:3,b:_0,track:dual",
        "107": "junction;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "108": "junction;path=a:0,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual;path=a:5,b:_0,track:dual",
        "118": "city=revenue:40;city=revenue:40;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3",
        "122": "city=revenue:80,slots:2;city=revenue:80,slots:2;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=T",
        "125": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "126": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=Lon",
        "127": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Bar",
        "130": "city=revenue:100,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=C",
        "132": "city=revenue:70;city=revenue:70;city=revenue:70;city=revenue:70;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=C",
        "133": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;label=M",
        "135": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=KC;label=SL;label=MSP",
        "145": "town=revenue:20;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "146": "town=revenue:20;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
        "147": "town=revenue:20;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "148": "town=revenue:20;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "168": "city=revenue:30;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0;label=Y",
        "169": "city=revenue:30;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;label=Y",
        "170": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=P",
        "191": "city=revenue:60,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=A",
        "193": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=S",
        "216": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y",
        "217": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:4,b:_0;path=a:5,b:_0;path=a:3,b:_0;label=X",
        "218": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=X",
        "219": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:5,b:_0;label=X",
        "220": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=B",
        "221": "city=revenue:60,slots:2,loc:3;city=revenue:60,loc:0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:1,b:_1;path=a:0,b:_1;path=a:5,b:_1;path=a:_0,b:_1;label=H",
        "239": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=K",
        "448": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
        "449": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "450": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "455a": "city=revenue:70;city=revenue:70;city=revenue:70;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:5;path=a:4,b:_2;path=a:_2,b:1;label=ATL",
        "456a": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:5,b:_0;label=Aug",
        "457a": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=B",
        "458a": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=M",
        "459a": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:5,b:_0;label=S",
        "465": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=K",
        "466": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;label=T",
        "480": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=G",
        "481": "city=revenue:40,slots:2,loc:center;path=a:0,b:_0;path=a:1,b:_0;path=a:5,b:_0;label=L",
        "482": "city=revenue:40,slots:2,loc:center;town=revenue:20,loc:0;path=a:2,b:_0;path=a:_0,b:3;path=a:4,b:_0;path=a:5,b:_0;path=a:_1,b:_0;label=M",
        "483": "city=revenue:40,loc:center;town=revenue:20,loc:3.5;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:5,b:_0;path=a:_1,b:_0;label=T",
        "484": "city=revenue:40,slots:2,loc:center;town=revenue:30,loc:0;path=a:2,b:_0;path=a:3,b:_0;path=a:_0,b:4;path=a:_1,b:_0;label=V",
        "492": "city=revenue:80,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=H",
        "515": "city=revenue:90;city=revenue:90;city=revenue:90;city=revenue:90;city=revenue:90;city=revenue:90;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=P",
        "544": "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "545": "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
        "546": "junction;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "582": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y",
        "583": "city=revenue:80,slots:2;city=revenue:80,slots:2;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;label=P",
        "584": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=B-V",
        "591": "city=revenue:80,slots:2;city=revenue:80,slots:2;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_0;path=a:_0,b:3;path=a:3,b:_0;path=a:_0,b:4;label=Chi",
        "593": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=B",
        "603": "city=revenue:30;town=revenue:30;path=a:0,b:_0;path=a:1,b:_1;path=a:2,b:_0;path=a:3,b:_1;path=a:4,b:_0;path=a:5,b:_1;label=V",
        "605": "city=revenue:150,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=M",
        "607": "city=revenue:90,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=T",
        "609": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=G",
        "610": "city=revenue:0;path=a:0,b:_0;path=a:3,b:_0",
        "611": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "616": "city=revenue:0,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "617": "city=revenue:0,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
        "618": "city=revenue:0,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "623": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=Y",
        "627": "path=a:0,b:3;path=a:0,b:1;path=a:1,b:2;path=a:2,b:3",
        "628": "path=a:1,b:3;path=a:3,b:4;path=a:0,b:4;path=a:0,b:1",
        "629": "path=a:0,b:2;path=a:2,b:3;path=a:3,b:4;path=a:0,b:4",
        "646": "path=a:0,b:2,track:narrow;path=a:2,b:4,track:narrow;path=a:0,b:4,track:narrow",
        "647": "path=a:0,b:2,track:narrow;path=a:0,b:1,track:narrow;path=a:1,b:2,track:narrow",
        "648": "path=a:0,b:3,track:narrow;path=a:0,b:1,track:narrow;path=a:1,b:3,track:narrow",
        "649": "path=a:0,b:3,track:narrow;path=a:3,b:5,track:narrow;path=a:0,b:5,track:narrow",
        "652": "city=revenue:130,slots:2;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0;label=P",
        "654": "city=revenue:90,slots:2;path=a:2,b:_0,track:dual;path=a:0,b:_0;path=a:4,b:_0,track:dual;path=a:3,b:_0;label=C",
        "656": "city=revenue:80,slots:2;path=a:2,b:_0;path=a:3,b:_0,track:narrow;path=a:4,b:_0;path=a:0,b:_0,track:dual;label=M",
        "672": "city=revenue:40,slots:2;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "673": "city=revenue:40,slots:2;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual",
        "674": "city=revenue:40,slots:2;path=a:0,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "676": "city=revenue:30,slots:2;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0;label=S",
        "696": "town=revenue:20;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "697": "town=revenue:20;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual",
        "698": "town=revenue:20;path=a:0,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "767": "town=revenue:10;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "768": "town=revenue:10;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "769": "town=revenue:10;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "794": "city=revenue:80,slots:4;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_0;path=a:_0,b:4;path=a:2,b:_0;path=a:_0,b:5;label=N",
        "796": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_0;path=a:_0,b:4;label=Y",
        "798": "path=a:0,b:3;path=a:1,b:4;path=a:2,b:5",
        "801": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=Y",
        "803": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=D",
        "804": "city=revenue:40;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0",
        "902": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=L",
        "905": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=B",
        "909": "city=revenue:50,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Z",
        "911": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "975": "city=revenue:40,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "976": "city=revenue:40,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow",
        "977": "city=revenue:40,slots:2;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "978": "city=revenue:60,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;frame=color:#800080",
        "979": "city=revenue:60,slots:2;path=a:0,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:4,b:_0,track:narrow;frame=color:#800080",
        "980": "city=revenue:60,slots:2;path=a:0,b:_0,track:narrow;path=a:5,b:_0,track:narrow;path=a:3,b:_0,track:narrow;frame=color:#800080",
        "985": "city=revenue:60,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:3,b:_0,track:narrow;frame=color:#800080",
        "986": "city=revenue:80,slots:2;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow;label=B;frame=color:#800080",
        "987": "city=revenue:90,slots:2;path=a:0,b:_0;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0;label=HQG;frame=color:#800080",
        "997": "city=revenue:60,slots:2;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:5,b:_0;label=Boston",
        "1064": "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:2;path=a:3,b:_1;path=a:_1,b:4",
        "1065": "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:4;path=a:2,b:_1;path=a:_1,b:3",
        "1066": "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:2",
        "1067": "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:2,b:_1;path=a:_1,b:4",
        "1068": "city=revenue:50;city=revenue:50;path=a:0,b:_0;path=a:_0,b:3;path=a:1,b:_1;path=a:_1,b:4",
        "61Y": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Y",
        "444b": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=B",
        "444m": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=M",
        "891Y": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=Y",
    }

    GRAY = {
        "48": "city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;label=L",
        "49": "city=revenue:70,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;label=L",
        "50": "city=revenue:70,loc:1.5;city=revenue:70,loc:3;city=revenue:70,loc:4.5;path=a:0,b:_1;path=a:_1,b:3;path=a:1,b:_0;path=a:_0,b:2;path=a:4,b:_2;path=a:_2,b:5;label=BGM",
        "51": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "60": "junction;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "109": "city=revenue:50,slots:2;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "110": "city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual;path=a:5,b:_0,track:dual;label=CP",
        "111": "city=revenue:70,slots:3;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;label=BM",
        "112": "junction;path=a:0,b:_0,track:dual;path=a:1,b:_0,track:dual;path=a:2,b:_0,track:dual;path=a:3,b:_0,track:dual;path=a:4,b:_0,track:dual",
        "114": "path=a:0,b:2;path=a:2,b:4;path=a:0,b:4;path=a:1,b:3;path=a:3,b:5;path=a:1,b:5",
        "123": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=H",
        "124": "city=revenue:100,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=T",
        "131": "city=revenue:100,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=C",
        "134": "city=revenue:100,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;label=M",
        "136": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=KC;label=SL;label=MSP",
        "167": "city=revenue:70,loc:0.5;city=revenue:70,loc:2.5;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;path=a:4,b:_0;path=a:5,b:_1;label=OO",
        "171": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "172": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "232": "city=revenue:100;city=revenue:100;city=revenue:100;city=revenue:100;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=C",
        "240": "city=revenue:80,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=K",
        "446": "city=revenue:70,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=B",
        "452": "path=a:0,b:3;path=a:2,b:4;path=a:0,b:4;path=a:0,b:2;path=a:2,b:3;path=a:3,b:4",
        "453": "path=a:0,b:3;path=a:1,b:4;path=a:1,b:3;path=a:0,b:4;path=a:0,b:1;path=a:3,b:4",
        "454": "path=a:0,b:3;path=a:1,b:3;path=a:0,b:2;path=a:0,b:1;path=a:1,b:2;path=a:2,b:3",
        "455": "city=revenue:50,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "513": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "516": "city=revenue:120;city=revenue:120;city=revenue:120;city=revenue:120;city=revenue:120;city=revenue:120;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=P",
        "596": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=L",
        "597": "city=revenue:80,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=B",
        "639": "city=revenue:100,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=M",
        "805": "city=revenue:60,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=D",
        "806": "town=revenue:10;path=a:0,b:_0;path=a:5,b:_0;path=a:3,b:_0",
        "807": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:3,b:_0",
        "808": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0",
        "895": "city=revenue:50,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "903": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;label=L",
        "906": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0;label=B",
        "910": "city=revenue:60,slots:4;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;label=Z",
        "912": "town=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0;path=a:5,b:_0",
        "915": "city=revenue:50,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "988": "city=revenue:50,slots:3;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow",
        "989": "city=revenue:70,slots:3;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow;frame=color:#800080",
        "990": "city=revenue:100,slots:3;path=a:0,b:_0,track:narrow;path=a:1,b:_0,track:narrow;path=a:2,b:_0,track:narrow;path=a:3,b:_0,track:narrow;path=a:4,b:_0,track:narrow;path=a:5,b:_0,track:narrow;label=B;frame=color:#800080",
        "1167": "city=revenue:70,loc:0;city=revenue:70,loc:3;path=a:0,b:_0;path=a:_0,b:1;path=a:2,b:_1;path=a:_1,b:3;path=a:4,b:_0;path=a:5,b:_1",
        "1168": "city=revenue:60,slots:3;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0",
    }

    GREENBROWN = {
        "119": "city=revenue:30,slots:2;path=a:0,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
    }

    BROWNGRAY = {
        "166": "city=revenue:40,slots:2;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
        "1200": "city=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:4,b:_0;path=a:5,b:_0",
    }

    BROWNSEPIA = {
        "200": "city=revenue:10;path=a:0,b:_0;path=a:1,b:_0;path=a:2,b:_0;path=a:3,b:_0;path=a:4,b:_0",
    }

    NONE = {
        "467": "path=a:0,b:3,track:narrow",
        "468": "path=a:0,b:2,track:narrow",
        "469": "path=a:0,b:1,track:narrow",
    }

    COLORS = [
        "white",
        "yellow",
        "green",
        "brown",
        "gray",
        "blue",
        "sepia",
        "greenbrown",
        "browngray",
        "brownsepia",
        "none",
        "red",
        "purple",
        "orange",
        "gray60",
        "gray50",
        "gray40",
        "salmon",
    ]


class Tile(TileConfig):
    ALL_EDGES = [0, 1, 2, 3, 4, 5]

    @classmethod
    def for_tile(cls, name, **opts):
        color = None
        code = None

        if name in cls.WHITE:
            color, code = "white", cls.WHITE[name]
        elif name in cls.YELLOW:
            color, code = "yellow", cls.YELLOW[name]
        elif name in cls.GREEN:
            color, code = "green", cls.GREEN[name]
        elif name in cls.GREENBROWN:
            color, code = "green", cls.GREENBROWN[name]
            code = "stripes=color:brown;" + code if code else "stripes=color:brown"
        elif name in cls.BROWN:
            color, code = "brown", cls.BROWN[name]
        elif name in cls.BROWNGRAY:
            color, code = "brown", cls.BROWNGRAY[name]
            code = "stripes=color:gray;" + code if code else "stripes=color:gray"
        elif name in cls.GRAY:
            color, code = "gray", cls.GRAY[name]
        elif name in cls.RED:
            color, code = "red", cls.RED[name]
        elif name in cls.BLUE:
            color, code = "blue", cls.BLUE[name]
        elif name in cls.BROWNSEPIA:
            color, code = "brown", cls.BROWNSEPIA[name]
            code = "stripes=color:sepia;" + code if code else "stripes=color:sepia"
        else:
            raise GameError(f"Tile '{name}' not found")

        return cls.from_code(name, color, code, **opts)

    @classmethod
    def decode(cls, code):
        cache = []
        parts = []

        for part_code in code.split(";"):
            type_param, _, params = part_code.partition("=")
            if ":" in params:
                params = dict(param.split(":") for param in params.split(","))
            else:
                params = {type_param: params}

            part = cls.part(type_param, params, cache)
            if part:
                if isinstance(part, list):
                    parts.extend(part)
                else:
                    parts.append(part)

        return parts

    @classmethod
    def from_code(cls, name, color, code, **opts):
        return Tile(name=name, color=color, code=code, parts=cls.decode(code), **opts)

    @staticmethod
    def part(type, params, cache):
        if type == "path":
            for k, v in params.items():
                if k in ["terminal", "a_lane", "b_lane", "ignore", "track"]:
                    params[k] = v
                elif k == "lanes":
                    params[k] = int(v)
                else:
                    if v[0] == "_":
                        params[k] = cache[int(v[1:])]
                    else:
                        params[k] = Edge(v)
            return Path.make_lanes(**params)

        elif type == "city":
            city = City(**params)
            cache.append(city)
            return city

        elif type == "pass":
            pass_ = Pass(**params)
            cache.append(pass_)
            return pass_

        elif type == "town":
            town = Town(**params)
            cache.append(town)
            return town

        elif type == "halt":
            halt = Halt(params.get("symbol"), **params)
            cache.append(halt)
            return halt

        elif type == "offboard":
            offboard = Offboard(**params)
            cache.append(offboard)
            return offboard

        elif type == "label":
            return Label(**params)

        elif type == "upgrade":
            terrain = params.get("terrain", "").split("|") if "terrain" in params else None
            return Upgrade(params.get("cost"), terrain, params.get("size"), loc=params.get("loc"))

        elif type == "border":
            return Border(params.get("edge"), params.get("type"), params.get("cost"), params.get("color"))

        elif type == "junction":
            junction = Junction()
            cache.append(junction)
            return junction

        elif type == "icon":
            return Icon(
                params.get("image"),
                params.get("name"),
                params.get("sticky"),
                params.get("blocks_lay"),
                **params,
            )

        elif type == "stub":
            return Stub(int(params.get("edge")))

        elif type == "partition":
            return Partition(params.get("a"), params.get("b"), params.get("type"), params.get("restrict"))

        elif type == "frame":
            return Frame(params.get("color"), params.get("color2"))

        elif type == "stripes":
            return Stripes(params.get("color"))

        elif type == "future_label":
            return FutureLabel(params.get("label"), params.get("color"))

        if type == "blank":
            return None

        else:
            raise GameError(f"unknown part type: {type}")

    def __init__(
        self,
        name,
        code,
        color,
        parts,
        rotation=0,
        preprinted=False,
        index=0,
        location_name=None,
        **opts,
    ):
        self.name = name
        self.code = code
        self.color = color
        self.parts = []
        for part in parts:
            if isinstance(part, list):
                self.parts += part
            else:
                self.parts.append(part)
        self.rotation = rotation
        self.cities = []
        self.paths_cache = None
        self._exits = None
        self._paths = []
        self._exit_count = None
        self.future_paths = []
        self.stubs = []
        self.partitions = []
        self.towns = []
        self.city_towns = []
        self.all_stop = []
        self.upgrades = []
        self.offboards = []
        self.original_borders = []
        self.borders = []
        self.nodes = None
        self.stops = None
        self.edges = None
        self.frame = None
        self.stripes = None
        self.junction = None
        self.hex = None
        self.icons = []
        self.location_name = location_name
        self.legal_rotations = []
        self.blockers = []
        self.hidden_blockers = []
        self.reservations = []
        self.preprinted = preprinted
        self.index = index
        self.blocks_lay = None
        self.reservation_blocks = opts.get("reservation_blocks", "never")
        self.unlimited = opts.get("unlimited", False)
        self.labels = []
        self.future_label = None
        self.halts = []
        self.opposite = None
        self.hidden = opts.get("hidden", False)
        self.id = f"{self.name}-{self.index}"
        self._preferred_city_town_edges = None

        self.separate_parts()

    def duplicate(self):
        """Create a duplicate of the tile with incremented index."""
        return Tile(
            self.name,
            code=self.code,
            color=self.color,
            parts=Tile.decode(self.code),
            rotation=self.rotation,
            preprinted=self.preprinted,
            index=self.index + 1,
            location_name=self.location_name,
            reservation_blocks=self.reservation_blocks,
            unlimited=self.unlimited,
            hidden=self.hidden,
        )

    def __lt__(self, other):
        """Define the less than comparison based on color and numeric tile name."""
        color_order = TileConfig.COLORS.index(self.color)
        other_color_order = TileConfig.COLORS.index(other.color)
        return (color_order, int(self.name)) < (other_color_order, int(other.name))

    def rotate_absolute(self, absolute=None):
        """Rotate the tile to an absolute rotation or the next legal rotation."""
        new_rotation = None
        if absolute is not None:
            new_rotation = absolute
        else:
            new_rotation = absolute or next(
                (r for r in self.legal_rotations if r > self.rotation),
                self.legal_rotations[0] if self.legal_rotations else self.rotation,
            )
        self.rotation = new_rotation
        for node in self.nodes:
            node.clear()
        if self.junction:
            self.junction.clear()
        self.paths_cache = None
        self._exits = None
        self._exit_count = None
        self._preferred_city_town_edges = None
        return self

    def rotate(self, num, ticks=1):
        """Rotate a given number by a specified number of ticks."""
        return (num + ticks) % 6

    @property
    def paths(self):
        if not self.paths_cache:
            self.paths_cache = [path.rotate(self.rotation) for path in self._paths]
        return self.paths_cache

    @property
    def exits(self):
        """Get unique rotated exits."""
        if not self._exits:
            self._exits = list(set(self.rotate(e.num, self.rotation) for e in self.edges))
        return self._exits

    def converging_exit(self, num):
        """Check if an exit has more than one path converging."""
        return self.exit_count[num] > 1

    @property
    def exit_count(self):
        """Count exits post-rotation."""
        if not self._exit_count:
            counts = defaultdict(int)
            for edge in self.edges:
                counts[self.rotate(edge.num, self.rotation)] += 1
            self._exit_count = dict(counts)
        return self._exit_count

    @property
    def ignore_gauge_walk(self):
        """The getter for ignore_gauge_walk."""
        return self._ignore_gauge_walk

    @ignore_gauge_walk.setter
    def ignore_gauge_walk(self, val):
        """The setter for ignore_gauge_walk."""
        self._ignore_gauge_walk = val
        for path in self._paths:
            path.ignore_gauge_walk = val
        for node in self.nodes:
            node.clear()
        if self.junction:
            self.junction.clear()
        self.paths_cache = None

    @property
    def ignore_gauge_compare(self):
        """The getter for ignore_gauge_compare."""
        return self._ignore_gauge_compare

    @ignore_gauge_compare.setter
    def ignore_gauge_compare(self, val):
        """The setter for ignore_gauge_compare."""
        self._ignore_gauge_compare = val
        for path in self._paths:
            path.ignore_gauge_compare = val
        for node in self.nodes:
            node.clear()
        if self.junction:
            self.junction.clear()
        self.paths_cache = None

    @property
    def terrain(self):
        """Get unique terrains from upgrades."""
        return list(set(terrain for upgrade in self.upgrades for terrain in upgrade.terrains))

    def ambiguous_connection(self):
        """Check if the tile has ambiguous intra-tile paths."""
        return len([path for path in self._paths if len(path.nodes) > 1]) > 1

    def paths_are_subset_of(self, other_paths):
        """Check if the tile's paths are a subset of another set of paths."""
        if self.junction and any(path.junction for path in other_paths):
            other_exits = set(path.exits for path in other_paths)
            return any(
                (set(self.exits) - set((e + ticks) % 6 for e in other_exits)).empty() for ticks in Tile.ALL_EDGES
            )
        else:
            return any(
                all(any(path.rotate(ticks) <= other for other in other_paths) for path in self._paths)
                for ticks in Tile.ALL_EDGES
            )

    def add_blocker(self, private_company, hidden=False):
        self.blockers.append(private_company)
        if hidden:
            self.hidden_blockers.append(private_company)

    def __str__(self):
        hex_name = self.hex.name if self.hex else None
        return f"<{self.__class__.__name__}: {self.name}, hex: {hex_name}>"

    def __repr__(self):
        return self.__str__()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = memo.get(id(self))
        if result:
            return result

        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy the attributes used in __hash__ first
        setattr(result, "id", copy.deepcopy(self.id, memo))

        # Copy the rest of the attributes
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __hash__(self):
        return hash((self.id))

    @property
    def preferred_city_town_edges(self):
        if self._preferred_city_town_edges is None:
            self._preferred_city_town_edges = self.compute_city_town_edges()
        return self._preferred_city_town_edges

    def reserved_by(self, corporation):
        return any(r for r in self.reservations if r == corporation or getattr(r, "owner", None) == corporation)

    def add_reservation(self, entity, city, slot=None, reserve_city=True):
        city_index = 0 if len(self.cities) == 1 and reserve_city else city
        if city_index is not None and slot is None:
            slot = self.cities[city_index].get_slot(entity)

        if city_index is not None and slot is not None:
            self.cities[city_index].add_reservation(entity, slot)
        else:
            self.reservations.append(entity)

    def remove_reservation(self, entity):
        for city in self.cities:
            if city.reserved_by(entity):
                city.remove_reservation(entity)
                break
        else:
            self.reservations.remove(entity)

    def token_blocked_by_reservation(self, corporation):
        if not self.reservations:
            return False

        if self.reservation_blocks == "always" or (
            self.reservation_blocks == "single_slot_cities" and any(city.slots == 1 for city in self.cities)
        ):
            return corporation not in self.reservations
        else:
            return sum(1 for x in self.reservations if x != corporation) >= sum(
                city.available_slots for city in self.cities
            )

    @property
    def city_town_edges(self):
        ct_edges = {}
        for path in self.paths:
            ct = path.city or path.town
            if ct:
                ct_edges.setdefault(ct, []).extend(path.exits)
        return ct_edges.values()

    def city_town_edges_are_subset_of(self, other_cte):
        cte = self.city_town_edges
        return any(
            all(
                any(all(self.rotate(edge, rotation) in other_city for edge in city) for other_city in other_cte)
                for city in cte
            )
            for rotation in self.ALL_EDGES
        )

    def compute_loc(self, loc=None):
        if loc is None or loc == "center":
            return None
        return (float(loc) + self.rotation) % 6

    def compute_city_town_edges(self):
        ct_edges = defaultdict(list)
        edge_count = defaultdict(float)

        # Handling special cases for multiple city/town option tiles
        if not self._paths and len(self.cities) == 2 and len(self.towns) == 2:
            div = 3
            for index, x in enumerate(self.cities + self.towns):
                edge_count[x] = index * div
            return edge_count

        # Avoid cities rendering on top of each other if the tile has no paths but multiple cities
        if not self._paths and len(self.cities) >= 2:
            div = 6 / len(self.cities)
            for index, x in enumerate(self.cities):
                edge_count[x] = index * div
            return edge_count

        # Place single city or town in the center if applicable
        if len(self.cities) == 1 and not self.towns and not self.compute_loc(self.cities[0].loc):
            ct_edges[self.cities[0]] = None
            return ct_edges
        if (
            not self.cities
            and len(self.towns) == 1
            and len(self.towns[0].exits) != 2
            and not self.compute_loc(self.towns[0].loc)
        ):
            ct_edges[self.towns[0]] = None
            return ct_edges

        edge_count[0] += 0.1  # Prefer keeping room at the bottom for location name

        # Populate ct_edges and edge_count
        for path in self.paths:
            ct = path.city or path.town
            if ct:
                for edge in path.exits:
                    ct_edges[ct].append(edge)
                    edge_count[edge] += 1
                    edge_count[(edge + 1) % 6] += 0.1
                    edge_count[(edge - 1) % 6] += 0.1

        # Sorting and final processing
        final_ct_edges = {}
        for ct, edges in sorted(ct_edges.items(), key=lambda item: min(edge_count[e] for e in item[1])):
            edge = self.compute_loc(ct.loc) if ct.loc else min(edges, key=lambda e: edge_count[e])
            if not ct.loc:
                edge_count[edge] += 1
                edge_count[(edge + 1) % 6] += 0.1
                edge_count[(edge - 1) % 6] += 0.1
            final_ct_edges[ct] = edge

        # Handling pathless and exitless city/towns
        self.handle_special_cases(ct_edges, final_ct_edges)

        return final_ct_edges

    def handle_special_cases(self, ct_edges, final_ct_edges):
        # Handling city/towns with no paths when there's only one other city/town
        pathless_cts = [ct for ct in self.cities + self.towns if not ct.paths and len(self.cities + self.towns) == 2]
        if len(pathless_cts) == 1:
            ct = pathless_cts[0]
            other_ct_edge = next(iter(final_ct_edges.values()), None)
            if other_ct_edge is not None:
                final_ct_edges[ct] = (other_ct_edge + 3) % 6

        # Handling city/towns with no exits
        exitless_cts = [ct for ct in self.cities + self.towns if not ct.exits]
        for ct in exitless_cts:
            if ct.loc:
                final_ct_edges[ct] = self.compute_loc(ct.loc)

    def crossover(self):
        if self._crossover is None:
            self._crossover = self.compute_crossover()
        return self._crossover

    def compute_crossover(self):
        if len(self._paths) <= 1:
            return False

        edge_paths = defaultdict(list)
        for p in self.paths:
            if len(p.nodes) > 1 or p.a_num == p.b_num:
                continue
            edge_paths[p.a_num].append(p)
            edge_paths[p.b_num].append(p)

        for p in self.paths:
            if len(p.nodes) > 1:
                continue

            a_num, b_num = p.a_num, p.b_num
            if p.straight():
                if any(ep.straight() for ep in edge_paths[(a_num + 1) % 6]):
                    return True
                if any(ep.straight() for ep in edge_paths[(a_num - 1) % 6]):
                    return True
            elif p.gentle_curve():
                low = min(a_num, b_num)
                middle = (low + 1) % 6 if abs(a_num - b_num) == 2 else (low - 1) % 6
                if any(ep.straight() or ep.gentle_curve() for ep in edge_paths[middle]):
                    return True
        return False

    def revenue_to_render(self):
        return [stop.revenue_to_render for stop in self.revenue_stops]

    def revenue_changed(self):
        self.revenue_to_render = None

    @property
    def label(self):
        return self.labels[-1] if self.labels else None

    @label.setter
    def label(self, label_name):
        self.labels.clear()
        if label_name:
            self.labels.append(Label(label_name))  # Assuming Label is a defined class or part

    def restore_borders(self, edges=None):
        if edges is None:
            edges = self.ALL_EDGES

        # Re-add borders that are in the edge list returning those that are missing
        missing = []
        for edge in edges:
            original = next((e for e in self.original_borders if e.edge == edge), None)
            if original is None or original in self.borders:
                continue

            self.borders.append(original)
            missing.append(edge)

        for edge in missing:
            neighbor = self.hex.neighbors.get(edge, None)
            if neighbor is not None:
                neighbor_tile = neighbor.tile
                if neighbor_tile is not None:
                    neighbor_tile.restore_borders([self.hex.invert(edge)])

    def reframe(self, color1, color2=None):
        self.frame = Frame(color1, color2) if color1 else None  # Assuming Frame is a defined class

    def restripe(self, color):
        self.stripes = Stripes(color) if color else None  # Assuming Stripes is a defined class

    def available_slot(self):
        return any(city.available_slots > 0 for city in self.cities)

    def hide(self):
        self.hidden = True

    def separate_parts(self):
        for part in self.parts:
            self.blocks_lay = part.blocks_lay() if hasattr(part, "blocks_lay") else False

            if part.is_city():
                self.cities.append(part)
                self.city_towns.append(part)
            elif part.is_label():
                self.labels.append(part)
            elif part.is_path():
                if part.track == "future":
                    self.future_paths.append(part)
                else:
                    self._paths.append(part)
            elif part.is_town():
                self.towns.append(part)
                self.city_towns.append(part)
                if part.is_halt():
                    self.halts.append(part)
            elif part.is_upgrade():
                self.upgrades.append(part)
            elif part.is_offboard():
                self.offboards.append(part)
            elif part.is_border():
                self.original_borders.append(part)
                self.borders.append(part)
            elif part.is_junction():
                self.junction = part
            elif part.is_icon():
                self.icons.append(part)
            elif part.is_stub():
                self.stubs.append(part)
            elif part.is_partition():
                self.partitions.append(part)
            elif part.is_frame():
                self.frame = part
            elif part.is_stripes():
                self.stripes = part
            elif part.is_future_label():
                self.future_label = part
            else:
                print(part)
                print(dir(part))
                print(part.__str__())
                raise Exception(f"Part {part} not separated.")

        for idx, part in enumerate(self.parts):
            part.index = idx
            part.tile = self

        self.nodes = list(set([node for path in self._paths for node in path.nodes]))
        self.stops = list(set([stop for path in self._paths for stop in path.stops]))
        self.edges = list(set([edge for path in self._paths for edge in path.edges]))

        self.revenue_stops = list(set(self.stops + self.offboards))

        for path in self._paths:
            for edge in path.edges:
                edge.tile = self


import re


class Hex:
    DIRECTIONS = {
        "flat": {
            (0, 2): 0,
            (-1, 1): 1,
            (-1, -1): 2,
            (0, -2): 3,
            (1, -1): 4,
            (1, 1): 5,
        },
        "pointy": {
            (-1, 1): 0,
            (-2, 0): 1,
            (-1, -1): 2,
            (1, -1): 3,
            (2, 0): 4,
            (1, 1): 5,
        },
    }

    LETTERS = [chr(x) for x in range(ord("A"), ord("Z") + 1)] + [f"A{chr(x)}" for x in range(ord("A"), ord("Z") + 1)]
    NEGATIVE_LETTERS = [0] + [chr(x) for x in range(ord("a"), ord("z") + 1)]

    COORD_LETTER = re.compile("([A-Za-z]+)")
    COORD_NUMBER = re.compile("(-?[0-9]+)")

    @staticmethod
    def invert(dir):
        return (dir + 3) % 6

    @staticmethod
    def init_x_y(coordinates, axes_config=None):
        if axes_config is None:
            axes_config = {"x": "letter", "y": "number"}

        letter_match = Hex.COORD_LETTER.search(coordinates)
        number_match = Hex.COORD_NUMBER.search(coordinates)
        letter = letter_match.group(1) if letter_match else None
        number = int(number_match.group(1)) if number_match else None

        if axes_config["x"] == "letter":
            x = Hex.LETTERS.index(letter) if letter in Hex.LETTERS else -Hex.NEGATIVE_LETTERS.index(letter)
        else:
            x = number - 1

        if axes_config["y"] == "letter":
            y = Hex.LETTERS.index(letter) if letter in Hex.LETTERS else -Hex.NEGATIVE_LETTERS.index(letter)
        else:
            y = number - 1

        column, row = (letter, number) if axes_config["x"] == "letter" else (number, letter)

        return x, y, column, row

    def __init__(
        self,
        coordinates,
        layout=None,
        axes=None,
        tile=None,
        location_name=None,
        hide_location_name=False,
        empty=False,
    ):
        self.coordinates = coordinates
        self.layout = layout
        self.axes = axes
        self.x, self.y, self.column, self.row = Hex.init_x_y(self.coordinates, axes)
        self.neighbors = {}
        self.all_neighbors = {}
        self.location_name = location_name
        self.hide_location_name = hide_location_name
        self._tile = tile
        self._paths = None
        self.original_tile = tile
        self._tile.hex = self
        self.empty = empty
        self.ignore_for_axes = False
        self.tokens = []

    @property
    def id(self):
        return self.coordinates

    def __lt__(self, other):
        return self.coordinates < other.coordinates

    def __eq__(self, other):
        return self.coordinates == other.coordinates

    def __hash__(self):
        return hash((self.coordinates))

    @property
    def name(self):
        return self.coordinates

    @property
    def full_name(self):
        return f"{self.name} ({self.location_name})"

    @property
    def tile(self):
        """The getter for the tile."""
        return self._tile

    @tile.setter
    def tile(self, new_tile):
        self.original_tile = self._tile = new_tile
        new_tile.hex = self

    def lay(self, tile):
        """Lays a tile on this hex, handling city reservations, tokens, and icons transfer."""
        city_map = self.city_map_for(tile)

        # Handle reservations transfer
        for old_city, new_city in city_map.items():
            if new_city:
                for entity in filter(None, old_city.reservations):
                    for ability in entity.all_abilities:
                        if ability.type == "reservation" and ability.hex == self.coordinates:
                            ability.tile = new_city.tile
                            ability.city = new_city.tile.cities.index(new_city)
                new_city.reservations.extend(old_city.reservations)
                new_city.groups = old_city.groups
                old_city.reservations.clear()

        self._tile.hex = None
        tile.hex = self

        # Handle tokens transfer
        for old_city, new_city in city_map.items():
            for index, token in enumerate(old_city.tokens):
                # set_trace()
                cheater = index and index >= old_city.normal_slots()
                if token:
                    new_city.exchange_token(token, cheater=cheater)
            for token in old_city.extra_tokens:
                new_city.exchange_token(token, extra_slot=True)
            old_city.reset()

        # Transfer sticky icons
        new_icons = {}
        for icon in tile.icons:
            new_icons.setdefault(icon.name, []).append(icon)

        for icon in self._tile.icons:
            if icon.sticky and not new_icons[icon.name]:
                new_icon = icon.copy()
                new_icon.preprinted = False
                tile.icons.append(new_icon)

        self._tile.icons = [icon for icon in self._tile.icons if icon.preprinted]

        # Future label handling
        if tile.future_label:
            tile.future_label.sticker = tile.future_label
        if self._tile.future_label:
            if self._tile.future_label.color != tile.color:
                tile.future_label = self._tile.future_label
            self._tile.future_label = self._tile.future_label.sticker

        # Transfer other properties
        tile.reservations = self._tile.reservations
        self._tile.reservations = []

        tile.borders.extend(self._tile.borders)
        self._tile.borders.clear()

        tile.partitions.extend(self._tile.partitions)
        self._tile.partitions.clear()

        tile.location_name = self.location_name
        self._tile.location_name = None

        self._tile = tile
        self._paths = None

    def lay_downgrade(self, tile):
        """Specific method to lay a tile that represents a downgrade."""
        self.lay(tile)
        tile.restore_borders()

    def paths(self):
        """Compute and cache paths for this hex."""
        if not self._paths:
            self._paths = {}
            for path in self._tile.paths:
                for exit in path.exits:
                    self._paths.setdefault(exit, []).append(path)
        return self._paths

    def neighbor_direction(self, other):
        """Calculate direction to a neighboring hex."""
        return self.DIRECTIONS[self.layout][(other.x - self.x, other.y - self.y)]

    def targeting(self, other):
        """Determine if the current hex targets a neighbor based on exits."""
        dir = self.neighbor_direction(other)
        return dir in self._tile.exits

    def invert(self, dir):
        """Invert direction."""
        return (dir + 3) % 6

    def distance(self, other):
        """Calculate the 'as-the-crow-flies' distance to another hex."""
        dx = abs(other.x - self.x)
        dy = abs(other.y - self.y)
        if self.layout == "pointy":
            return dy + max(0, (dx - dy) // 2)
        else:
            return dx + max(0, (dy - dx) // 2)

    def place_token(self, token, logo=None, blocks_lay=None, preprinted=True, loc=None):
        """Place a token on the hex, adding an icon for it."""
        token.place(self)
        self.tokens.append(token)
        icon = Icon("", token.corporation.id, True, blocks_lay, preprinted, loc=loc)
        icon.image = logo or token.corporation.logo
        self._tile.icons.append(icon)

    def remove_token(self, token):
        """Remove a token from the hex."""
        self._tile.icons = [icon for icon in self._tile.icons if icon.name != token.corporation.id]
        self.tokens.remove(token)

    def city_map_for(self, tile):
        """Map cities on the current tile to cities on a new tile based on connectivity."""
        if not any(city.exits for city in self._tile.cities) and len(self._tile.cities) == len(tile.cities):
            city_map = dict(zip(self._tile.cities, tile.cities))
        else:
            city_map = {
                old_city: next(
                    (
                        new_city
                        for new_city in tile.cities
                        if not old_city.exits or set(old_city.exits).issubset(new_city.exits)
                    ),
                    None,
                )
                for old_city in self._tile.cities
            }

        new_cities = [city for city in city_map.values() if city]
        for index, old_city in enumerate(self._tile.cities):
            if old_city not in city_map or city_map[old_city] is None:
                new_city = (
                    tile.cities[index]
                    if index < len(tile.cities)
                    else next((city for city in tile.cities if city not in new_cities), None)
                )
                city_map[old_city] = new_city
                if new_city:
                    new_cities.append(new_city)

        return city_map

    def __repr__(self):
        """Return a string representation of the hex."""
        return f"<Hex: {self.name}, tile: {self._tile.name}>"
