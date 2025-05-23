__all__ = [
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
    "Graph",
    "DistanceGraph",
]


from .core import GameError, Ownable


class BasePart:
    def __init__(self, tile=None, index=None, loc=None, **kwargs):
        self.tile = tile
        self.index = index
        self.loc = loc
        self.id = f"{type_s(tile)}-{index}"

    @property
    def signature(self):
        return (
            f"{self.hex.id}-{self.index}"
            if self.hex and self.index is not None
            else None
        )

    @property
    def hex(self):
        return self.tile.hex if self.tile else None

    def __le__(self, other):
        return isinstance(self, other.__class__)

    def __lt__(self, other):
        if self.edge and other.edge:
            return self.num - other.num
        elif self.edge:
            return -1
        elif other.edge:
            return 1
        else:
            return 0

    def __eq__(self, other):
        return self.num == other.num  # Implement equality comparison if neede

    def rotate(self, ticks):
        return self

    def blocks(self, corporation):
        return False

    # Define other methods similarly...

    def inspect(self):
        return (
            f"<{self.__class__.__name__}: hex: {self.hex.name if self.hex else None}>"
        )


class Node(BasePart):
    def __init__(self, lanes=None, paths=None, exits=None, **kwargs):
        super().__init__(**kwargs)
        self.paths = None
        self.exits = None
        self.lanes = None

    def clear(self):
        self.paths = None
        self.exits = None

    def solo(self):
        return len(self.tile.nodes) == 1

    def get_paths(self):
        if self.paths is None:
            self.paths = [p for p in self.tile.paths if self in p.nodes]
        return self.paths

    def get_exits(self):
        if self.exits is None:
            self.exits = [exit for path in self.get_paths() for exit in path.exits]
        return self.exits

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
    ):
        if visited is None:
            visited = set()
        if visited_paths is None:
            visited_paths = set()
        if skip_paths is None:
            skip_paths = set()
        if counter is None:
            counter = defaultdict(int)

        if self in visited:
            return
        visited.add(self)

        for node_path in self.get_paths():
            if (
                node_path.track == skip_track
                or node_path in skip_paths
                or node_path.ignore
            ):
                continue

            for path in node_path.walk(
                visited=visited_paths,
                skip_paths=skip_paths,
                skip_track=skip_track,
                counter=counter,
                converging=converging_path,
            ):
                yield path

                if not path.terminal:
                    for next_node in path.nodes:
                        if next_node == self or (
                            corporation and next_node.blocks(corporation)
                        ):
                            continue

                        yield from next_node.walk(
                            visited=visited,
                            counter=counter,
                            corporation=corporation,
                            visited_paths=visited_paths,
                            skip_track=skip_track,
                            skip_paths=skip_paths,
                            converging_path=converging_path or path.converging,
                        )

        if converging_path:
            visited.remove(self)


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
                b_lanes = [lanes, lanes - index - 1] if a.edge and b.edge else a_lanes
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
        self.terminal = terminal
        self.lanes = lanes if lanes else self.LANES
        self.edges = []
        self.stops = []
        self.nodes = []
        self.exit_lanes = {}
        self.track = track
        self.ignore = ignore
        self.ignore_gauge_walk = ignore_gauge_walk
        self.ignore_gauge_compare = ignore_gauge_compare

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

    def __le__(self, other):
        other_ends = other.ends()
        return all(t <= o for t in self.ends() for o in other_ends) and (
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
        if not hasattr(self, "_ends"):
            self._ends = []
            for part in [self.a, self.b]:
                if part.junction:
                    for path in part.paths:
                        if path != self:
                            self._ends.extend(
                                [p for p in [path.a, path.b] if not p.junction]
                            )
                else:
                    self._ends.append(part)
        return self._ends

    def connects_to(self, other, corporation):
        for part in [self.a, self.b]:
            if part.edge:
                edge = part.num
                neighbor = self.hex.neighbors.get(edge)
                np_edge = self.hex.invert(edge)
                if neighbor == other.hex and any(e.num == np_edge for e in other.edges):
                    return True
            elif part.junction:
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
        return [larger_width, new_index][self.LANE_INDEX] == self.lane_invert(lanes_b)[
            self.LANE_INDEX
        ]

    @property
    def path(self):
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
        return (
            self.lanes[0][self.LANE_WIDTH] == 1 and self.lanes[1][self.LANE_WIDTH] == 1
        )

    @property
    def exits(self):
        if not hasattr(self, "_exits"):
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

    def walk(
        self,
        skip=None,
        jskip=None,
        visited=None,
        skip_paths=None,
        counter=None,
        skip_track=None,
        converging=True,
    ):
        if visited is None:
            visited = set()
        if skip_paths is None:
            skip_paths = set()
        if counter is None:
            counter = defaultdict(int)

        if self in visited or self in skip_paths:
            return
        if self.junction and counter[self.junction] > 1:
            return
        if any(counter[edge.id] for edge in self.edges):
            return
        if self.track == skip_track:
            return
        if self.junction and self.terminal:
            return

        visited.add(self)
        if self.junction:
            counter[self.junction] += 1

        yield self, visited, counter, converging

        if self.junction and self.junction != jskip:
            for jp in self.junction.paths:
                yield from jp.walk(
                    jskip=self.junction,
                    visited=visited,
                    skip_paths=skip_paths,
                    counter=counter,
                    converging=converging,
                )

        for edge in self.edges:
            edge_id = edge.id
            edge_num = edge.num
            if edge_num == skip:
                continue
            neighbor = self.hex.neighbors.get(edge_num)
            if not neighbor:
                continue

            counter[edge_id] += 1
            np_edge = self.hex.invert(edge_num)

            for np in neighbor.paths[np_edge]:
                if not self.lane_match(
                    self.exit_lanes[edge_num], np.exit_lanes[np_edge]
                ):
                    continue
                if not self.ignore_gauge_walk and not self.tracks_match(
                    np, dual_ok=True
                ):
                    continue

                yield from np.walk(
                    skip=np_edge,
                    visited=visited,
                    skip_paths=skip_paths,
                    counter=counter,
                    skip_track=skip_track,
                    converging=converging or self.hex.converging_exit(edge_num),
                )

            counter[edge_id] -= 1

        if converging:
            visited.remove(self)
        if self.junction:
            counter[self.junction] -= 1


class Edge(BasePart):
    def __init__(self, num, lanes=None):
        super().__init__()
        self.num = int(num)
        self.lanes = lanes if lanes is not None else [None, None]
        self._id = f"{self.hex.id}_{self.num}_{self.lanes[1]}"

    def __le__(self, other):
        return isinstance(other, Edge) and (self.num == other.num)

    def edge(self):
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

    def frame(self):
        return True


class FutureLabel(BasePart):
    def __init__(self, label=None, color=None):
        super().__init__()
        self.label = label
        self.color = color
        self.sticker = None

    def future_label(self):
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

    def icon(self):
        return True


class Junction(BasePart):
    def __init__(self, lanes=[], **kwargs):
        super().__init__(**kwargs)
        self.lanes = lanes

    def junction(self):
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

    def label(self):
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
        self.outer = (
            []
            if restrict == "inner"
            else list(set(range(6)) - set(range(self.a, self.b)))
        )

    def add_blocker(self, private_company):
        self.blockers.append(private_company)

    def partition(self):
        return True


class Stripes(BasePart):
    def __init__(self, color, **kwargs):
        super().__init__(**kwargs)
        self.color = color

    def stripes(self):
        return True


class Stub(BasePart):
    def __init__(self, edge, track="broad", **kwargs):
        super().__init__(**kwargs)
        self.edge = edge
        self.track = track

    def stub(self):
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

    def upgrade(self):
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
            revenue_dict = {
                color: int(r) for color, r in (part.split("_") for part in parts)
            }
            self.revenue_to_render = {
                phase: (format % rev if format else rev)
                for phase, rev in revenue_dict.items()
            }
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
        base_multiplier = getattr(train, "multiplier", 1)

        if isinstance(distance, (int, float)):
            return base_multiplier

        for h in distance:
            if self.type in h["nodes"]:
                return h.get("multiplier", base_multiplier)

        return base_multiplier

    def uniq_revenues(self):
        return list(set(self.revenue.values()))


class Border(BasePart):
    def __init__(self, edge, type=None, cost=None, color=None):
        self.edge = int(edge)
        self.type = (
            type and type.lower()
        )  # Converting to lowercase for symbol-like behavior
        self.cost = int(cost) if cost is not None else None
        self.color = (
            color and color.lower()
        )  # Converting to lowercase for symbol-like behavior

    def is_border(self):
        return True


class City(RevenueCenter):
    def __init__(self, revenue, **opts):
        super().__init__(revenue, **opts)
        self.slots = opts.get("slots", 1)
        self.tokens = [None] * self.slots
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
        return any(
            t is not None and t.corporation == corporation
            for t in self.tokens + self.extra_tokens
        )

    def find_reservation(self, corporation):
        for index, reservation in enumerate(self.reservations):
            if reservation and (
                reservation == corporation or reservation.owner == corporation
            ):
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

    def city(self):
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
        if tokens is None:
            tokens = corporation.tokens_by_type
        tokens = list(tokens)
        self.error = "generic"
        if not extra_slot and not tokens:
            self.error = "no_tokens"
            return False

        return any(
            self._is_tokenable(
                t, corporation, free, cheater, extra_slot, spender, same_hex_allowed
            )
            for t in tokens
        )

    def _is_tokenable(
        self, token, corporation, free, cheater, extra_slot, spender, same_hex_allowed
    ):
        if not extra_slot and not self.get_slot(token.corporation, cheater):
            self.error = "no_slots"
            return False
        if not free and token.price > (spender or corporation).cash:
            self.error = "no_money"
            return False
        if not same_hex_allowed and any(
            c.tokened_by(token.corporation) for c in self.tile.cities
        ):
            self.error = "existing_token"
            return False
        if self.reserved_by(corporation):
            return True
        if self.tile.token_blocked_by_reservation(corporation) and not cheater:
            self.error = "blocked_reservation"
            return False
        return True

    def get_slot(self, corporation, cheater=False):
        reservation = self.find_reservation(corporation)
        open_slot = next(
            (
                i
                for i, t in enumerate(self.tokens)
                if t is None and self.reservations[i] is None
            ),
            None,
        )
        return open_slot if cheater or reservation is None else reservation

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
        if check_tokenable and not self.tokenable(
            corporation, free, [token], cheater, extra_slot, spender, same_hex_allowed
        ):
            self._raise_token_error()
        self.exchange_token(token, cheater, extra_slot)
        self.tile.reservations.remove(corporation)
        self.remove_reservation(corporation)

    def exchange_token(self, token, cheater=False, extra_slot=False):
        token.place(self, extra=extra_slot, cheater=cheater)
        if extra_slot:
            self.extra_tokens.append(token)
            return
        slot = self.get_slot(token.corporation, cheater)
        self.tokens[slot] = token

    def _raise_token_error(self):
        error_messages = {
            "no_tokens": "cannot lay token - has no tokens left",
            "existing_token": "cannot lay token - already has a token",
            "blocked_reservation": "cannot lay token - remaining token slots are reserved",
            "no_money": "cannot lay token - cannot afford token",
            "no_slots": "cannot lay token - no token slots available",
        }
        error_msg = error_messages.get(self.error, "cannot lay token")
        raise GameError(
            f"{corporation.name} {error_msg} on {self.tile.hex.id if self.tile.hex else 'N/A'}"
        )


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

    def town(self):
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
        if other.town():
            return True
        return super().__le__(other)

    def halt(self):
        return True


class Offboard(RevenueCenter):
    def blocks(self, corporation):
        return True

    def offboard(self):
        return True


class Pass(City):
    def __init__(self, revenue, **opts):
        super().__init__(revenue, **opts)
        self.color = opts.get("color", "gray").lower()
        self.size = int(opts.get("size", 1))
        self.route = opts.get("route", "never").lower()

    def _pass(self):
        return True


class Graph:
    def __init__(self, game, **opts):
        self.game = game
        self.connected_hexes = {}
        self.connected_nodes = {}
        self.connected_paths = {}
        self.connected_hexes_by_token = {}
        self.connected_paths_by_token = {}
        self.connected_nodes_by_token = {}
        self.reachable_hexes = {}
        self.tokenable_cities = {}
        self.routes = {}
        self.tokens = {}
        self.cheater_tokens = {}
        self.home_as_token = opts.get("home_as_token", False)
        self.no_blocking = opts.get("no_blocking", False)
        self.skip_track = opts.get("skip_track")
        self.check_tokens = opts.get("check_tokens")
        self.check_regions = opts.get("check_regions")

    def clear(self):
        self.connected_hexes.clear()
        self.connected_nodes.clear()
        self.connected_paths.clear()
        self.connected_hexes_by_token.clear()
        self.connected_nodes_by_token.clear()
        self.connected_paths_by_token.clear()
        self.reachable_hexes.clear()
        self.tokenable_cities.clear()
        self.tokens.clear()
        self.cheater_tokens.clear()
        to_delete = [
            key
            for key, route in self.routes.items()
            if not route.get("route_train_purchase")
        ]
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
            list(self.compute(corporation, routes_only=True))  # Consume the generator
        return self.routes.get(corporation)

    def can_token(
        self, corporation, cheater=False, same_hex_allowed=False, tokens=None
    ):
        if tokens is None:
            tokens = corporation.tokens_by_type()
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
        if corporation in self.tokenable_cities:
            return self.tokenable_cities[corporation]

        cities = []
        for node in self.compute(corporation):
            if node.tokenable(corporation, free=True):
                cities.append(node)

        if cities:
            self.tokenable_cities[corporation] = cities
        return cities

    def connected_hexes(self, corporation):
        if corporation not in self.connected_hexes:
            list(
                self.compute(corporation)
            )  # Consume the generator to populate the data
        return self.connected_hexes[corporation]

    def connected_nodes(self, corporation):
        if corporation not in self.connected_nodes:
            list(self.compute(corporation))
        return self.connected_nodes[corporation]

    def connected_paths(self, corporation):
        if corporation not in self.connected_paths:
            list(self.compute(corporation))
        return self.connected_paths[corporation]

    def connected_hexes_by_token(self, corporation, token):
        if token not in self.connected_hexes_by_token[corporation]:
            self.compute_by_token(corporation)
        return self.connected_hexes_by_token[corporation][token]

    def connected_nodes_by_token(self, corporation, token):
        if token not in self.connected_nodes_by_token[corporation]:
            self.compute_by_token(corporation)
        return self.connected_nodes_by_token[corporation][token]

    def connected_paths_by_token(self, corporation, token):
        if token not in self.connected_paths_by_token[corporation]:
            self.compute_by_token(corporation)
        return self.connected_paths_by_token[corporation][token]

    def reachable_hexes(self, corporation):
        if corporation not in self.reachable_hexes:
            list(self.compute(corporation))
        return self.reachable_hexes[corporation]

    def compute_by_token(self, corporation):
        # Assuming compute also updates the connected_*_by_token dictionaries
        list(self.compute(corporation))  # Consume the generator for the side effects
        for hex in self.game.hexes():
            for city in hex.tile.cities:
                if self.game.city_tokened_by(city, corporation) and not (
                    self.check_tokens and self.game.skip_token(self, corporation, city)
                ):
                    list(self.compute(corporation, one_token=city))

    def home_hexes(self, corporation):
        home_hexes = {}
        hexes = [
            self.game.hex_by_id(h) for h in corporation.coordinates
        ]  # Assuming corporation.coordinates returns a list of IDs
        for hex in hexes:
            for edge, _ in hex.neighbors.items():
                home_hexes.setdefault(hex, {})[edge] = True
        return home_hexes

    def home_hex_nodes(self, corporation):
        nodes = {}
        # Assuming corporation.coordinates returns a list of hex ID strings
        hexes = [self.game.hex_by_id(h) for h in corporation.coordinates]
        for hex in hexes:
            if corporation.city is not None:
                # If corporation.city is a single value, this makes it a list
                city_indices = (
                    [corporation.city]
                    if not isinstance(corporation.city, list)
                    else corporation.city
                )
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
        hexes = {}
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
            for ability, owner in self.game.abilities(corporation, ability_type):
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
            self.game.graph_border_paths(corporation)
            if self.check_regions
            else self.game.graph_skip_paths(corporation)
        )

        for node in tokens:
            if routes.get("route_train_purchase") and routes_only:
                return None

            visited = {k: v for k, v in tokens.items() if k != node}
            local_nodes = {}

            # Assuming node.walk is a generator
            for path in node.walk(
                visited=visited,
                corporation=walk_corporation,
                skip_track=self.skip_track,
                skip_paths=skip_paths,
                converging_path=False,
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

            mandatory_nodes = sum(
                1 for p_node in local_nodes if p_node.route == "mandatory"
            )
            optional_nodes = sum(
                1 for p_node in local_nodes if p_node.route == "optional"
            )

            if mandatory_nodes > 1:
                routes["route_available"] = True
                routes["route_train_purchase"] = True
            elif mandatory_nodes == 1 and optional_nodes > 0:
                routes["route_available"] = True

        if one_token:
            self.connected_hexes_by_token[corporation][one_token] = hexes
            self.connected_nodes_by_token[corporation][one_token] = nodes
            self.connected_paths_by_token[corporation][one_token] = paths
        else:
            self.routes[corporation] = routes
            self.connected_hexes[corporation] = hexes
            self.connected_nodes[corporation] = nodes
            self.connected_paths[corporation] = paths
            self.reachable_hexes[corporation] = {path.hex: True for path in paths}


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
            counter = defaultdict(int)

        self.merge_distance(node_distances, node, distance)
        if corporation and node.blocks(corporation):
            return

        count = 1 if node.visit_cost > 0 else 0
        distance_key = "node"
        if self.separate_node_types:
            if node.city():
                distance_key = "city"
            elif node.town() and not node.halt():
                distance_key = "town"
        distance[distance_key] += count

        for node_path in node.get_paths():
            for path, _, ct in node_path.walk(counter=counter):
                self.merge_distance(path_distances, path, distance)
                yield path, distance

                if not path.terminal:
                    for next_node in path.nodes:
                        if next_node == node:
                            continue
                        a_or_b = "a" if path.a == next_node else "b"
                        next_distance = {
                            **distance
                        }  # Copy to avoid mutating the original distance

                        if a_or_b == "a":
                            if not self.smaller_or_equal_distance(
                                a_distances.get(path, {}), next_distance
                            ):
                                self.merge_distance(a_distances, path, next_distance)
                            else:
                                continue
                        else:
                            if not self.smaller_or_equal_distance(
                                b_distances.get(path, {}), next_distance
                            ):
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
            start_distance = (
                {"city": 0, "town": 0} if self.separate_node_types else {"node": 0}
            )
            for path, dist in self.node_walk(
                node, start_distance, n_distances, p_distances, {}, {}, corporation
            ):
                self.merge_distance(h_distances, path.hex, dist)

        self.node_distances[corporation] = n_distances
        self.path_distances[corporation] = p_distances
        self.hex_distances[corporation] = h_distances
