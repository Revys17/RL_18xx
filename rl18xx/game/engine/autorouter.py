__all__ = ["AutoRouter"]


from rl18xx.game.engine.core import (
    RouteTooLong,
    ReusesCity,
    NoToken,
    RouteTooShort,
    GameError,
)
from .graph import Route
import time, itertools


class AutoRouter:
    def __init__(self, game, flash=None, debug=False, verbose=False):
        self.game = game
        self.next_hexside_bit = 0
        self.flash = flash
        self.debug = debug
        self.verbose = verbose

    def compute(self, corporation, **opts):
        # if self.debug:
        #    set_trace()
        static = opts.get("routes", [])
        path_timeout = opts.get("path_timeout", 30)
        route_timeout = opts.get("route_timeout", 10)
        route_limit = opts.get("route_limit", 10000)

        connections = {}
        trains = sorted(self.game.route_trains(corporation), key=lambda x: x.price, reverse=True)

        graph = self.game.graph_for_entity(corporation)
        nodes = sorted(
            graph.connected_nodes(corporation).keys(),
            key=lambda node: (
                0 if node.tokened_by(corporation) else 1,
                0 if node.is_offboard() else 1,
                -max(node.route_revenue(self.game.phase, train) for train in trains),
            ),
        )

        path_walk_timed_out = False
        now = time.time()

        skip_paths = {path: True for route in static for path in route.paths}
        skip_trains = {train for route in static for train in route.get("trains", [])}
        trains = [train for train in trains if train not in skip_trains]

        train_routes = {}
        hexside_bits = {}
        self.next_hexside_bit = 0

        for node in nodes:
            if time.time() - now > path_timeout:
                if self.verbose:
                    print("Path timeout reached")
                path_walk_timed_out = True
                break
            else:
                if self.verbose:
                    print(f"Path search: {nodes.index(node)} / {len(nodes)} - paths starting from {node.hex.name}")

            # debug = False
            # if self.debug and node.hex.name == "F2":
            #     set_trace()
            #     debug = True
            walk_corporation = None if graph.no_blocking else corporation
            # path_history = {}
            visited = {}
            visited_paths = {}
            counter = {}
            for _, vp, _ in node.walk(
                visited=visited,
                visited_paths=visited_paths,
                counter=counter,
                corporation=walk_corporation,
                skip_paths=skip_paths,
                # all_paths=True,
                # debug=debug
            ):
                self.process_path(vp, trains, connections, hexside_bits, train_routes, skip_paths)
                # for path in vp.keys():
                #    path_history[path] = True
            # if self.debug and node.hex.name == "F2":
            #    set_trace()
            #    path_history.keys()

        if self.verbose:
            print(
                f"Evaluated {len(connections)} paths, found {self.next_hexside_bit} unique hexsides, and found valid routes "
                f"{', '.join(f'{train.name}:{len(routes)}' for train, routes in train_routes.items())} in: {time.time() - now}"
            )
            print(train_routes)

        if self.debug:
            set_trace()
        for route in static:
            route.bitfield = self.bitfield_from_connection(route.connection_data, hexside_bits)
            train_routes[route.train] = [route]

        for train, routes in train_routes.items():
            train_routes[train] = sorted(routes, key=lambda x: x.revenue(), reverse=True)[:route_limit]

        sorted_routes = list(train_routes.values())

        limit = 1
        for routes in sorted_routes:
            limit *= len(routes)

        if self.verbose:
            print(
                f"Finding route combos of best {' '.join(f'{train.name}:{len(routes)}' for train, routes in train_routes.items())} "
                f"routes with depth {limit}"
            )

        now = time.time()
        possibilities = self.js_evaluate_combos(sorted_routes, route_timeout)

        if path_walk_timed_out:
            if self.flash:
                self.flash("Auto route path walk failed to complete (PATH TIMEOUT)")
        elif time.time() - now > route_timeout:
            if self.flash:
                self.flash("Auto route selection failed to complete (ROUTE TIMEOUT)")

        if self.debug:
            set_trace()
        max_routes = self.final_revenue_check(possibilities)

        # for route in max_routes:
        #    route.routes = max_routes

        return max_routes

    def process_path(self, vp, trains, connections, hexside_bits, train_routes, skip_paths):
        # if self.debug:
        #    set_trace()
        paths = vp.keys()
        chains = []
        chain = []
        left = right = last_left = last_right = None

        def complete():
            nonlocal left, right, chain, last_left, last_right
            chains.append({"nodes": [left, right], "paths": chain})
            last_left, last_right = left, right
            left = right = None
            chain = []

        def assign(a, b):
            nonlocal left, right
            if a and b:
                if a == last_left or b == last_right:
                    left, right = b, a
                else:
                    left, right = a, b
                complete()
            elif not left:
                left = a if a else b
            elif not right:
                right = a if a else b
                complete()

        for path in paths:
            chain.append(path)
            if not path.nodes:
                continue
            elif len(path.nodes) == 1:
                a, b = path.nodes[0], None
            else:
                a, b = path.nodes[0], path.nodes[1]

            assign(a, b)

        if chains or left:
            if not chains:
                chains.append({"nodes": [left, None], "paths": []})

            # if self.debug:
            #    set_trace()
            id = tuple(sorted(list(itertools.chain.from_iterable(c["paths"] for c in chains))))
            if id in connections:
                return

            connections[id] = [{"left": c["nodes"][0], "right": c["nodes"][1], "chain": c} for c in chains]
            connection = connections[id]

            path_abort = {train: True for train in trains}

            for train in trains:
                try:
                    route = Route(
                        self.game,
                        self.game.phase,
                        train,
                        connection_data=connection,
                        bitfield=self.bitfield_from_connection(connection, hexside_bits),
                    )
                    route.routes = [route]
                    # set_trace()
                    route.revenue(suppress_check_other=True)
                    train_routes.setdefault(train, []).append(route)
                except RouteTooLong:
                    path_abort[train] = False
                except ReusesCity:
                    path_abort.clear()
                except (GameError, NoToken, RouteTooShort) as e:
                    if self.verbose:
                        print(e)

                if not path_abort[train]:
                    return "abort"

    def bitfield_from_connection(self, connection, hexside_bits):
        bitfield = [0]

        def check_and_set(bitfield, hexside_left, hexside_right, hexside_bits):
            check_edge_and_set(bitfield, hexside_left, hexside_bits)
            check_edge_and_set(bitfield, hexside_right, hexside_bits)

        def check_edge_and_set(bitfield, hexside_edge, hexside_bits):
            if hexside_edge in hexside_bits:
                set_bit(bitfield, hexside_bits[hexside_edge])
            else:
                hexside_bits[hexside_edge] = self.next_hexside_bit
                set_bit(bitfield, self.next_hexside_bit)
                self.next_hexside_bit += 1

        def set_bit(bitfield, bit):
            entry = bit // 32
            mask = 1 << (bit % 32)
            while len(bitfield) <= entry:
                bitfield.append(0)
            bitfield[entry] |= mask

        for conn in connection:
            paths = conn["chain"]["paths"]
            if len(paths) == 1:
                # special case for tiny intra-tile path
                hexside_left = paths[0].nodes[0].id
                check_edge_and_set(bitfield, hexside_left, hexside_bits)
                if len(paths[0].nodes) > 1:
                    hexside_right = paths[0].nodes[1].id
                    check_edge_and_set(bitfield, hexside_right, hexside_bits)
            else:
                for index in range(len(paths) - 1):
                    node1, node2 = paths[index], paths[index + 1]
                    if len(node1.edges) == 1:
                        hexside_left = node1.edges[0].id
                        hexside_right = node2.edges[0].id
                        check_and_set(bitfield, hexside_left, hexside_right, hexside_bits)
                    elif len(node1.edges) == 2:
                        hexside_left = node1.edges[0].id
                        hexside_right = node1.edges[1].id
                        check_and_set(bitfield, hexside_left, hexside_right, hexside_bits)
                        hexside_left = hexside_right
                        hexside_right = node2.edges[0].id
                        check_and_set(bitfield, hexside_left, hexside_right, hexside_bits)
                    else:
                        if self.verbose:
                            print(
                                "ERROR: auto-router found unexpected number of path node edges. Route combos may be incorrect"
                            )
        return bitfield

    def js_evaluate_combos(self, sorted_routes, route_timeout):
        start_time = time.time()

        # Adjusted to include an option to select 'None' from each set of routes, akin to including the empty set
        sorted_routes_with_empty_option = [routes + [None] for routes in sorted_routes]

        def bitfield_conflicts(route_bitfields, test_bitfield):
            for bitfield in route_bitfields:
                if bitfield and test_bitfield and any(b & t for b, t in zip(bitfield, test_bitfield)):
                    return True
            return False

        def is_valid_combo(routes):
            # Exclude 'None' values before checking
            filtered_routes = [route for route in routes if route is not None]
            try:
                for route in filtered_routes:
                    route.check_other()
                return True
            except GameError:
                return False

        def generate_bitfields(routes):
            # Exclude 'None' values before generating bitfields
            return [route.bitfield for route in routes if route is not None]

        def evaluate_route_combos(routes_combinations):
            best_combos = []
            highest_revenue = 0
            for combo in routes_combinations:
                if time.time() - start_time > route_timeout:
                    if self.verbose:
                        print("Route timeout reached")
                    return best_combos, True

                route_bitfields = generate_bitfields(combo)
                if not any(
                    bitfield_conflicts(route_bitfields[:i], route_bitfields[i]) for i in range(1, len(route_bitfields))
                ):
                    if is_valid_combo(combo):
                        # Calculate revenue excluding 'None' values
                        combo_revenue = sum(route.revenue() for route in combo if route is not None)
                        if combo_revenue > highest_revenue:
                            best_combos = [combo]
                            highest_revenue = combo_revenue
                        elif combo_revenue == highest_revenue:
                            best_combos.append(combo)
            return best_combos, False

        # Adjusted to use 'sorted_routes_with_empty_option'
        possibilities, timeout_reached = evaluate_route_combos(itertools.product(*sorted_routes_with_empty_option))
        if timeout_reached and self.flash:
            self.flash("Route selection timed out.")

        if self.verbose:
            print(
                f"Found {len(possibilities)} best route combinations after evaluating with timeout of {route_timeout} seconds."
            )
        return possibilities

    def final_revenue_check(self, possibilities):
        def calculate_total_revenue(routes):
            filtered_routes = [route for route in routes if route is not None]
            try:
                for route in filtered_routes:
                    route.clear_cache(only_routes=True)  # Clear cache for accurate calculation
                    route.routes = routes  # Ensure route is aware of the full combination for context
                    route.revenue()  # Calculate revenue for this configuration
                return self.game.routes_revenue(
                    filtered_routes
                )  # Calculate and return total revenue for the combination
            except GameError as e:
                if self.verbose:
                    print(f"Sanity check error, likely an auto_router bug: {e}")
                return None  # Return None to signify an error was encountered

        # Evaluate each possibility to find the one with the maximum revenue
        # Filter out any possibilities that resulted in an error (None return value)
        # set_trace()
        valid_possibilities = filter(lambda x: calculate_total_revenue(x) is not None, possibilities)
        # Select the possibility with the highest total revenue, falling back to an empty list if none are valid
        max_routes = max(valid_possibilities, key=calculate_total_revenue, default=[])

        return max_routes
