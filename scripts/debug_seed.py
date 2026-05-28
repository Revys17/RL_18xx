"""Debug a single seed and dump the failing action context."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.gamemap import GameMap
from rl18xx.rust_adapter import RustGameAdapter
from tests.validate_rust_engine import compare_state


def run(seed: int, max_actions: int = 2000, stop_at: int = -1, dump_legal: bool = False):
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    game_cls = GameMap().game_by_title("1830")
    py_game = game_cls(names)
    rust_game = RustGame(names)
    rust_adapter = RustGameAdapter(rust_game)
    mapper = ActionMapper()

    action_count = 0
    while not py_game.finished and action_count < max_actions:
        indices, price_ranges, _action_types = mapper.get_legal_actions_factored(rust_adapter)
        if not indices:
            break

        idx = rng.choice(indices)
        pr = price_ranges.get(idx)
        if pr is not None:
            lo, hi = pr
            price = rng.randint(lo, hi)
            action = mapper.map_index_to_action_with_price(idx, rust_adapter, price)
        else:
            action = mapper.map_index_to_action(idx, rust_adapter)

        action_dict = action.to_dict()

        if stop_at >= 0 and action_count == stop_at:
            print(f"=== At action #{action_count}, idx={idx} ===")
            print(f"  Action: {action_dict}")
            if dump_legal:
                # Print action regardless, then deep dive if sell_shares
                pass
            if dump_legal and action_dict.get("type") == "sell_shares":
                print(f"  Total legal idxs: {len(indices)}")
                step = py_game.active_step()
                entity = py_game.current_entity
                shares = action_dict.get("shares", [])
                percent = action_dict.get("percent")
                # Look up share
                share_objs = [py_game.share_by_id(sid) for sid in shares]
                print(f"  Share objs: {share_objs}")
                for s in share_objs:
                    if s:
                        print(f"    share id={s.id} owner={s.owner} pres={s.president} pct={s.percent} corp={s.corporation.id}")
                # Construct bundle and check can_sell
                from rl18xx.game.engine.entities import ShareBundle
                if share_objs and all(s is not None for s in share_objs):
                    bundle = ShareBundle(share_objs, percent)
                    print(f"  Bundle pct={bundle.percent} num_shares={bundle.num_shares()} corp={bundle.corporation.id}")
                    print(f"  can_sell? {step.can_sell(entity, bundle)}")
                    print(f"  fit_in_bank? {py_game.share_pool.fit_in_bank(bundle)}")
                    print(f"  can_dump? {step.can_dump(entity, bundle)}")
                    print(f"  can_sell_order? {step.can_sell_order()}")
                    print(f"  check_sale_timing? {py_game.check_sale_timing(entity, bundle)}")
                    print(f"  players_sold: {py_game.round.players_sold.get(entity, {})}")
                    # Look at sellable_shares
                    ss = step.sellable_shares(entity)
                    print(f"  Python sellable_shares count: {len(ss)}")
                    for b in ss:
                        print(f"    bundle: corp={b.corporation.id} pct={b.percent} num_shares={b.num_shares()}")
                raise SystemExit(0)
            if dump_legal:
                print(f"  Total legal idxs: {len(indices)}")
                # Dump Rust factored choices for context
                factored = rust_adapter.get_factored_choices()
                print(f"  Rust factored count: {len(factored)}")
                # Show ones for the relevant entity / hex
                hex_ = action_dict.get("hex")
                for fc in factored:
                    if fc.type == "LayTile" and fc.params.get("hex") == hex_:
                        print(f"    type={fc.type} entity={fc.entity} params={fc.params}")
                # also: show old tile state
                hex_obj = py_game.hex_by_id(hex_) if hex_ else None
                if hex_obj is None:
                    print(f"  (no hex in action)")
                    # try city resolution
                    city_id = action_dict.get("city")
                    if city_id:
                        print(f"  city id: {city_id}")
                        city_obj = py_game.city_by_id(city_id)
                        print(f"  Python city_by_id -> {city_obj}")
                    step = py_game.active_step()
                    entity = py_game.current_entity
                    print(f"  Step: {step}, entity: {entity}")
                    raise SystemExit(0)
                print(f"  Old tile: {hex_obj.tile.name} rot={hex_obj.tile.rotation}")
                print(f"  Old tile paths: {[(p.a, p.b) for p in hex_obj.tile.paths]}")
                print(f"  Old tile cities: {hex_obj.tile.cities}")
                print(f"  Old tile city paths: {[(c.exits) for c in hex_obj.tile.cities]}")
                print(f"  hex.neighbors: {list(hex_obj.neighbors.keys())}")
                # Check legal_tile_rotations
                step = py_game.active_step()
                entity = py_game.current_entity
                tile_obj = py_game.tile_by_id(action_dict['tile'])
                rot_legal = step.legal_tile_rotations(entity, hex_obj, tile_obj)
                print(f"  Python legal rotations for tile {action_dict['tile']}: {rot_legal}")
            print(f"  Active step: {py_game.active_step()}")
            print(f"  Current entity: {py_game.current_entity}")
            # Try the action
            try:
                py_game.process_action(action_dict)
                print("  Python OK")
            except Exception as e:
                print(f"  Python ERROR: {e}")
            return

        try:
            py_game = py_game.process_action(action_dict)
        except Exception as e:
            print(f"#{action_count} Python error: {e}")
            print(f"  action: {action_dict}")
            print(f"  step: {py_game.active_step()}")
            print(f"  entity: {py_game.current_entity}")
            return

        try:
            rust_game.process_action(action_dict)
        except Exception as e:
            print(f"#{action_count} Rust error: {e}")
            print(f"  action: {action_dict}")
            return

        mismatches = compare_state(rust_game, py_game)
        if mismatches:
            print(f"#{action_count} mismatch:")
            for m in mismatches[:10]:
                print(f"   {m}")
            return

        action_count += 1

    print(f"Finished at action {action_count}, finished={py_game.finished}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--stop-at", type=int, default=-1)
    parser.add_argument("--dump-legal", action="store_true")
    args = parser.parse_args()
    run(args.seed, max_actions=args.max_actions, stop_at=args.stop_at, dump_legal=args.dump_legal)
