"""Validate the Rust engine against the Python engine by replaying game actions.

Simple approach: replay raw actions through both engines, compare state after each.
No action filtering, no skip/pass insertion.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.game.gamemap import GameMap


def _rust_share_ownership(rust_game, corp_sym):
    """Build {owner_key: percent} for a Rust corporation."""
    rc = None
    for c in rust_game.corporations:
        if c.sym == corp_sym:
            rc = c
            break
    if not rc:
        return {}
    owners = {}
    for s in rc.shares:
        owners[s.owner] = owners.get(s.owner, 0) + s.percent
    return owners


def _python_share_ownership(py_game, corp_sym):
    """Build {owner_key: percent} for a Python corporation."""
    pc = py_game.corporation_by_id(corp_sym)
    if not pc:
        return {}
    owners = {}
    # Player shares
    for p in py_game.players:
        for s in p.shares:
            corp = s.corporation() if callable(s.corporation) else s.corporation
            if corp.id == corp_sym:
                key = f"player:{p.id}"
                owners[key] = owners.get(key, 0) + s.percent
    # Market shares
    for s in py_game.share_pool.shares:
        corp = s.corporation() if callable(s.corporation) else s.corporation
        if corp.id == corp_sym:
            owners["market"] = owners.get("market", 0) + s.percent
    # Remaining = IPO
    total_owned = sum(owners.values())
    ipo_pct = 100 - total_owned
    if ipo_pct > 0:
        owners[f"ipo:{corp_sym}"] = ipo_pct
    return owners


def compare_state(rust_game, py_game):
    errors = []

    # Game finished status
    if rust_game.finished != py_game.finished:
        errors.append(f"Finished: Rust={rust_game.finished} Python={py_game.finished}")
    # If both finished, skip round/step checks — stale state is expected
    if rust_game.finished and py_game.finished:
        return errors

    # ---- Round / turn structure ----

    # Round type
    r_round = rust_game.round.round_type
    p_round_obj = py_game.round
    p_round = "Operating" if p_round_obj.operating else ("Stock" if p_round_obj.stock else "Auction")
    if r_round != p_round:
        errors.append(f"Round type: Rust={r_round} Python={p_round}")

    # Round num (OR number within a set: 1, 2, 3) — only meaningful in OR
    if r_round == "Operating" and p_round == "Operating":
        r_rnum = rust_game.round.round_num
        p_rnum = p_round_obj.round_num
        if r_rnum != p_rnum:
            errors.append(f"Round num: Rust={r_rnum} Python={p_rnum}")

    # Current entity
    p_cur = p_round_obj.current_entity
    p_cur_id = str(p_cur.id) if p_cur else None
    # Rust: derive from round state
    r_cur_id = None
    if r_round == "Operating":
        r_or_step = rust_game.get_or_step()  # e.g. "LayTile entity_idx=0 corp=PRR"
        if "corp=" in r_or_step:
            r_cur_id = r_or_step.split("corp=")[1]
    elif r_round == "Stock":
        # Current player from round state
        for rp in rust_game.players:
            # The round state tracks current player but we need the PyO3 accessor
            pass
        # Use the round's active entity from the round_state
        # (RoundState stores active_entity_id internally but we can derive from
        # the stock round's player order + index)
        pass
    if r_round == "Operating" and p_round == "Operating" and r_cur_id and p_cur_id:
        # During company abilities (DH/CS), Python's current entity is the company
        # while Rust keeps the owning corp. Don't compare in that case.
        is_company_ability = p_cur_id in ("DH", "CS", "SV", "MH", "CA", "BO")
        if r_cur_id != p_cur_id and not is_company_ability:
            errors.append(f"Current entity: Rust={r_cur_id} Python={p_cur_id}")

    # Active step (OR only)
    if r_round == "Operating" and p_round == "Operating":
        r_step_str = rust_game.get_or_step()
        r_step = r_step_str.split(" ")[0] if r_step_str else "?"
        p_step = py_game.active_step().__class__.__name__
        # Map Python step names to Rust step names
        step_map = {
            "Track": "LayTile", "Token": "PlaceToken", "HomeToken": "PlaceToken",
            "Route": "RunRoutes", "Dividend": "Dividend",
            "DiscardTrain": "DiscardTrain", "BuyTrain": "BuyTrain",
            "BuyCompany": "BuyCompany", "SpecialTrack": "LayTile",
            "SpecialToken": "PlaceToken",
        }
        p_step_mapped = step_map.get(p_step, p_step)
        # Don't compare special steps — they interleave with regular steps
        if p_step not in ("SpecialTrack", "SpecialToken"):
            if r_step != p_step_mapped:
                errors.append(f"OR step: Rust={r_step} Python={p_step}({p_step_mapped})")

    # Operating order (OR only)
    if r_round == "Operating" and p_round == "Operating":
        p_order = [e.id for e in p_round_obj.entities]
        # Rust: we can extract from get_or_step's entity_idx
        # For now, compare indirectly via current entity check above
        pass

    # ---- Bank ----
    r_bank = rust_game.bank.cash
    p_bank = py_game.bank.cash
    if r_bank != p_bank:
        errors.append(f"Bank: Rust={r_bank} Python={p_bank}")

    # ---- Players ----
    for rp in rust_game.players:
        pp = py_game.player_by_id(rp.id)
        if pp and rp.cash != pp.cash:
            errors.append(f"Player {rp.id} ({rp.name}): Rust=${rp.cash} Python=${pp.cash}")

    # ---- Corporations ----
    for rc in rust_game.corporations:
        pc = py_game.corporation_by_id(rc.sym)
        if not pc:
            continue

        # Float status
        if rc.floated != pc.floated():
            errors.append(f"{rc.sym} floated: Rust={rc.floated} Python={pc.floated()}")

        # Cash (only compare if both floated)
        if rc.floated and pc.floated():
            if rc.cash != pc.cash:
                errors.append(f"{rc.sym} cash: Rust={rc.cash} Python={pc.cash}")

        # IPO/par state
        r_ipoed = rc.ipo_price is not None
        p_ipoed = pc.ipoed
        if r_ipoed != p_ipoed:
            errors.append(f"{rc.sym} ipoed: Rust={r_ipoed} Python={p_ipoed}")

        # Par price
        if r_ipoed and p_ipoed:
            r_par = rc.ipo_price.price
            p_par_val = pc.par_price() if callable(pc.par_price) else pc.par_price
            p_par = p_par_val.price if p_par_val else None
            if r_par != p_par:
                errors.append(f"{rc.sym} par_price: Rust={r_par} Python={p_par}")

        # Share price (price AND coordinates)
        r_sp = rc.share_price
        p_sp = pc.share_price
        if (r_sp is None) != (p_sp is None):
            errors.append(f"{rc.sym} share_price: Rust={'None' if not r_sp else r_sp.price} Python={'None' if not p_sp else p_sp.price}")
        elif r_sp and p_sp:
            if r_sp.price != p_sp.price:
                errors.append(f"{rc.sym} share_price: Rust={r_sp.price} Python={p_sp.price}")
            r_coords = (r_sp.row, r_sp.column)
            p_coords = p_sp.coordinates
            if r_coords != p_coords:
                errors.append(f"{rc.sym} share_price coords: Rust={r_coords} Python={p_coords}")

        # President
        r_pres = None
        for s in rc.shares:
            if s.president:
                pid = s.owner.replace("player:", "") if s.owner.startswith("player:") else None
                r_pres = int(pid) if pid and pid.isdigit() else None
                break
        p_pres = pc.owner.id if pc.owner and hasattr(pc.owner, "id") else None
        if r_pres != p_pres:
            errors.append(f"{rc.sym} president: Rust={r_pres} Python={p_pres}")

        # Share ownership (only compare after IPO)
        if r_ipoed and p_ipoed:
            r_owners = _rust_share_ownership(rust_game, rc.sym)
            p_owners = _python_share_ownership(py_game, rc.sym)
            if r_owners != p_owners:
                errors.append(f"{rc.sym} shares: Rust={r_owners} Python={p_owners}")

        # Trains (sorted by name)
        r_trains = sorted(t.name for t in rc.trains)
        p_trains = sorted(t.name for t in pc.trains)
        if r_trains != p_trains:
            errors.append(f"{rc.sym} trains: Rust={r_trains} Python={p_trains}")

        # Token positions
        if rc.floated and pc.floated():
            r_tokens = sorted(t.city_hex_id for t in rc.tokens if t.used)
            p_tokens = sorted(t.hex.id for t in pc.tokens if t.used)
            if r_tokens != p_tokens:
                errors.append(f"{rc.sym} tokens: Rust={r_tokens} Python={p_tokens}")

    # ---- Graph connectivity (OR only, per floated corp) ----
    if r_round == "Operating" and p_round == "Operating":
        for rc in rust_game.corporations:
            if not rc.floated:
                continue
            pc = py_game.corporation_by_id(rc.sym)
            if not pc or not pc.floated():
                continue

            p_graph = py_game.token_graph_for_entity(pc)
            # Clear stale cache — graph may not have been invalidated for this corp
            p_graph.clear_graph_for(pc)

            # Connected nodes (revenue locations reachable from tokens).
            # This is what matters for route running and token placement.
            # connected_hexes differs because Python includes adjacent hexes
            # that don't have matching exits (hex-level adjacency vs path-level).
            p_node_hexes = set()
            for n in p_graph.connected_nodes(pc):
                if hasattr(n, 'hex'):
                    p_node_hexes.add(n.hex.id)
            r_node_hexes = set(rust_game.get_connected_hexes(rc.sym))
            # Only flag if Rust is missing a hex that has a connected node
            missing_in_rust = sorted(p_node_hexes - r_node_hexes)
            if missing_in_rust:
                errors.append(f"{rc.sym} connected_nodes missing in Rust: {missing_in_rust}")

            # Tokenable cities (only compare if corp has unplaced tokens)
            has_unplaced_r = any(not t.used for t in rc.tokens)
            has_unplaced_p = any(not t.used for t in pc.tokens)
            if has_unplaced_r and has_unplaced_p:
                r_tc = sorted(set(rust_game.get_tokenable_cities(rc.sym)))
                p_tc = sorted(set((c.hex.id, c.index) for c in p_graph.tokenable_cities(pc)))
                if r_tc != p_tc:
                    errors.append(f"{rc.sym} tokenable_cities: Rust={r_tc} Python={p_tc}")

    # ---- Phase ----
    r_phase = rust_game.phase.name
    p_phase = py_game.phase.name
    if r_phase != p_phase:
        errors.append(f"Phase: Rust={r_phase} Python={p_phase}")

    # ---- Depot ----
    r_depot = sorted(t.name for t in rust_game.depot.trains)
    p_depot = sorted(t.name for t in py_game.depot.upcoming)
    if r_depot != p_depot:
        errors.append(f"Depot: Rust={r_depot[:8]}... Python={p_depot[:8]}...")

    # ---- Companies ----
    for i, rc in enumerate(rust_game.companies):
        if i < len(py_game.companies):
            pc = py_game.companies[i]
            # Closed status
            if rc.closed != pc.closed:
                errors.append(f"Company {rc.sym} closed: Rust={rc.closed} Python={pc.closed}")

    # ---- Hex tiles (compare placed/upgraded tiles only) ----
    # Preprinted tiles have different naming conventions between engines.
    # Only compare tiles that have been upgraded (numeric tile IDs like "9-0", "57", etc).
    py_hex_map = {h.id: h for h in py_game.hexes}
    for rh in rust_game.hexes:
        ph = py_hex_map.get(rh.id)
        if not ph:
            continue
        r_tile = rh.tile.name if rh.tile else None
        p_tile = ph.tile.name if ph.tile else None
        # Skip preprinted tiles — names differ by convention
        if r_tile and (r_tile == rh.id or r_tile.startswith("preprinted")):
            continue
        if p_tile and (p_tile == ph.id or p_tile in ("city", "town", "")):
            continue
        # Both are placed tiles — compare base tile ID (strip instance suffix)
        r_base = r_tile.split("-")[0] if r_tile else None
        p_base = p_tile.split("-")[0] if p_tile else None
        if r_base != p_base:
            errors.append(f"Hex {rh.id} tile: Rust={r_tile} Python={p_tile}")

    return errors


def replay_game(game_path: str, max_actions: int = None, verbose: bool = False):
    data = json.load(open(game_path))
    actions = data.get("actions", [])
    if max_actions:
        actions = actions[:max_actions]

    total = len(actions)
    name = Path(game_path).stem
    print(f"[{name}] Replaying {total} actions")

    names = {p.get("id"): p["name"] for p in data["players"]}
    game_cls = GameMap().game_by_title("1830")
    py_game = game_cls(names)
    rust_game = RustGame(names)

    passed = 0
    failed = 0
    rust_errors = 0
    py_errors = 0

    for i, action in enumerate(actions):
        action_type = action.get("type", "?")
        entity = action.get("entity", "?")

        # Python
        try:
            py_game = py_game.process_action(action)
        except Exception as e:
            py_errors += 1
            if verbose or py_errors <= 3:
                print(f"  #{i} {action_type} entity={entity}: Python error: {e}")
            if py_errors >= 5:
                print("  Too many Python errors, stopping.")
                break
            continue

        # Rust
        try:
            rust_game.process_action(action)
        except Exception as e:
            rust_errors += 1
            if verbose or rust_errors <= 10:
                print(f"  #{i} {action_type} entity={entity}: Rust error: {e}")
            if rust_errors >= 20:
                print("  Too many Rust errors, stopping.")
                break
            continue

        # Compare
        errors = compare_state(rust_game, py_game)
        if errors:
            failed += 1
            if verbose or failed <= 5:
                print(f"  #{i} {action_type} entity={entity}: State mismatch:")
                for e in errors:
                    print(f"     {e}")
            if failed >= 30:
                print("  Too many mismatches, stopping.")
                break
        else:
            passed += 1
            if verbose:
                print(f"  #{i} {action_type} entity={entity}: ok")

    print(f"[{name}] {passed} matched, {failed} mismatches, {rust_errors} Rust errors, {py_errors} Python errors (of {total})")
    return failed == 0 and rust_errors == 0


DEFAULT_GAMES = [
    "tests/test_games/manual_game.json",
    "tests/test_games/manual_game_bankrupcy.json",
    "tests/test_games/manual_game_discard_train.json",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("game_paths", nargs="*", default=DEFAULT_GAMES)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    all_ok = True
    for path in args.game_paths:
        if not Path(path).exists():
            print(f"[{Path(path).stem}] File not found, skipping")
            continue
        ok = replay_game(path, max_actions=args.max_actions, verbose=args.verbose)
        if not ok:
            all_ok = False
        print()

    sys.exit(0 if all_ok else 1)
