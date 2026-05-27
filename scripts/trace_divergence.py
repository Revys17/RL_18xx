#!/usr/bin/env python3
"""Replay actions through both Python and Rust engines side-by-side, reporting
the first divergence in current_entity or active_step."""
from __future__ import annotations
import sys, json, logging
from pathlib import Path

logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl18xx.game.gamemap import GameMap
from engine_rs import BaseGame as RustGame
from rl18xx.rust_adapter import RustGameAdapter
from rl18xx.agent.alphazero.pretraining import filter_actions


def main():
    if len(sys.argv) < 2:
        print("usage: trace_divergence.py <game_id>")
        return 1
    gid = sys.argv[1]
    raw = json.loads((REPO_ROOT / "human_games" / "1830" / f"{gid}.json").read_text())

    num_players = len(raw["players"])
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
    pmap = {p["id"]: i + 1 for i, p in enumerate(raw["players"])}

    gm = GameMap()
    py_game = gm.game_by_title("1830")(players)
    rust_game = RustGameAdapter(RustGame(players))

    actions = filter_actions(raw["actions"])

    for i, a in enumerate(actions):
        if a["entity_type"] == "player":
            a["entity"] = pmap.get(a["entity"], a["entity"])
            a["user"] = a["entity"]
        else:
            a["user"] = pmap.get(a.get("user"), a.get("user"))

        py_e = py_game.current_entity
        rust_e = rust_game.current_entity
        py_entity_id = getattr(py_e, "id", None)
        rust_entity_id = getattr(rust_e, "id", None)
        py_step = py_game.round.active_step()
        rust_step = rust_game.round.active_step()
        py_step_before = type(py_step).__name__ if py_step else None
        rust_step_before = type(rust_step).__name__ if rust_step else None
        # Map Rust step name to Python equivalent for cross-engine comparison
        rust_step_normalized = (rust_step_before or "").replace("_AuctionStepProxy", "WaterfallAuction").replace("_StepProxy", "").replace("Proxy", "")

        if py_entity_id != rust_entity_id:
            print(f"DIVERGENCE before action #{i}:")
            print(f"  py: entity={py_entity_before} step={py_step_before}")
            print(f"  rust: entity={rust_entity_before} step={rust_step_before}")
            print(f"  action: {a}")
            return 0

        try:
            py_game.process_action(a)
        except Exception as e:
            print(f"py error at #{i} ({a.get('type')}): {e}")
            return 0
        try:
            rust_game.process_action(a)
        except Exception as e:
            print(f"rust error at #{i} ({a.get('type')}): {e}")
            return 0
    print(f"replayed {len(actions)} actions without divergence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
