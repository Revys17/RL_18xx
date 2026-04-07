"""Benchmark Python vs Rust engine: replay recorded games and measure time.

Usage:
    uv run python tests/bench_engines.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame
from rl18xx.game.gamemap import GameMap

GAME_FILES = [
    "tests/test_games/manual_game.json",
    "tests/test_games/manual_game_bankrupcy.json",
    "tests/test_games/manual_game_discard_train.json",
]


def bench_python(actions, names):
    game_cls = GameMap().game_by_title("1830")
    game = game_cls(names)
    start = time.perf_counter()
    for action in actions:
        game = game.process_action(action)
    elapsed = time.perf_counter() - start
    return elapsed


def bench_rust(actions, names):
    game = RustGame(names)
    start = time.perf_counter()
    for action in actions:
        game.process_action(action)
    elapsed = time.perf_counter() - start
    return elapsed


def bench_rust_with_clone(actions, names):
    """Benchmark Rust with pickle_clone at every step (simulating MCTS)."""
    game = RustGame(names)
    start = time.perf_counter()
    for action in actions:
        _ = game.pickle_clone()
        game.process_action(action)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    # Warmup
    print("Warming up...")
    data = json.load(open(GAME_FILES[0]))
    names = {p.get("id"): p["name"] for p in data["players"]}
    actions = data.get("actions", [])[:50]
    bench_python(actions, names)
    bench_rust(actions, names)
    print()

    results = []

    for path in GAME_FILES:
        if not Path(path).exists():
            print(f"  {Path(path).stem}: not found, skipping")
            continue

        data = json.load(open(path))
        names = {p.get("id"): p["name"] for p in data["players"]}
        actions = data.get("actions", [])
        name = Path(path).stem
        n = len(actions)

        # Run each 3 times, take best
        py_times = []
        rust_times = []
        rust_clone_times = []

        for _ in range(3):
            py_times.append(bench_python(actions, names))
            rust_times.append(bench_rust(actions, names))
            rust_clone_times.append(bench_rust_with_clone(actions, names))

        py_best = min(py_times)
        rust_best = min(rust_times)
        rust_clone_best = min(rust_clone_times)
        speedup = py_best / rust_best if rust_best > 0 else float("inf")
        speedup_clone = py_best / rust_clone_best if rust_clone_best > 0 else float("inf")

        print(f"{name} ({n} actions):")
        print(f"  Python:            {py_best:.3f}s  ({n / py_best:.0f} actions/s)")
        print(f"  Rust:              {rust_best:.3f}s  ({n / rust_best:.0f} actions/s)  {speedup:.1f}x faster")
        print(f"  Rust + clone:      {rust_clone_best:.3f}s  ({n / rust_clone_best:.0f} actions/s)  {speedup_clone:.1f}x faster")
        print()

        results.append({
            "name": name,
            "actions": n,
            "python": py_best,
            "rust": rust_best,
            "rust_clone": rust_clone_best,
            "speedup": speedup,
            "speedup_clone": speedup_clone,
        })

    # Summary
    if results:
        total_py = sum(r["python"] for r in results)
        total_rust = sum(r["rust"] for r in results)
        total_rust_clone = sum(r["rust_clone"] for r in results)
        total_actions = sum(r["actions"] for r in results)
        print("=" * 60)
        print(f"Total ({total_actions} actions across {len(results)} games):")
        print(f"  Python:        {total_py:.3f}s  ({total_actions / total_py:.0f} actions/s)")
        print(f"  Rust:          {total_rust:.3f}s  ({total_actions / total_rust:.0f} actions/s)  {total_py / total_rust:.1f}x faster")
        print(f"  Rust + clone:  {total_rust_clone:.3f}s  ({total_actions / total_rust_clone:.0f} actions/s)  {total_py / total_rust_clone:.1f}x faster")


if __name__ == "__main__":
    main()
