"""Verify that pretraining cleaning produces equivalent results with use_rust=True vs False.

For a random sample of games:
1. Clean with use_rust=False (Python engine)
2. Clean with use_rust=True (Rust engine via adapter)
3. Compare action streams — they should be identical for both engines.
"""
from __future__ import annotations
import sys, json, logging
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
logging.disable(logging.CRITICAL)

from rl18xx.agent.alphazero.pretraining import get_game_object_for_game

CORPUS_DIR = REPO_ROOT / "human_games" / "1830"

def cmp_game(game_id):
    raw = json.loads((CORPUS_DIR / f"{game_id}.json").read_text())
    if raw.get("status") != "finished":
        return (game_id, "SKIP_NOT_FINISHED")
    try:
        py = get_game_object_for_game(raw, use_rust=False)
    except Exception as e:
        return (game_id, f"PY_ERR: {type(e).__name__}")
    try:
        rs = get_game_object_for_game(raw, use_rust=True)
    except Exception as e:
        return (game_id, f"RUST_ERR: {type(e).__name__}")
    if (py is None) != (rs is None):
        return (game_id, f"DROP_MISMATCH py={py is None} rust={rs is None}")
    if py is None:
        return (game_id, "BOTH_DROPPED")
    # Compare action streams. Python's BaseGame.to_dict has "actions";
    # the Rust adapter doesn't, so read raw_actions directly.
    py_actions = py.to_dict().get("actions") or list(py.raw_actions)
    rs_actions = list(rs.raw_actions)
    if len(py_actions) != len(rs_actions):
        return (game_id, f"LEN_DIFF py={len(py_actions)} rust={len(rs_actions)}")
    # Normalize and compare each action — ignore id/created_at/user noise
    # plus empty/None fields (Python emits extras as None or []; Rust omits them).
    # Run_routes route subdicts get extra normalization:
    #
    #   * ``nodes`` — derived field that the Rust engine doesn't track
    #     (Python re-computes from ``route.node_signatures`` at emit time).
    #   * ``hexes`` — Python iterates these from a hash-randomized ``set`` and
    #     in some configurations drops the start-of-route token hex entirely
    #     (the Route's ``connection_data`` skips the token's chain when the
    #     train is non-local, leaving only stops in ``hexes``). The Rust
    #     adapter preserves the caller-provided ``hexes`` verbatim. ``hexes``
    #     is a denormalization of ``connections`` anyway, so dropping it from
    #     the comparison still catches semantic route divergences.
    def norm_route(r):
        out = {}
        for k, v in r.items():
            if k in ("nodes", "hexes"):
                continue
            if v in (None, [], {}):
                continue
            out[k] = v
        return out

    def norm(a):
        out = {}
        for k, v in a.items():
            if k in ("id", "created_at", "user", "auto_actions"):
                continue
            if k == "routes" and isinstance(v, list):
                out[k] = [norm_route(r) for r in v]
                continue
            if v in (None, [], {}):
                continue
            out[k] = v
        return out

    for i, (a, b) in enumerate(zip(py_actions, rs_actions)):
        if norm(a) != norm(b):
            return (game_id, f"ACTION_DIFF at #{i}: py={norm(a)} rust={norm(b)}")
    return (game_id, "MATCH")

def main():
    import multiprocessing as mp
    paths = sorted(CORPUS_DIR.glob("*.json"))
    import random
    rng = random.Random(0)
    sample = rng.sample([p.stem for p in paths], min(100, len(paths)))
    with mp.Pool() as pool:
        results = pool.map(cmp_game, sample)
    counts = Counter(r[1].split(":", 1)[0] for r in results)
    print(f"Sample of {len(sample)} games:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    bad = [r for r in results if r[1] not in ("MATCH", "BOTH_DROPPED", "SKIP_NOT_FINISHED")]
    if bad:
        print(f"\nFailures ({len(bad)}):")
        for r in bad[:10]:
            print(f"  {r[0]}: {r[1][:200]}")

if __name__ == "__main__":
    main()
