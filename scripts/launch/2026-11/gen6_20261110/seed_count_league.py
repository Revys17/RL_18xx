"""Seeds gen6's per-count PFSP (CountLeague, 2026-11-12) from its saved training games.

gen6 ran its first ~470 updates with one pooled PFSP rate per league member. The new code reads a pooled
state by seeding every count at the pooled rate (as 4 pseudo-games), so a member the learner loses to at one
count only (pg2p at 2 players) would wait for a slot draw at its pooled weight before its own games count.
This rebuilds each count's rates from the saved training games (~1 in 20, ``save_game_every 20``): per count
and member, the pooled seed (``prior`` pseudo-games) plus every comparison in saved games since
``--since-update``, computed as the trainer does (per opponent seat, the share of the learner's seats that
finished ahead of it by net worth, ties half).

    seed_count_league.py <run dir> --since-update 270            # dry run: per-count rates and shares
    seed_count_league.py <run dir> --since-update 270 --write    # state.json -> by_count (backup kept)
Run it with the deployment's python (imports rl18xx) while the run is stopped.
"""

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from rl18xx.agent.alphazero import policy_gradient as pg


def comparisons(games_path: Path, since: int):
    """``(players, member, learner result)`` per opponent seat of the saved league games since ``since``."""
    for line in games_path.open():
        r = json.loads(line)
        if r.get("opponent") != "league" or int(r.get("update", 0)) < since:
            continue
        n, mine = len(r["worth_share"]), r["learner_seats"]
        for seat, member in enumerate(r["seat_labels"]):
            if seat in mine:
                continue
            theirs = r["worth_share"][seat]
            results = [1.0 if r["worth_share"][x] > theirs else 0.5 if r["worth_share"][x] == theirs else 0.0 for x in mine]
            yield n, member, sum(results) / len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--since-update", type=int, default=270)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    state_path = args.run_dir / "state.json"
    state = json.loads(state_path.read_text())
    cfg = pg.PGConfig(**state["config"])
    if "by_count" in state["league"]:
        raise SystemExit("state.json already has per-count league stats")
    league = pg.league_for_run(cfg, state["league"])  # pooled -> every count at the pooled rate (prior games)
    seen = defaultdict(int)
    for n, member, result in comparisons(args.run_dir / "games.jsonl", args.since_update):
        if member in league.members and n in league.counts:
            wins, games = league.league(n).stats[member]
            league.league(n).stats[member] = [wins + result, games + 1.0]
            seen[(n, member)] += 1
    for n in league.counts:
        lg = league.league(n)
        shares = dict(zip(lg.members, lg.weights()))
        top = sorted(lg.members, key=lambda m: -shares[m])[:6]
        print(f"{n}p: " + "; ".join(
            f"{'/'.join(Path(m).parts[-3::2])} rate {lg.win_rate(m):.2f} (n {seen[(n, m)]}) share {shares[m]:.3f}" for m in top
        ))
    if args.write:
        backup = state_path.with_name(f"state.pooled-{time.strftime('%Y%m%d%H%M%S')}.json")
        shutil.copy2(state_path, backup)
        state["league"] = league.state()
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_path)
        print(f"wrote per-count league stats to {state_path} (pooled state kept in {backup})")


if __name__ == "__main__":
    main()
