"""Bankruptcies in the games an ``eval_policy_only.py --save-games`` run kept.

Per match: the share of saved games that ended with a bankrupt player, split by
the side (A/B) of the player who went bankrupt. Each game is replayed in the
Rust engine from its saved actions.

    uv run python scripts/count_bankruptcies.py logs/eval/<out dir>

(2026-10-15: the round-3 bankruptcy hunter put gen4 bankrupt in 8.3% of games,
gen3 12.0%, gen2 17.5%.)
"""

import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl18xx.agent.alphazero.start_positions import apply_actions, new_game  # noqa: E402


def main():
    out_dir = Path(sys.argv[1])
    per_match = collections.defaultdict(collections.Counter)
    for line in open(out_dir / "games.jsonl"):
        record = json.loads(line)
        if "game_file" not in record:  # not one of the saved games
            continue
        data = json.loads((out_dir / record["game_file"]).read_text())
        actions = data["actions"] if isinstance(data, dict) else data
        game = apply_actions(new_game(len(record["seats"])), actions)
        bankrupt = {p.id: p.bankrupt for p in game._game.players}
        counts = per_match[record["match"]]
        counts["games"] += 1
        seats = [s for s, pid in enumerate(sorted(bankrupt)) if bankrupt[pid]]
        if seats:
            counts["any"] += 1
            for s in seats:
                counts[f"side {record['seats'][s]} bankrupt"] += 1
    for match, counts in per_match.items():
        n = counts["games"]
        a, b = match.split(" vs ")
        print(
            f"{a[-45:]} vs {b[-45:]}: {n} games, ended by bankruptcy {counts['any'] / n:.3f} "
            f"(A side bankrupt {counts['side A bankrupt'] / n:.3f}, B side {counts['side B bankrupt'] / n:.3f})"
        )


if __name__ == "__main__":
    main()
