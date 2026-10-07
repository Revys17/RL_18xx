"""Replay a saved game: check it in the Python engine, print it, and open it in the game viewer.

Reads anything ``game_records.load_game_file`` does: a saved game
(``<collection>/games/<name>.json``, written by ``scripts/eval_head_to_head.py``,
``scripts/eval_policy_only.py --save-games`` and ``main.py policy-gradient
--save-game-every``), an 18xx.games-style export (the viewer's "Download
JSON"), or a log with self-play's ``Game actions: [...]`` line (``main.py
train``'s log).

    python main.py replay <file>          replay it, print the standings and the viewer URL
    python main.py replay <file> --log    ... and print the game log

A file outside the viewer's collections (``game_records.DEFAULT_ROOTS``) is
copied into ``logs/games/replays/`` so the dashboard (``/games``) lists it.

``--local-server URL`` re-creates the game on a self-hosted 18xx.games server
(``GameSync``: the server's accounts a/b/c/z play the four seats) instead.
Never point it at the public site. Games that used the ``auction_unlock``
variant (eval_head_to_head.py without ``--start-positions``) can't be
replayed there: the Ruby engine doesn't have it.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

from rl18xx.agent.alphazero import game_records

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAYS_COLLECTION = "logs/games/replays"
DASHBOARD_URL = "http://localhost:5001"


def get_game_actions_from_log_file(log_file_path: str) -> list[dict]:
    """The actions of a saved game or log (see the module docstring)."""
    return game_records.load_game_file(log_file_path).actions


def replay_locally(path, auction_unlock_hint: Optional[bool] = None) -> dict:
    """``game_viewer.game_summary`` of the game: its log, standings and where (if anywhere) the engine stopped."""
    from rl18xx.agent.dashboard import game_viewer

    return game_viewer.game_summary(path, auction_unlock_hint)


def _collection_of(path: Path, repo_root: Path) -> Optional[str]:
    """The viewer collection a game file belongs to (its ``games/`` dir's parent), if any."""
    path = path.resolve()
    if path.parent.name != game_records.GAMES_DIR:
        return None
    try:
        collection = str(path.parent.parent.relative_to(repo_root.resolve()))
    except ValueError:
        return None
    if game_records.resolve_collection(repo_root, collection) is None:
        return None
    return collection


def add_to_viewer(path: Path, summary: dict, repo_root: Optional[Path] = None) -> tuple:
    """``(collection, game name)`` for the viewer: the file's own collection, or a
    copy written to ``logs/games/replays/games/``."""
    repo_root = Path(repo_root or REPO_ROOT)
    collection = _collection_of(path, repo_root)
    if collection is not None:
        return collection, path.stem
    loaded = game_records.load_game_file(path)
    name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in path.stem) or "game"
    meta = {**loaded.meta, "name": name, "description": f"replayed from {path}", "source": str(path)}
    meta.setdefault("termination", "finished" if summary["finished"] else None)
    record = game_records.make_game_record(
        loaded.actions, loaded.num_players, auction_unlock=summary["auction_unlock"], meta=meta
    )
    game_records.save_game_record(repo_root / REPLAYS_COLLECTION, name, record)
    return REPLAYS_COLLECTION, name


def viewer_url(collection: str, name: str, base_url: str = DASHBOARD_URL) -> str:
    return f"{base_url}/games/view?collection={quote(collection)}&game={quote(name)}"


def replay_on_local_server(path, base_url: str):
    """Re-create the game on a self-hosted 18xx server (``GameSync``)."""
    host = (urlparse(base_url).hostname or "").lower()
    if host.endswith("18xx.games"):
        raise SystemExit("Refusing to replay onto the public 18xx.games site; use a self-hosted server.")
    from rl18xx.client.game_sync import GameSync

    loaded = game_records.load_game_file(path)
    # A game whose log only replays with the variant on used it.
    if replay_locally(path)["auction_unlock"]:
        raise SystemExit("This game used the auction_unlock variant, which an 18xx server doesn't have.")
    game_sync = GameSync(base_url=base_url)
    game_sync.replay_game(loaded.actions)
    print(f"Replicated {path} to game {game_sync.game_id} on {base_url}")
    return game_sync


def replay_game_from_log_file(log_file_path: str, print_log: bool = False, base_url: str = DASHBOARD_URL):
    """Replay a saved game in the Python engine, print its standings (and log),
    and return the viewer URL to step through it."""
    path = Path(log_file_path)
    summary = replay_locally(path)
    if print_log:
        for step, message in summary["log"]:
            print(f"{step:5d}  {message}")
    status = "finished" if summary["finished"] else "not finished"
    if summary["error"]:
        status = f"stopped at {summary['error']}"
    print(f"{path}: {summary['playable_steps']}/{summary['steps']} actions replayed ({status})")
    for player, value in summary["result"].items():
        print(f"  {player}: ${value}")
    collection, name = add_to_viewer(path, summary)
    url = viewer_url(collection, name, base_url)
    print(f"View it in the dashboard (main.py dashboard / ./startup.sh): {url}")
    return url


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_file_path", type=str)
    parser.add_argument("--log", action="store_true", help="Print the game log")
    parser.add_argument("--local-server", type=str, default=None, help="Self-hosted 18xx server URL to replay onto")
    args = parser.parse_args()
    if args.local_server:
        replay_on_local_server(args.log_file_path, args.local_server)
    else:
        replay_game_from_log_file(args.log_file_path, print_log=args.log)
