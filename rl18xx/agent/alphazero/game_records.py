"""Saved games: the action logs of trained agents' games, kept for browsing.

A *collection* is a directory with one JSON file per game under ``games/``
and an index, ``games.jsonl``, with one record per game. Writers:

- ``scripts/eval_head_to_head.py`` (every game): ``games/<game_idx>.json`` is
  a bare list of action dicts; its index record names the match (an index
  into ``settings.json``'s ``matches``) and each seat's side (``seats``,
  A/B). Games without ``--start-positions`` used the ``auction_unlock``
  variant, games with them didn't.
- ``scripts/eval_policy_only.py --save-games N`` and the policy-gradient
  stage's ``save_game_every`` (``main.py policy-gradient --save-game-every``):
  ``games/<name>.json`` is a :func:`make_game_record` -- an 18xx.games-style
  export with this repo's metadata under ``"rl18xx"`` -- and its index
  record names the file (``game_file``) and labels each seat
  (``seat_labels``).

The dashboard's game viewer (``/games``) lists the collections under
:data:`DEFAULT_ROOTS` and steps through a game by replaying it in the Python
reference engine; ``main.py replay <file>`` reads the same files. Records of
games whose actions weren't kept (``eval_policy_only.py`` keeps only some)
have no file and aren't listed.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from rl18xx.shared.atomic_io import atomic_write_json

GAME_RECORD_VERSION = 1
GAMES_DIR = "games"
INDEX_FILE = "games.jsonl"
# Where the dashboard looks for collections (relative to the repo root): each
# root itself, and each directory directly under it, that has a games/ dir.
DEFAULT_ROOTS = ("logs/eval", "model_checkpoints_pg", "logs/games")
# Self-play's log line with a game's actions (``self_play.py``; ``main.py train`` logs).
LOG_ACTIONS_MARKER = "Game actions: "
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def make_game_record(
    actions: Iterable[dict],
    num_players: int,
    *,
    auction_unlock: bool = False,
    meta: Optional[dict] = None,
    game_id: Optional[str] = None,
) -> dict:
    """An 18xx.games-style export of a game's ``actions`` (what ``BaseGame.load``
    and a self-hosted 18xx server's import read) carrying ``meta`` -- who held
    which seat, the result, where the game came from -- under ``"rl18xx"``.
    Players are "Player 1".."Player N" with ids 1..N, as every engine game in
    this repo is built; seat ``i`` is player ``i + 1``."""
    meta = dict(meta or {})
    return {
        "id": game_id or meta.get("name") or "rl18xx",
        "title": "1830",
        "description": str(meta.get("description") or ""),
        "players": [{"id": i, "name": f"Player {i}"} for i in range(1, num_players + 1)],
        "min_players": num_players,
        "max_players": num_players,
        "settings": {"optional_rules": [], "seed": 0},
        "actions": [dict(action) for action in actions],
        "rl18xx": {"version": GAME_RECORD_VERSION, "auction_unlock": bool(auction_unlock), **meta},
    }


def save_game_record(collection_dir, name: str, record: dict) -> str:
    """Write ``record`` to ``<collection_dir>/games/<name>.json`` (atomically) and
    return its path relative to the collection, for the index's ``game_file``."""
    if not _NAME_RE.match(name):
        raise ValueError(f"Bad game file name {name!r}")
    relative = f"{GAMES_DIR}/{name}.json"
    atomic_write_json(Path(collection_dir) / relative, record, indent=None)
    return relative


def append_index(collection_dir, entry: dict) -> None:
    """Append ``entry`` to the collection's ``games.jsonl``."""
    path = Path(collection_dir) / INDEX_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def winners(win_share) -> list:
    """Seats with a share of the win (all tied leaders)."""
    return [seat for seat, share in enumerate(win_share or []) if share and share > 0]


# --------------------------------------------------------------------- reading
@dataclass
class LoadedGame:
    actions: list
    num_players: int
    # None when the file doesn't say (eval_head_to_head's bare lists, logs):
    # the viewer takes it from the collection's settings or tries both.
    auction_unlock: Optional[bool] = None
    meta: dict = field(default_factory=dict)


def _infer_num_players(actions: list, default: int = 4) -> int:
    """Every player acts in the private auction, so the highest player id is the count."""
    ids = [a.get("entity") for a in actions if a.get("entity_type") == "player" and isinstance(a.get("entity"), int)]
    return max(ids) if ids else default


def _actions_from_log(text: str) -> Optional[list]:
    """The actions of the first ``Game actions: [...]`` line of a log (a Python
    list repr, as self-play logs it)."""
    for line in text.splitlines():
        if LOG_ACTIONS_MARKER in line:
            payload = line.split(LOG_ACTIONS_MARKER, 1)[1].strip()
            try:
                actions = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                continue
            if isinstance(actions, list):
                return actions
    return None


def load_game_file(path) -> LoadedGame:
    """Read a saved game: a :func:`make_game_record` file, any 18xx.games-style
    export (``{"players": [...], "actions": [...]}``), a bare list of action
    dicts (``eval_head_to_head.py``), or a log with self-play's ``Game actions:``
    line. Raises ValueError for anything else."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _actions_from_log(text)
        if data is None:
            raise ValueError(f"{path}: neither game JSON nor a log with a '{LOG_ACTIONS_MARKER}' line")
    if isinstance(data, list):
        if not all(isinstance(a, dict) for a in data):
            raise ValueError(f"{path}: a list, but not of action dicts")
        return LoadedGame(actions=data, num_players=_infer_num_players(data))
    if isinstance(data, dict) and isinstance(data.get("actions"), list):
        meta = data.get("rl18xx") if isinstance(data.get("rl18xx"), dict) else {}
        players = data.get("players")
        num_players = len(players) if isinstance(players, list) and players else _infer_num_players(data["actions"])
        unlock = meta.get("auction_unlock")
        return LoadedGame(
            actions=data["actions"],
            num_players=num_players,
            auction_unlock=bool(unlock) if unlock is not None else None,
            meta={k: v for k, v in meta.items() if k not in ("version", "auction_unlock")},
        )
    raise ValueError(f"{path}: not a saved game")


def _read_json(path) -> Optional[object]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path) -> list:
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # an append in progress
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        pass
    return records


# ----------------------------------------------------------------- collections
def find_collections(repo_root, roots: Iterable[str] = DEFAULT_ROOTS) -> list:
    """Summaries of the collections under ``roots`` (newest first): ``id`` (the
    path relative to ``repo_root``), ``kind``, ``title``, ``games`` (saved game
    files) and ``updated_unix``."""
    repo_root = Path(repo_root)
    found = []
    for root in roots:
        base = repo_root / root
        if not base.is_dir():
            continue
        candidates = [base]
        try:
            candidates += sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            pass
        for directory in candidates:
            games_dir = directory / GAMES_DIR
            if not games_dir.is_dir():
                continue
            try:
                count = sum(1 for p in games_dir.iterdir() if p.suffix == ".json")
            except OSError:
                continue
            if not count:
                continue
            kind, title = describe_collection(directory)
            index = directory / INDEX_FILE
            try:
                updated = max(games_dir.stat().st_mtime, index.stat().st_mtime if index.exists() else 0.0)
            except OSError:
                updated = 0.0
            found.append(
                {
                    "id": str(directory.relative_to(repo_root)),
                    "kind": kind,
                    "title": title,
                    "games": count,
                    "updated_unix": updated,
                }
            )
    found.sort(key=lambda c: c["updated_unix"], reverse=True)
    return found


def describe_collection(directory) -> tuple:
    """``(kind, title)`` of a collection from the files its writer leaves."""
    directory = Path(directory)
    settings = _read_json(directory / "settings.json")
    if isinstance(settings, dict) and settings.get("matches"):
        matches = "; ".join(f"{a} vs {b}" for a, b in settings["matches"])
        return "head_to_head", f"{matches} (search, {settings.get('readouts')} readouts)"
    config = _read_json(directory / "config.json")
    if isinstance(config, dict) and "policy_checkpoint" in config:
        return "policy_gradient", f"policy-gradient run {config.get('run_name') or directory.name}"
    summary = _read_json(directory / "summary.json")
    if isinstance(summary, list) and summary and isinstance(summary[0], dict) and "A" in summary[0]:
        matches = "; ".join(f"{s.get('A')} vs {s.get('B')}" for s in summary)
        return "policy_only", f"{matches} (policy only)"
    return "games", directory.name


def _h2h_context(directory: Path) -> dict:
    settings = _read_json(directory / "settings.json")
    if not isinstance(settings, dict) or not settings.get("matches"):
        return {}
    return {"matches": settings["matches"], "auction_unlock": not settings.get("start_positions")}


def _index_entry(record: dict, context: dict) -> Optional[dict]:
    """A listing entry for an index record, or None when it has no game file."""
    entry = {
        k: v
        for k, v in record.items()
        if k not in ("rows", "raw_actions") and (v is None or isinstance(v, (str, int, float, bool, list)))
    }
    game_file = record.get("game_file")
    if not game_file and "game_idx" in record and context.get("matches") is not None:
        game_file = f"{GAMES_DIR}/{record['game_idx']}.json"  # eval_head_to_head.py
        match = record.get("match")
        matches = context["matches"]
        if isinstance(match, int) and 0 <= match < len(matches):
            a, b = matches[match]
            entry["match"] = f"{a} vs {b}"
            sides = record.get("seats") or []
            entry["sides"] = sides
            entry["seat_labels"] = [a if side == "A" else b for side in sides]
        entry["auction_unlock"] = context.get("auction_unlock")
        entry.pop("seats", None)
    if not game_file:
        return None
    entry["game_file"] = game_file
    entry["name"] = Path(game_file).stem
    if "seats" in entry and "sides" not in entry:
        entry["sides"] = entry.pop("seats")
    count = record.get("decision_count", record.get("decisions"))
    if isinstance(count, dict):  # older eval_head_to_head.py records: decisions by round type
        count = sum(v for v in count.values() if isinstance(v, int))
    entry["decisions"] = count if isinstance(count, int) else None
    entry["winners"] = winners(record.get("win_share"))
    return entry


def list_games(collection_dir) -> list:
    """Listing entries for a collection's saved games: its index records that
    have a file (newest last, as written), then any files the index doesn't
    name (with the metadata stored in them)."""
    directory = Path(collection_dir)
    games_dir = directory / GAMES_DIR
    context = _h2h_context(directory)
    entries, named = [], set()
    for record in _read_jsonl(directory / INDEX_FILE):
        entry = _index_entry(record, context)
        if entry is None or entry["name"] in named or not (directory / entry["game_file"]).is_file():
            continue
        named.add(entry["name"])
        entries.append(entry)
    try:
        unindexed = sorted(p for p in games_dir.iterdir() if p.suffix == ".json" and p.stem not in named)
    except OSError:
        unindexed = []
    for path in unindexed:
        if not _NAME_RE.match(path.stem):
            continue
        entry = {"name": path.stem, "game_file": f"{GAMES_DIR}/{path.name}"}
        try:
            meta = load_game_file(path).meta
        except (OSError, ValueError):
            meta = {}
        entry.update({k: v for k, v in meta.items() if v is None or isinstance(v, (str, int, float, bool, list))})
        entry.setdefault("winners", winners(meta.get("win_share")))
        if "auction_unlock" not in entry and context:
            entry["auction_unlock"] = context.get("auction_unlock")
        entries.append(entry)
    return entries


def resolve_collection(repo_root, collection_id: str, roots: Iterable[str] = DEFAULT_ROOTS) -> Optional[Path]:
    """The directory of collection ``collection_id`` (a path relative to
    ``repo_root``), or None unless it's a collection inside one of ``roots``."""
    if not collection_id or "\x00" in collection_id:
        return None
    repo_root = Path(repo_root).resolve()
    directory = (repo_root / collection_id).resolve()
    allowed = [(repo_root / root).resolve() for root in roots]
    if not any(directory == base or base in directory.parents for base in allowed):
        return None
    return directory if (directory / GAMES_DIR).is_dir() else None


def resolve_game_file(collection_dir, name: str) -> Optional[Path]:
    """``<collection>/games/<name>.json`` if it exists and ``name`` is a plain file stem."""
    if not name or not _NAME_RE.match(name):
        return None
    path = Path(collection_dir) / GAMES_DIR / f"{name}.json"
    return path if path.is_file() else None


def game_context(collection_dir, name: str) -> dict:
    """What the index knows about game ``name`` (its listing entry), or {}."""
    for entry in list_games(collection_dir):
        if entry["name"] == name:
            return entry
    return {}


def file_mtime_ns(path) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return 0
