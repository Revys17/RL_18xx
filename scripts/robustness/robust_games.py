"""Reading the games the robustness tools work on, whatever wrote them.

``iter_games(inputs)`` yields one :class:`GameItem` per game from

- a collection directory (``games.jsonl`` index + ``games/``): ``eval_policy_only.py --save-games`` / the
  policy-gradient stage's ``make_game_record`` files (seats labelled by ``sides`` / ``learner_seats``),
  ``eval_head_to_head.py``'s bare action lists (seats from the index's ``seats``, the rule variant from
  ``settings.json``), generic runs' saved games (the same ``make_game_record`` files);
- a directory of game files (the human corpus ``human_games/1830_clean_*``, ``data/caleb_games``) or single files;
- ``.json`` and ``.json.gz`` files: a ``make_game_record`` / 18xx.games-style export with engine actions (players
  1..N), this folder's converted site games (``rl18xx.site_action_ids``), or a raw export from an 18xx.games site
  (players are the site's user ids): converted through the advisor's importer (:func:`convert_site_export`).

Each item says which group every seat belongs to (``groups``: player id -> label), so the behaviour statistics can
be split by side: "A" / "B" (evals), "learner" / "league" (policy-gradient games), "bot" / "human" (site games),
"human" (the human corpus).
"""

from __future__ import annotations

import copy
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

ENGINE_RULES = ("multiple_brown_from_ipo", "optional_6_train")


@dataclass
class GameItem:
    path: str
    actions: list
    num_players: int
    optional_rules: list = field(default_factory=list)
    auction_unlock: Optional[bool] = None  # None: unknown (try off, then on)
    start: Optional[str] = None  # start label (``start_positions``): events before it are the recorded prefix
    groups: dict = field(default_factory=dict)  # player id -> group label
    names: dict = field(default_factory=dict)  # player id -> name (site games)
    site_ids: Optional[list] = None  # per action, the site's action id (converted site games)
    finished_hint: Optional[bool] = None  # the file's own "finished" flag, when it has one
    meta: dict = field(default_factory=dict)


def read_json(path) -> object:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def write_json(path, data, compress: Optional[bool] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, separators=(",", ":"), sort_keys=False)
    if compress or (compress is None and path.suffix == ".gz"):
        with gzip.open(path, "wt", compresslevel=9) as f:
            f.write(text)
    else:
        path.write_text(text)


def is_site_export(data: dict) -> bool:
    """A raw export from an 18xx.games site: its players are the site's user ids, not 1..N."""
    players = data.get("players") or []
    ids = [p.get("id") for p in players if isinstance(p, dict)]
    return bool(ids) and ids != list(range(1, len(ids) + 1)) and not (data.get("rl18xx") or {}).get("site_action_ids")


def convert_site_export(game: dict) -> dict:
    """An 18xx.games export (``{"players": [{id, name}] in seat order, "actions": [...site actions...]}``) as an
    engine-format game: the actions the advisor's importer (``pretraining.filter_actions`` +
    ``replay_human_action``, training-data drop rules off) plays in the Rust engine -- undos and messages resolved,
    G19's preprinted cities in the engines' numbering -- with each engine action's site action id
    (``rl18xx.site_action_ids``). Players become "Player 1".."Player N" in seat order (the real names are kept in
    ``rl18xx.names``); the export's own players list must be in seat order."""
    from engine_rs import BaseGame as RustGame

    from rl18xx.agent.advisor.autoplay import swap_city_refs_in_log
    from rl18xx.agent.alphazero.pretraining import engine_optional_rules, filter_actions, replay_human_action
    from rl18xx.rust_adapter import RustGameAdapter

    players = game["players"]
    mapping = {p["id"]: i + 1 for i, p in enumerate(players)}
    rules = list(engine_optional_rules(game))
    rust = RustGameAdapter(RustGame({i: f"Player {i}" for i in range(1, len(players) + 1)}, optional_rules=rules))
    annotated = []
    for a in game["actions"]:
        a = copy.deepcopy(a)
        a["_sid"] = a.get("id")
        for aa in a.get("auto_actions") or []:
            aa["_sid"] = a.get("id")
        annotated.append(a)
    filtered = filter_actions(annotated)
    sids = [f.pop("_sid", None) for f in filtered]
    filtered = swap_city_refs_in_log(filtered)
    site_ids = []
    error = None
    for i in range(len(filtered)):
        before = len(rust.raw_actions)
        try:
            replay_human_action(rust, filtered, i, mapping, use_rust=True, drop_rules=False)
        except Exception as e:  # keep what replayed
            error = f"site action {sids[i]}: {type(e).__name__}: {e}"
            break
        site_ids += [sids[i]] * (len(rust.raw_actions) - before)
    names = {i + 1: p.get("name") or f"Player {i + 1}" for i, p in enumerate(players)}
    return {
        "id": f"site-{game.get('id')}",
        "title": "1830",
        "players": [{"id": i, "name": f"Player {i}"} for i in range(1, len(players) + 1)],
        "settings": {"optional_rules": rules},
        "actions": [dict(a) for a in rust.raw_actions],
        "rl18xx": {
            "source": "18xx.games site export (converted with the advisor's importer)",
            "site_game_id": game.get("id"),
            "site_action_ids": site_ids,
            "names": {str(k): v for k, v in names.items()},
            "site_result": game.get("result"),
            "site_status": game.get("status"),
            "finished": bool(rust.finished),
            "conversion_error": error,
        },
    }


def _groups_from_meta(meta: dict, num_players: int, names: dict) -> dict:
    sides = meta.get("sides") or meta.get("seats")
    if isinstance(sides, list) and len(sides) == num_players:
        return {i + 1: str(s) for i, s in enumerate(sides)}
    learner = meta.get("learner_seats")
    if isinstance(learner, list):
        return {i + 1: ("learner" if i in learner else "league") for i in range(num_players)}
    if names:
        return {pid: ("bot" if str(n).startswith("Bot") else "human") for pid, n in names.items()}
    return {}


def item_from_data(data, path: str, index_record: Optional[dict] = None, settings: Optional[dict] = None,
                   default_group: Optional[str] = None) -> GameItem:
    index_record = index_record or {}
    if isinstance(data, list):  # eval_head_to_head: a bare action list
        from rl18xx.agent.alphazero.game_records import _infer_num_players

        n = int(index_record.get("num_players") or len(index_record.get("seats") or []) or _infer_num_players(data))
        unlock = None
        if settings is not None:
            unlock = not bool(settings.get("start_positions")) and bool(settings.get("auction_unlock", True))
        return GameItem(path=path, actions=data, num_players=n, auction_unlock=unlock,
                        start=index_record.get("start") or ("auction" if unlock is not None and not settings.get(
                            "start_positions") else None),
                        groups=_groups_from_meta(index_record, n, {}), meta=index_record)
    if not isinstance(data, dict) or not isinstance(data.get("actions"), list):
        raise ValueError(f"{path}: not a saved game")
    if is_site_export(data):
        data = convert_site_export(data)
    meta = dict(data.get("rl18xx") or {})
    meta.update({k: v for k, v in index_record.items() if k not in meta})
    players = data.get("players") or []
    n = len(players) if players else int(meta.get("num_players") or 4)
    names = {int(k): v for k, v in (meta.get("names") or {}).items()}
    groups = _groups_from_meta(meta, n, names)
    if not groups and default_group:
        groups = {i: default_group for i in range(1, n + 1)}
    unlock = meta.get("auction_unlock")
    return GameItem(
        path=path,
        actions=data["actions"],
        num_players=n,
        optional_rules=[r for r in ((data.get("settings") or {}).get("optional_rules") or []) if r in ENGINE_RULES],
        auction_unlock=bool(unlock) if unlock is not None else None,
        start=meta.get("start"),
        groups=groups,
        names=names,
        site_ids=meta.get("site_action_ids"),
        finished_hint=data.get("finished") if "finished" in data else meta.get("finished"),
        meta={k: v for k, v in meta.items() if k not in ("site_action_ids",)},
    )


def _game_files(directory: Path) -> list:
    return sorted(p for p in directory.iterdir() if p.is_file() and (p.name.endswith(".json") or p.name.endswith(
        ".json.gz")))


def iter_sources(inputs, min_update: Optional[int] = None) -> Iterator[tuple]:
    """``(path, index record, collection settings)`` of every game file under ``inputs`` (cheap: nothing is read
    but the collections' indexes). ``min_update`` keeps policy-gradient games from that update on."""
    for inp in inputs:
        p = Path(inp)
        if p.is_dir() and (p / "games.jsonl").exists() and (p / "games").is_dir():
            settings = None
            if (p / "settings.json").exists():
                try:
                    settings = json.loads((p / "settings.json").read_text())
                except ValueError:
                    settings = None
            with open(p / "games.jsonl") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if min_update is not None and int(rec.get("update", min_update)) < min_update:
                        continue
                    name = rec.get("game_file") or (
                        f"games/{rec['game_idx']}.json" if "game_idx" in rec else None)
                    if not name or not (p / name).exists():
                        continue
                    yield str(p / name), rec, settings
        elif p.is_dir():
            for f in _game_files(p):
                yield str(f), {}, None
        elif p.exists():
            yield str(p), {}, None
        else:
            raise FileNotFoundError(inp)


def load_item(path: str, index_record: Optional[dict] = None, settings: Optional[dict] = None,
              default_group: Optional[str] = None) -> GameItem:
    return item_from_data(read_json(path), path, index_record, settings, default_group)
