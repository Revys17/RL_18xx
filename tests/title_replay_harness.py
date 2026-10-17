"""Per-title replay harness — the validation oracle for non-1830 titles.

The 1830 methodology (dual-engine lockstep against the Python oracle) does
not scale to new titles: the Python engine stays 1830-only. New titles are
validated by replaying recorded games through the RUST engine alone
(see docs/multi_title_roadmap.md "Per-title validation strategy"):

  1. Ruby fixtures (tests/fixtures/<title>/*.json, vendored from
     tobymao/18xx public/fixtures) — the gold standard, few but exact:
     every action must be ACCEPTED, and the final scores must MATCH.
  2. 18xx.games human corpora (human_games/<title>/*.json) — volume:
     outcome-level assertions over thousands of games.

Both share the replay core in this module. ``replay_game`` raises
``ReplayError`` with the failing action index and engine error on any
rejection; otherwise returns the engine's final result() mapped back to
player names for comparison with the recorded result.

The harness is intentionally built BEFORE the mechanics (Phase 1 rule:
every new step lands against an executable oracle). While a title is not
yet registered in the engine, ``replay_game`` raises ``TitleUnsupported``
and the pytest wrappers skip.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import engine_rs


class TitleUnsupported(Exception):
    pass


class ReplayError(Exception):
    def __init__(self, message: str, step: int, action: dict | None, file_id: int | None = None):
        super().__init__(message)
        self.step = step
        self.action = action
        # The rejected action's id in the corpus FILE (the filters
        # renumber ids; Ruby's per-action dumps key by file ids).
        self.file_id = file_id


@dataclass
class ReplayReport:
    game_id: str
    title: str
    actions_total: int
    actions_applied: int
    # Engine final scores keyed by recorded player NAME.
    result: dict[str, int] = field(default_factory=dict)
    finished: bool = False


def supported_titles() -> set[str]:
    return set(engine_rs.supported_titles_py())


# Seat id base per title. The engine resolves an action's `entity` id
# against corporations first for titles with NUMERIC corporation syms (1822's
# minors are "1".."30"), so those titles seat players at 1001+ (the engine
# rejects colliding ids). Every other title keeps the 1830 convention 1..N.
SEAT_ID_BASE = {"1822": 1000}


def game_seed(game: dict) -> int:
    """Ruby `initialize_seed`: settings.seed, else the game id's first digit
    run (0 when neither exists)."""
    seed = (game.get("settings") or {}).get("seed")
    if seed not in (None, ""):
        return int(seed)
    m = re.search(r"\d+", str(game.get("id", "")))
    return int(m.group(0)) if m else 0


def ruby_filtered_actions(actions: list[dict], mark_auto: bool = False) -> list[dict]:
    """The action stream Ruby's ``Game.load`` processes, flattened.

    A port of ``Engine::Game::Base.filtered_actions`` (undo / redo; a
    ``message`` is never undone and never an undo target), then the
    processing order of ``Game#process_action`` while loading: each kept
    action, followed by its recorded ``auto_actions``. Messages, program
    actions (``program_*``), ``end_game`` and ``log`` change no rules state
    and are dropped AFTER their auto_actions are hoisted — the server
    records the auto-actions a chat message or a programmed pass triggered
    on that very action (59414 @30: a message carries the programmed pass
    of the player whose turn it was). ``mark_auto`` tags the hoisted
    auto-actions with ``_auto: True``. ``pretraining.filter_actions``, the
    1830 cleaning filter, drops messages with their auto_actions and stays
    as it is for the 1830 corpus.
    """
    active_undos: list[list[tuple[dict, int]]] = []
    kept: list[dict | None] = [None] * len(actions)
    for index, action in enumerate(actions):
        t = action.get("type")
        if t == "undo":
            target = action.get("action_id")
            if target is not None:
                if target == 0:
                    undo_to = 0
                else:
                    idx = next((i for i, a in enumerate(actions) if a.get("id") == target), None)
                    if idx is None:
                        raise ValueError(f"undo target {target} not found")
                    undo_to = idx + 1
            else:
                undo_to = next((i for i in range(index - 1, -1, -1) if kept[i] and kept[i]["type"] != "message"), 0)
            cleared = []
            for i in range(undo_to, index):
                a = kept[i]
                if a is None or a["type"] == "message":
                    continue
                kept[i] = None
                cleared.append((a, i))
            active_undos.append(cleared)
        elif t == "redo":
            if active_undos:
                for a, i in active_undos.pop():
                    kept[i] = a
        elif t == "message":
            kept[index] = action
        else:
            active_undos.clear()
            kept[index] = action
    out: list[dict] = []
    for a in kept:
        if a is None:
            continue
        t = a["type"]
        if t not in ("message", "end_game", "log") and not t.startswith("program_"):
            entry = {k: v for k, v in a.items() if k != "auto_actions"}
            out.append(entry)
        for au in a.get("auto_actions") or []:
            at = au.get("type", "")
            if at in ("message", "end_game", "log") or at.startswith("program_"):
                continue
            entry = dict(au)
            if mark_auto:
                entry["_auto"] = True
            out.append(entry)
    for i, a in enumerate(out):
        a["id"] = i + 1
    return out


def replay_game(game: dict, max_actions: int | None = None, on_action=None) -> ReplayReport:
    """Replay one recorded 18xx.games/fixture game dict through the Rust
    engine. Raises TitleUnsupported / ReplayError; returns a ReplayReport.

    ``on_action(rust, index, action)`` is called BEFORE each action is
    applied (after player-id renumbering) — used by per-action oracle
    checks such as the route-revenue cross-check.
    """
    title = game.get("title")
    if title not in supported_titles():
        raise TitleUnsupported(f"engine does not support title {title!r}")

    # Seat players in recorded order; renumber to base+1..base+N (base 0 —
    # the 1830 cleaning convention — except titles in SEAT_ID_BASE).
    base = SEAT_ID_BASE.get(title, 0)
    players = {base + i + 1: f"Player {i + 1}" for i in range(len(game["players"]))}
    player_mapping = {p["id"]: base + i + 1 for i, p in enumerate(game["players"])}
    id_to_name = {base + i + 1: p["name"] for i, p in enumerate(game["players"])}

    optional_rules = list((game.get("settings") or {}).get("optional_rules") or [])
    try:
        rust = engine_rs.BaseGame.new_titled(title, players, optional_rules or None, game_seed(game))
    except ValueError as exc:
        raise ReplayError(f"construction rejected: {exc}", 0, None) from exc

    raw = json.loads(json.dumps(game["actions"]))
    for a in raw:
        a["_fid"] = a.get("id")
        for au in a.get("auto_actions") or []:
            au["_fid"] = a.get("id")
    actions = ruby_filtered_actions(raw)
    if max_actions is not None:
        actions = actions[:max_actions]

    applied = 0
    fid = None
    for i, action in enumerate(actions):
        action = dict(action)
        fid = action.pop("_fid", fid)
        for au in action.get("auto_actions") or []:
            au.pop("_fid", None)
        if action.get("entity_type") == "player":
            mapped = player_mapping.get(action.get("entity"))
            if mapped is None:
                raise ReplayError(
                    f"action {i}: unknown player entity {action.get('entity')!r}",
                    i,
                    action,
                )
            action["entity"] = mapped
            action["user"] = mapped
        if on_action is not None:
            on_action(rust, i, action)
        try:
            rust.process_action(action)
        except BaseException as exc:  # incl. pyo3 PanicException
            raise ReplayError(f"action {i} ({action.get('type')}) rejected: {exc}", i, action, fid) from exc
        # Server-generated auto actions (programmed passes, 1867's
        # BuyCompanyPreloan auto-pass for loan-free corps, ...) are embedded
        # in their parent action and processed right after it — exactly how
        # Ruby replays them (base.rb process_action: `action.auto_actions
        # .each { process_single_action }`).
        for j, auto in enumerate(action.get("auto_actions") or []):
            auto = dict(auto)
            if auto.get("type") == "message":
                continue
            if auto.get("entity_type") == "player":
                mapped = player_mapping.get(auto.get("entity"))
                if mapped is None:
                    raise ReplayError(
                        f"action {i} auto[{j}]: unknown player entity" f" {auto.get('entity')!r}",
                        i,
                        auto,
                    )
                auto["entity"] = mapped
                auto["user"] = mapped
            try:
                rust.process_action(auto)
            except BaseException as exc:
                raise ReplayError(
                    f"action {i} auto[{j}] ({auto.get('type')}) rejected: {exc}",
                    i,
                    auto,
                    fid,
                ) from exc
        applied += 1

    result = {id_to_name[pid]: cash for pid, cash in rust.result().items() if pid in id_to_name}
    return ReplayReport(
        game_id=str(game.get("id", "?")),
        title=title,
        actions_total=len(actions),
        actions_applied=applied,
        result=result,
        finished=bool(rust.finished),
    )


def check_fixture(path: Path) -> ReplayReport:
    """Replay a vendored Ruby fixture and assert the recorded final scores."""
    game = json.loads(path.read_text())
    report = replay_game(game)
    # Fixture results key players by ID; the report keys by seat NAME
    # (identical for the hs_* fixtures, distinct for 21268).
    id_to_name = {str(p["id"]): p["name"] for p in game.get("players", [])}
    recorded = {id_to_name.get(str(key), str(key)): int(score) for key, score in (game.get("result") or {}).items()}
    if recorded and report.result != recorded:
        raise AssertionError(
            f"fixture {path.name}: final scores diverge\n"
            f"  engine:   {dict(sorted(report.result.items()))}\n"
            f"  recorded: {dict(sorted(recorded.items()))}"
        )
    return report
