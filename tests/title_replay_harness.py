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
from dataclasses import dataclass, field
from pathlib import Path

import engine_rs

from rl18xx.agent.alphazero.pretraining import filter_actions


class TitleUnsupported(Exception):
    pass


class ReplayError(Exception):
    def __init__(self, message: str, step: int, action: dict | None):
        super().__init__(message)
        self.step = step
        self.action = action


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


def replay_game(
    game: dict, max_actions: int | None = None, on_action=None
) -> ReplayReport:
    """Replay one recorded 18xx.games/fixture game dict through the Rust
    engine. Raises TitleUnsupported / ReplayError; returns a ReplayReport.

    ``on_action(rust, index, action)`` is called BEFORE each action is
    applied (after player-id renumbering) — used by per-action oracle
    checks such as the route-revenue cross-check.
    """
    title = game.get("title")
    if title not in supported_titles():
        raise TitleUnsupported(f"engine does not support title {title!r}")

    # Seat players in recorded order; renumber to 1..N (the engine convention
    # used by the 1830 cleaning pipeline).
    players = {i + 1: f"Player {i + 1}" for i in range(len(game["players"]))}
    player_mapping = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    id_to_name = {i + 1: p["name"] for i, p in enumerate(game["players"])}

    rust = engine_rs.BaseGame.new_titled(title, players)

    actions = filter_actions(game["actions"])
    if max_actions is not None:
        actions = actions[:max_actions]

    applied = 0
    for i, action in enumerate(actions):
        action = dict(action)
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
            raise ReplayError(
                f"action {i} ({action.get('type')}) rejected: {exc}", i, action
            ) from exc
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
                        f"action {i} auto[{j}]: unknown player entity"
                        f" {auto.get('entity')!r}",
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
                ) from exc
        applied += 1

    result = {
        id_to_name[pid]: cash for pid, cash in rust.result().items() if pid in id_to_name
    }
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
    recorded = {
        id_to_name.get(str(key), str(key)): int(score)
        for key, score in (game.get("result") or {}).items()
    }
    if recorded and report.result != recorded:
        raise AssertionError(
            f"fixture {path.name}: final scores diverge\n"
            f"  engine:   {dict(sorted(report.result.items()))}\n"
            f"  recorded: {dict(sorted(recorded.items()))}"
        )
    return report
