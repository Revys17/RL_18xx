#!/usr/bin/env python3
"""Audit the full raw human game corpus end-to-end.

For each game in ``human_games/1830/``:
  1. Apply the cleaning pipeline (``get_game_object_for_game`` — filters
     for optional_rules, cross_player_buy_company, cross_president_buy_train,
     mh_out_of_turn, illegal_share_buy, mis-attributed corp actions).
  2. For games that survive the filter, replay the cleaned action stream
     through both Python and Rust engines, comparing state at every step.

This is the full-corpus counterpart to ``audit_rust_engine_corpus.py``,
which only processes the already-cleaned ``human_games/1830_clean/``
subset. It validates engine parity across all player counts (2..6),
not just the 4-player games that the prior cleaning round happened to
keep.

Output:
- ``docs/rust_engine_full_corpus_audit.csv`` (one row per game)
- ``docs/rust_engine_full_corpus_audit.md`` (summary)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from tqdm import tqdm

from engine_rs import BaseGame as RustGame
from rl18xx.agent.alphazero.pretraining import (
    filter_to_completed_games,
    get_game_object_for_game,
    load_games_from_json,
)
from rl18xx.game.gamemap import GameMap

# Reuse the per-step comparator + categorizers from the existing audit.
from scripts.audit_rust_engine_corpus import (  # type: ignore
    compare_state,
    _classify_divergence,
    _classify_rust_error,
)


@dataclass
class FullAuditResult:
    game_id: str
    num_players: int = 0
    status: str = ""  # 'perfect' | 'divergence' | 'dropped' | 'python_error' | 'rust_error' | 'cleaning_error'
    total_actions: int = 0
    num_matched: int = 0
    first_divergence_action_index: Optional[int] = None
    first_divergence_action_type: Optional[str] = None
    divergence_kind: str = ""
    drop_reason: str = ""
    first_error_msg: str = ""

    def to_row(self) -> dict:
        return {
            "game_id": self.game_id,
            "num_players": self.num_players,
            "status": self.status,
            "total_actions": self.total_actions,
            "num_matched": self.num_matched,
            "first_divergence_action_index": (
                self.first_divergence_action_index
                if self.first_divergence_action_index is not None
                else ""
            ),
            "first_divergence_action_type": self.first_divergence_action_type or "",
            "divergence_kind": self.divergence_kind,
            "drop_reason": self.drop_reason,
            "first_error_msg": self.first_error_msg[:300],
        }


_DROP_KEYWORDS = {
    "optional_rules": "optional_rules",
    "cross-player company purchase": "cross_player_buy_company",
    "cross-player train purchase": "cross_president_buy_train",
    "the MH took an action out of turn": "mh_out_of_turn",
    "illegal share": "illegal_share_buy",
    "action entity does not match current operator": "mis_attributed_corp_action",
}


def _classify_cleaning_drop(captured_log: str) -> str:
    for kw, reason in _DROP_KEYWORDS.items():
        if kw in captured_log:
            return reason
    return "other"


class _LogCapture:
    """Stash debug-level cleaning-filter logs so we can classify drops."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __enter__(self):
        self.handler = logging.Handler(level=logging.DEBUG)
        self.handler.emit = lambda record: self.lines.append(record.getMessage())
        logger = logging.getLogger("rl18xx.agent.alphazero.pretraining")
        self._prior_level = logger.level
        self._prior_disabled = logger.disabled
        logger.setLevel(logging.DEBUG)
        logger.disabled = False
        logger.addHandler(self.handler)
        # The top-level disable was set at module import; lift it just for this block.
        self._prior_root_disable = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        return self

    def __exit__(self, *exc):
        logger = logging.getLogger("rl18xx.agent.alphazero.pretraining")
        logger.removeHandler(self.handler)
        logger.setLevel(self._prior_level)
        logger.disabled = self._prior_disabled
        logging.disable(self._prior_root_disable)


def audit_one_raw_game(raw_path: Path, game_cls) -> FullAuditResult:
    game_id = raw_path.stem
    try:
        raw = json.load(open(raw_path))
    except Exception as e:
        return FullAuditResult(
            game_id=game_id,
            status="audit_error",
            first_error_msg=f"failed to load raw JSON: {e}",
        )

    num_players = len(raw.get("players", []) or [])

    if raw.get("status") != "finished":
        return FullAuditResult(
            game_id=game_id,
            num_players=num_players,
            status="dropped",
            drop_reason="not_finished",
        )

    # 1) Run cleaning pipeline
    try:
        with _LogCapture() as cap:
            cleaned_game = get_game_object_for_game(raw)
        log_text = "\n".join(cap.lines)
    except Exception as e:
        return FullAuditResult(
            game_id=game_id,
            num_players=num_players,
            status="cleaning_error",
            first_error_msg=f"{type(e).__name__}: {e}",
        )

    if cleaned_game is None:
        return FullAuditResult(
            game_id=game_id,
            num_players=num_players,
            status="dropped",
            drop_reason=_classify_cleaning_drop(log_text),
        )

    # 2) Extract cleaned actions and replay through fresh engines
    cleaned_dict = cleaned_game.to_dict()
    actions = cleaned_dict.get("actions") or []
    names = {p["id"]: p["name"] for p in cleaned_dict["players"]}

    try:
        py_game = game_cls(names)
        rust_game = RustGame(names)
    except Exception as e:
        return FullAuditResult(
            game_id=game_id,
            num_players=num_players,
            status="audit_error",
            total_actions=len(actions),
            first_error_msg=f"engine init failed: {e}",
        )

    num_matched = 0
    for i, action in enumerate(actions):
        action_type = action.get("type", "?")
        try:
            py_game = py_game.process_action(action)
        except Exception as e:
            return FullAuditResult(
                game_id=game_id,
                num_players=num_players,
                status="python_error",
                total_actions=len(actions),
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind="python_error",
                first_error_msg=f"{type(e).__name__}: {e}",
            )

        try:
            rust_game.process_action(action)
        except Exception as e:
            return FullAuditResult(
                game_id=game_id,
                num_players=num_players,
                status="rust_error",
                total_actions=len(actions),
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind=_classify_rust_error(e),
                first_error_msg=f"{type(e).__name__}: {e}",
            )

        try:
            errors = compare_state(rust_game, py_game)
        except Exception as e:
            return FullAuditResult(
                game_id=game_id,
                num_players=num_players,
                status="audit_error",
                total_actions=len(actions),
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind="compare_state_error",
                first_error_msg=f"{type(e).__name__}: {e}",
            )

        if errors:
            return FullAuditResult(
                game_id=game_id,
                num_players=num_players,
                status="divergence",
                total_actions=len(actions),
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind=_classify_divergence(errors[0]),
                first_error_msg=errors[0],
            )

        num_matched += 1

    return FullAuditResult(
        game_id=game_id,
        num_players=num_players,
        status="perfect",
        total_actions=len(actions),
        num_matched=num_matched,
    )


def write_csv(results: list[FullAuditResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("")
        return
    fields = list(results[0].to_row().keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())


def write_markdown(results: list[FullAuditResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    by_status = Counter(r.status for r in results)
    by_player_status = Counter((r.num_players, r.status) for r in results)
    by_player = Counter(r.num_players for r in results)
    div_kinds = Counter(r.divergence_kind for r in results if r.divergence_kind and r.status != "dropped")
    drops = Counter(r.drop_reason for r in results if r.status == "dropped")
    div_actions = Counter(
        r.first_divergence_action_type for r in results if r.first_divergence_action_type
    )

    def pct(n_: int) -> str:
        return f"{100 * n_ / n:.2f}%" if n else "0%"

    lines = []
    lines.append("# Rust engine full-corpus audit")
    lines.append("")
    lines.append(
        f"Audited **{n}** raw games from `human_games/1830/` through the full cleaning + "
        "engine-parity pipeline."
    )
    lines.append("")
    lines.append("## Headline stats")
    lines.append("")
    for s in ("perfect", "dropped", "divergence", "rust_error", "python_error", "cleaning_error", "audit_error"):
        if by_status.get(s, 0):
            lines.append(f"- {s}: **{by_status[s]} / {n}** ({pct(by_status[s])})")
    lines.append("")
    lines.append("## By player count")
    lines.append("")
    lines.append("| Players | Total | Perfect | Dropped | Diverged | Other |")
    lines.append("|---|---|---|---|---|---|")
    for pc in sorted(by_player):
        total = by_player[pc]
        perfect = by_player_status.get((pc, "perfect"), 0)
        dropped = by_player_status.get((pc, "dropped"), 0)
        diverged = sum(
            by_player_status.get((pc, s), 0)
            for s in ("divergence", "rust_error", "python_error")
        )
        other = total - perfect - dropped - diverged
        lines.append(f"| {pc} | {total} | {perfect} | {dropped} | {diverged} | {other} |")
    lines.append("")
    if div_kinds:
        lines.append("## Top divergence categories")
        lines.append("")
        for kind, c in div_kinds.most_common():
            lines.append(f"- `{kind}`: **{c}**")
        lines.append("")
    if div_actions:
        lines.append("## Top first-divergence action types")
        lines.append("")
        for at, c in div_actions.most_common(10):
            lines.append(f"- `{at}`: **{c}**")
        lines.append("")
    if drops:
        lines.append("## Drop reasons")
        lines.append("")
        for reason, c in drops.most_common():
            lines.append(f"- `{reason}`: **{c}**")
        lines.append("")

    dropped = by_status.get("dropped", 0)
    eligible = n - dropped
    if eligible:
        perfect = by_status.get("perfect", 0)
        lines.append("## Replay-survivors-only view")
        lines.append("")
        lines.append(f"- Replay-eligible (status != dropped): **{eligible}**")
        lines.append(
            f"- Perfect among survivors: **{perfect} / {eligible}** "
            f"({100 * perfect / eligible:.2f}%)"
        )
        lines.append(
            f"- Non-perfect among survivors: **{eligible - perfect} / {eligible}**"
        )
        lines.append("")
    lines.append("Per-game rows: `docs/rust_engine_full_corpus_audit.csv`")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_dir",
        nargs="?",
        type=str,
        default=str(REPO_ROOT / "human_games" / "1830"),
        help="Directory of raw human-game JSONs",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=str(REPO_ROOT / "docs" / "rust_engine_full_corpus_audit.csv"),
    )
    parser.add_argument(
        "--md-out",
        type=str,
        default=str(REPO_ROOT / "docs" / "rust_engine_full_corpus_audit.md"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of games for testing (0 = all)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    csv_path = Path(args.csv_out)
    md_path = Path(args.md_out)

    game_cls = GameMap().game_by_title("1830")

    # Load + filter to completed games (any player count 2-6).
    games = load_games_from_json(str(raw_dir))
    games = filter_to_completed_games(games)

    if args.limit:
        games = games[: args.limit]

    print(f"Auditing {len(games)} completed raw games from {raw_dir}")
    print(f"  csv out : {csv_path}")
    print(f"  md  out : {md_path}")

    # We can't iterate by file because filter_to_completed_games returns
    # the in-memory dicts. Process each in-place.
    results: list[FullAuditResult] = []
    for game in tqdm(games, desc="audit"):
        # Synthesize a temporary "raw_path" for filename consistency.
        # The id is set by load_games_from_json from the file stem.
        game_id = str(game["id"])
        raw_path = raw_dir / f"{game_id}.json"

        # Inline the per-game logic so we don't re-read the file.
        try:
            with _LogCapture() as cap:
                cleaned_game = get_game_object_for_game(game)
            log_text = "\n".join(cap.lines)
        except Exception as e:
            results.append(
                FullAuditResult(
                    game_id=game_id,
                    num_players=len(game.get("players", []) or []),
                    status="cleaning_error",
                    first_error_msg=f"{type(e).__name__}: {e}",
                )
            )
            continue

        num_players = len(game.get("players", []) or [])

        if cleaned_game is None:
            results.append(
                FullAuditResult(
                    game_id=game_id,
                    num_players=num_players,
                    status="dropped",
                    drop_reason=_classify_cleaning_drop(log_text),
                )
            )
            continue

        # Replay cleaned actions through fresh engines + state-compare.
        cleaned_dict = cleaned_game.to_dict()
        actions = cleaned_dict.get("actions") or []
        names = {p["id"]: p["name"] for p in cleaned_dict["players"]}

        try:
            py_game = game_cls(names)
            rust_game = RustGame(names)
        except Exception as e:
            results.append(
                FullAuditResult(
                    game_id=game_id,
                    num_players=num_players,
                    status="audit_error",
                    total_actions=len(actions),
                    first_error_msg=f"engine init failed: {e}",
                )
            )
            continue

        num_matched = 0
        outcome: Optional[FullAuditResult] = None
        for i, action in enumerate(actions):
            action_type = action.get("type", "?")
            try:
                py_game = py_game.process_action(action)
            except Exception as e:
                outcome = FullAuditResult(
                    game_id=game_id, num_players=num_players, status="python_error",
                    total_actions=len(actions), num_matched=num_matched,
                    first_divergence_action_index=i, first_divergence_action_type=action_type,
                    divergence_kind="python_error", first_error_msg=f"{type(e).__name__}: {e}",
                )
                break
            try:
                rust_game.process_action(action)
            except Exception as e:
                outcome = FullAuditResult(
                    game_id=game_id, num_players=num_players, status="rust_error",
                    total_actions=len(actions), num_matched=num_matched,
                    first_divergence_action_index=i, first_divergence_action_type=action_type,
                    divergence_kind=_classify_rust_error(e), first_error_msg=f"{type(e).__name__}: {e}",
                )
                break
            try:
                errors = compare_state(rust_game, py_game)
            except Exception as e:
                outcome = FullAuditResult(
                    game_id=game_id, num_players=num_players, status="audit_error",
                    total_actions=len(actions), num_matched=num_matched,
                    first_divergence_action_index=i, first_divergence_action_type=action_type,
                    divergence_kind="compare_state_error", first_error_msg=f"{type(e).__name__}: {e}",
                )
                break
            if errors:
                outcome = FullAuditResult(
                    game_id=game_id, num_players=num_players, status="divergence",
                    total_actions=len(actions), num_matched=num_matched,
                    first_divergence_action_index=i, first_divergence_action_type=action_type,
                    divergence_kind=_classify_divergence(errors[0]), first_error_msg=errors[0],
                )
                break
            num_matched += 1

        if outcome is None:
            outcome = FullAuditResult(
                game_id=game_id, num_players=num_players, status="perfect",
                total_actions=len(actions), num_matched=num_matched,
            )
        results.append(outcome)

    write_csv(results, csv_path)
    write_markdown(results, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
