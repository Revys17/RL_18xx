"""Audit the Rust engine against the Python engine across the full human
1830 corpus.

For every game in ``human_games/1830_clean/`` we:

1. Load the cleaned action stream (already produced by
   ``pretraining.fix_online_games`` -> ``get_game_object_for_game``).
   Games marked ``status == "error"`` were dropped by the cleaning pass;
   for those we reload the raw 1830 JSON and re-run the cleaning checks
   to figure out *why* (cross-player buy_company / cross-president buy_train
   / MH out of turn / illegal share buy / optional-rules failure).
2. Replay the cleaned actions through both Python and Rust engines in
   lockstep, comparing state after each action via
   ``tests/validate_rust_engine.compare_state``.
3. Categorize the first observed divergence into a short categorical
   label so we can summarize the kinds of bugs we're hitting.
4. Write a per-game CSV row and a summary markdown.

The audit does not modify either engine. Games that crash the audit
(Python or Rust exception during replay) are recorded with a
``python_error`` / ``rust_error`` divergence kind so the script never
silently drops data.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from tests.validate_rust_engine import compare_state
from engine_rs import BaseGame as RustGame
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.pretraining import (
    filter_actions,
    should_add_pass,
    should_skip_action,
)
from rl18xx.game.engine.round import BuySellParShares


# --------------------------------------------------------------------------- #
# Drop-reason categorization (reruns the raw game through the same checks
# as get_game_object_for_game, but records which check fired instead of
# returning None).
# --------------------------------------------------------------------------- #


def _categorize_drop_reason(raw_game: dict) -> str:
    """Re-run the cleaning machinery to determine why a game was dropped.

    Returns one of:
      - 'optional_rules'             (Python engine threw with optional_rules set)
      - 'cross_player_buy_company'
      - 'cross_president_buy_train'
      - 'mh_out_of_turn'
      - 'illegal_share_buy'
      - 'python_error'               (engine threw without optional_rules)
      - 'unknown_drop'               (no filter fired but clean dir marked error)
    """
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    num_players = len(raw_game["players"])
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
    game_state = game_class(players)
    player_mapping = {p["id"]: i + 1 for i, p in enumerate(raw_game["players"])}
    optional_rules = bool(raw_game.get("settings", {}).get("optional_rules"))

    try:
        filtered_actions = filter_actions(raw_game["actions"])
    except Exception:
        return "filter_actions_error"

    for i, action in enumerate(filtered_actions):
        action = dict(action)  # local copy; we may rewrite entity below

        if action["entity_type"] == "player":
            try:
                action["entity"] = player_mapping[action["entity"]]
            except KeyError:
                return "unknown_player"
            action["user"] = action["entity"]
        else:
            try:
                entity_obj = game_state.get(action["entity_type"], action["entity"])
                entity_owner = entity_obj.player()
            except Exception:
                # Engine can't resolve entity — counts as a structural problem
                return "entity_resolution_error"
            if action.get("user", None):
                if action["user"] in player_mapping:
                    action["user"] = entity_owner.id

        # Filter triggers — order mirrors get_game_object_for_game
        if action["type"] == "buy_company":
            try:
                purchaser = game_state.get(action["entity_type"], action["entity"]).player()
                owner = game_state.company_by_id(action["company"]).player()
                if purchaser.id != owner.id:
                    return "cross_player_buy_company"
            except Exception:
                return "buy_company_lookup_error"

        if action["type"] == "buy_train":
            try:
                train_purchaser = game_state.get(action["entity_type"], action["entity"])
                train_owner = game_state.train_by_id(action["train"]).owner
                if train_owner.is_corporation():
                    if train_purchaser.player() != train_owner.player():
                        return "cross_president_buy_train"
            except Exception:
                return "buy_train_lookup_error"

        if action["entity_type"] == "company" and action["entity"] == "MH":
            try:
                mh_owner = game_state.company_by_id("MH").player()
                current_player = game_state.current_entity.player()
                if mh_owner != current_player:
                    return "mh_out_of_turn"
            except Exception:
                pass

        try:
            if should_add_pass(action, game_state):
                pass_action = {
                    "type": "pass",
                    "entity": game_state.current_entity.id,
                    "entity_type": game_state.current_entity.__class__.__name__.lower(),
                    "user": game_state.current_entity.player().id,
                }
                game_state.process_action(pass_action)

            if should_skip_action(filtered_actions, action, game_state, i):
                continue

            if (
                isinstance(game_state.round.active_step(), BuySellParShares)
                and action["type"] == "buy_shares"
            ):
                shares = [game_state.share_by_id(s) for s in action["shares"]]
                if not game_state.round.active_step().can_buy_shares(
                    game_state.current_entity, shares
                ):
                    return "illegal_share_buy"

            game_state.process_action(action)
        except Exception:
            if optional_rules:
                return "optional_rules"
            return "python_error"

    # No filter fired but the clean-dir wrote status=error. Most likely the
    # action stream finished without raising (e.g. game never produced a
    # divergence and was actually fine but got mis-stubbed) — rare.
    return "unknown_drop"


# --------------------------------------------------------------------------- #
# Divergence categorization for the side-by-side replay.
# --------------------------------------------------------------------------- #


def _classify_divergence(error_msg: str) -> str:
    """Map a compare_state error string to a short categorical label."""
    m = error_msg.lower()
    if "share_price coords" in m:
        return "share_price_coord"
    if "share_price" in m:
        return "share_price_value"
    if "shares:" in m:
        return "share_ownership"
    if "president" in m:
        return "president"
    if "floated" in m:
        return "floated_mismatch"
    if "ipoed" in m:
        return "ipoed_mismatch"
    if "par_price" in m:
        return "par_price"
    if "cash:" in m and "rust=" in m and ("co" in m or "corp" in m or any(c.isalpha() for c in m.split("cash:")[0][-4:])):
        # corp cash lines look like "PRR cash: Rust=X Python=Y"
        return "corp_cash"
    if m.startswith("player ") and "rust=$" in m:
        return "player_cash"
    if "bank:" in m:
        return "bank_cash"
    if "or step:" in m:
        return "or_step_mismatch"
    if "current entity" in m:
        return "current_entity"
    if "round type" in m:
        return "round_type"
    if "round num" in m:
        return "round_num"
    if "phase:" in m:
        return "phase_mismatch"
    if "depot:" in m:
        return "depot_mismatch"
    if "trains:" in m:
        return "corp_trains"
    if "tokens:" in m:
        return "corp_tokens"
    if "tokenable_cities" in m:
        return "tokenable_cities"
    if "connected_nodes" in m:
        return "graph_connectivity"
    if "tile:" in m:
        return "hex_tile"
    if "company" in m and "closed" in m:
        return "company_closed"
    if "finished:" in m:
        return "finished_mismatch"
    return "other"


def _classify_rust_error(err: Exception) -> str:
    return _classify_rust_error_msg(str(err))


def _classify_rust_error_msg(s: str) -> str:
    s = s.lower()
    # More specific patterns first
    if "discard_train" in s or "discard train" in s:
        return "rust_rejected_discard_train"
    if "forced train buy" in s or "forced buy" in s or "cannot afford" in s:
        return "rust_rejected_forced_train_buy"
    if "not in laytile step" in s or "lay_tile" in s or "laytile" in s:
        return "rust_rejected_lay_tile"
    if "buytrain" in s or "buy_train" in s:
        return "rust_rejected_buy_train"
    if "buyshares" in s or "buy_shares" in s:
        return "rust_rejected_buy_shares"
    if "placetoken" in s or "place_token" in s:
        return "rust_rejected_place_token"
    if "runroutes" in s or "run_routes" in s:
        return "rust_rejected_run_routes"
    if "buy_company" in s:
        return "rust_rejected_buy_company"
    if "dividend" in s:
        return "rust_rejected_dividend"
    if "bid" in s:
        return "rust_rejected_bid"
    if " par " in s or "par price" in s:
        return "rust_rejected_par"
    if "pass" in s:
        return "rust_rejected_pass"
    return "rust_error"


# --------------------------------------------------------------------------- #
# Per-game audit
# --------------------------------------------------------------------------- #


@dataclass
class GameAuditResult:
    game_id: str
    status: str  # 'perfect' | 'divergence' | 'dropped' | 'python_error' | 'rust_error' | 'audit_error'
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


def audit_one_game(clean_path: Path, raw_dir: Path) -> GameAuditResult:
    game_id = clean_path.stem
    try:
        clean = json.load(open(clean_path))
    except Exception as e:
        return GameAuditResult(
            game_id=game_id,
            status="audit_error",
            first_error_msg=f"failed to load clean JSON: {e}",
        )

    # Dropped by filter
    if clean.get("status") == "error":
        reason = "unknown_drop"
        raw_path = raw_dir / f"{game_id}.json"
        if raw_path.exists():
            try:
                raw = json.load(open(raw_path))
                reason = _categorize_drop_reason(raw)
            except Exception as e:
                reason = f"audit_error: {e}"[:80]
        return GameAuditResult(
            game_id=game_id,
            status="dropped",
            drop_reason=reason,
        )

    actions = clean.get("actions") or []
    total = len(actions)

    # Build engines. The clean game's player list is already mapped to 1..N.
    names = {p["id"]: p["name"] for p in clean["players"]}
    try:
        game_cls = GameMap().game_by_title("1830")
        py_game = game_cls(names)
        rust_game = RustGame(names)
    except Exception as e:
        return GameAuditResult(
            game_id=game_id,
            status="audit_error",
            total_actions=total,
            first_error_msg=f"engine init failed: {e}",
        )

    num_matched = 0
    for i, action in enumerate(actions):
        action_type = action.get("type", "?")

        # Python step
        try:
            py_game = py_game.process_action(action)
        except Exception as e:
            return GameAuditResult(
                game_id=game_id,
                status="python_error",
                total_actions=total,
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind="python_error",
                first_error_msg=f"{type(e).__name__}: {e}",
            )

        # Rust step
        try:
            rust_game.process_action(action)
        except Exception as e:
            return GameAuditResult(
                game_id=game_id,
                status="rust_error",
                total_actions=total,
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind=_classify_rust_error(e),
                first_error_msg=f"{type(e).__name__}: {e}",
            )

        # State comparison
        try:
            errors = compare_state(rust_game, py_game)
        except Exception as e:
            return GameAuditResult(
                game_id=game_id,
                status="audit_error",
                total_actions=total,
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind="compare_state_error",
                first_error_msg=f"{type(e).__name__}: {e}",
            )

        if errors:
            return GameAuditResult(
                game_id=game_id,
                status="divergence",
                total_actions=total,
                num_matched=num_matched,
                first_divergence_action_index=i,
                first_divergence_action_type=action_type,
                divergence_kind=_classify_divergence(errors[0]),
                first_error_msg=errors[0],
            )

        num_matched += 1

    return GameAuditResult(
        game_id=game_id,
        status="perfect",
        total_actions=total,
        num_matched=num_matched,
    )


# --------------------------------------------------------------------------- #
# Aggregate + report
# --------------------------------------------------------------------------- #


def write_csv(results: list[GameAuditResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_id",
        "status",
        "total_actions",
        "num_matched",
        "first_divergence_action_index",
        "first_divergence_action_type",
        "divergence_kind",
        "drop_reason",
        "first_error_msg",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())


def write_summary(results: list[GameAuditResult], path: Path, csv_path: Path) -> dict:
    total = len(results)
    by_status: Counter = Counter(r.status for r in results)
    perfect = by_status.get("perfect", 0)
    divergence = by_status.get("divergence", 0)
    dropped = by_status.get("dropped", 0)
    rust_err = by_status.get("rust_error", 0)
    py_err = by_status.get("python_error", 0)
    audit_err = by_status.get("audit_error", 0)

    diverging_results = [
        r for r in results if r.status in ("divergence", "rust_error", "python_error")
    ]
    div_kinds = Counter(r.divergence_kind for r in diverging_results)
    first_action_types = Counter(
        r.first_divergence_action_type
        for r in diverging_results
        if r.first_divergence_action_type
    )
    drop_reasons = Counter(r.drop_reason for r in results if r.status == "dropped")

    def pct(n: int) -> str:
        if total == 0:
            return "0.00%"
        return f"{100.0 * n / total:.2f}%"

    lines: list[str] = []
    lines.append("# Rust engine corpus audit\n")
    lines.append(
        f"Audited **{total}** games from `human_games/1830_clean/` "
        f"against the Python and Rust engines side-by-side.\n"
    )
    lines.append("## Headline stats\n")
    lines.append(f"- Perfect (zero state divergence): **{perfect} / {total}** ({pct(perfect)})")
    lines.append(f"- Dropped by pretraining filters: **{dropped} / {total}** ({pct(dropped)})")
    lines.append(
        f"- Divergence in state compare: **{divergence} / {total}** ({pct(divergence)})"
    )
    lines.append(
        f"- Rust engine rejected an action: **{rust_err} / {total}** ({pct(rust_err)})"
    )
    lines.append(
        f"- Python engine raised during replay: **{py_err} / {total}** ({pct(py_err)})"
    )
    lines.append(f"- Audit harness error: **{audit_err} / {total}** ({pct(audit_err)})\n")

    lines.append("## Top divergence categories (across divergence + rust_error + python_error)\n")
    if div_kinds:
        for kind, count in div_kinds.most_common(10):
            lines.append(f"- `{kind}`: **{count}**")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Top first-divergence action types\n")
    if first_action_types:
        for atype, count in first_action_types.most_common(10):
            lines.append(f"- `{atype}`: **{count}**")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Drop reasons (for status=dropped)\n")
    if drop_reasons:
        for reason, count in drop_reasons.most_common():
            lines.append(f"- `{reason}`: **{count}**")
    else:
        lines.append("- (none)")
    lines.append("")

    # Slice divergence-only stats so they're easy to reason about separately.
    lines.append("## Replay-survivors-only view\n")
    survivors = total - dropped
    lines.append(f"- Replay-eligible games (status != dropped): **{survivors}**")
    if survivors:
        s_perfect = perfect
        s_diverge = divergence + rust_err + py_err
        lines.append(
            f"- Perfect among survivors: **{s_perfect} / {survivors}** "
            f"({100.0 * s_perfect / survivors:.2f}%)"
        )
        lines.append(
            f"- Divergence among survivors: **{s_diverge} / {survivors}** "
            f"({100.0 * s_diverge / survivors:.2f}%)"
        )
    lines.append("")

    lines.append(f"Per-game rows: `{csv_path.relative_to(REPO_ROOT)}`")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))

    return {
        "total": total,
        "perfect": perfect,
        "divergence": divergence,
        "dropped": dropped,
        "rust_error": rust_err,
        "python_error": py_err,
        "audit_error": audit_err,
        "top_divergence_kinds": div_kinds.most_common(5),
        "top_first_action_types": first_action_types.most_common(3),
        "top_drop_reasons": drop_reasons.most_common(5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "clean_dir",
        nargs="?",
        default=str(REPO_ROOT / "human_games" / "1830_clean"),
        help="Directory of cleaned game JSON files",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(REPO_ROOT / "human_games" / "1830"),
        help="Directory of raw (pre-cleaning) game JSON files, used to "
        "categorize drop reasons",
    )
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N games")
    parser.add_argument(
        "--csv-out",
        default=str(REPO_ROOT / "docs" / "rust_engine_corpus_audit.csv"),
    )
    parser.add_argument(
        "--md-out",
        default=str(REPO_ROOT / "docs" / "rust_engine_corpus_audit.md"),
    )
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    raw_dir = Path(args.raw_dir)
    csv_path = Path(args.csv_out)
    md_path = Path(args.md_out)

    paths = sorted(clean_dir.glob("*.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"No JSON files found in {clean_dir}", file=sys.stderr)
        return 1

    print(f"Auditing {len(paths)} games from {clean_dir}")
    print(f"  raw dir : {raw_dir}")
    print(f"  csv out : {csv_path}")
    print(f"  md out  : {md_path}")

    results: list[GameAuditResult] = []
    for p in tqdm(paths, desc="audit"):
        try:
            r = audit_one_game(p, raw_dir)
        except Exception as e:
            tb = traceback.format_exc()
            r = GameAuditResult(
                game_id=p.stem,
                status="audit_error",
                divergence_kind="harness_crash",
                first_error_msg=f"{type(e).__name__}: {e}\n{tb}"[:500],
            )
        results.append(r)

    write_csv(results, csv_path)
    summary = write_summary(results, md_path, csv_path)

    print("\n=== Audit summary ===")
    print(f"  total     : {summary['total']}")
    print(f"  perfect   : {summary['perfect']}")
    print(f"  divergence: {summary['divergence']}")
    print(f"  dropped   : {summary['dropped']}")
    print(f"  rust_error: {summary['rust_error']}")
    print(f"  py_error  : {summary['python_error']}")
    print(f"  audit_err : {summary['audit_error']}")
    print(f"  top divergence kinds: {summary['top_divergence_kinds']}")
    print(f"  top first action types: {summary['top_first_action_types']}")
    print(f"  top drop reasons: {summary['top_drop_reasons']}")
    print(f"\nCSV : {csv_path}")
    print(f"MD  : {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
