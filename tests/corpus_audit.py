"""Human-corpus outcome audit for any non-1830 title (validation leg 2).

Replays finished 18xx.games games (human_games/<title>/*.json and the
scrape folders) through the Rust engine and asserts outcome (final
``result()``) against the recorded scores — outcome-level assertions over
thousands of games.

Usage:
    uv run python tests/corpus_audit.py --title 1822 \\
        [--glob 'human_games/scrape_2026_10/1822/*.json'] [--limit N]
        [--json /tmp/audit.json] [--ruby-results tests/g1822_ruby_head_results.json]

Buckets:
  outcome_match       full replay, exact recorded scores
  recorded_drift      full replay, scores != record but == Ruby HEAD's full
                      replay (``--ruby-results``: game id -> Ruby HEAD scores
                      by seat, from scripts/ruby_diff.py --results-json)
  result_mismatch     full replay, scores != record (not Ruby-verified)
  rejected            replay stops on a rejected action
  ruby_rejects_same   rejected, and Ruby HEAD rejects the same file action
                      with no earlier per-action state divergence (an
                      old-rules record; ``--ruby-results`` from
                      scripts/ruby_diff.py --results-json)
  unsupported_rules   construction rejected (optional rules the title does
                      not implement, player count) — not an engine bucket
  not_finished        corpus file status active/new — never replayed
  title_mismatch      the file is another title (the old 1822 folder mixes
                      variants)

Exit codes: 0 = every replayed game matched (drops are reported, not fatal);
1 = at least one mismatch/rejection; 2 = title unsupported.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.title_replay_harness import ReplayError, TitleUnsupported, replay_game


def align_recorded_keys(
    recorded: dict[str, int], game: dict, engine_names: set[str]
) -> dict[str, int]:
    """Re-key the file's ``result`` onto the seat names the harness reports.

    Corpus files key ``result`` by player NAME as of the final snapshot;
    the ``players`` list (which the harness maps seats from) can carry a
    LATER rename. Align by player ID when the key is numeric (fixture
    convention), exact name otherwise; a single leftover key on each side
    is then paired — score equality is still required afterwards, so the
    pairing can only un-flag a spurious mismatch, never mask a real one.
    """
    id_to_name = {str(p["id"]): p["name"] for p in game.get("players", [])}
    out: dict[str, int] = {}
    leftovers: dict[str, int] = {}
    for key, score in recorded.items():
        if key in id_to_name:
            out[id_to_name[key]] = score
        elif key in engine_names:
            out[key] = score
        else:
            leftovers[key] = score
    unclaimed = engine_names - set(out)
    if len(leftovers) == 1 and len(unclaimed) == 1:
        out[next(iter(unclaimed))] = next(iter(leftovers.values()))
    else:
        out.update(leftovers)
    return out


def audit_one(
    title: str, path: str, ruby_head: dict, skip_optional_rules: bool
) -> tuple[str, str | None, dict | None]:
    """Audit one corpus file -> (bucket, by-player-count key, failure record)."""
    game_id = Path(path).stem
    try:
        game = json.load(open(path))
    except Exception as exc:
        return "unparseable", None, {"id": game_id, "kind": "unparseable", "reason": str(exc)}
    if game.get("title") != title:
        return "title_mismatch", None, None
    if game.get("status") not in ("finished", "archived"):
        return "not_finished", None, None
    if skip_optional_rules and game.get("settings", {}).get("optional_rules"):
        return "optional_rules_skipped", None, None
    n_players = len(game.get("players", []))
    try:
        report = replay_game(game)
    except TitleUnsupported:
        raise
    except ReplayError as exc:
        if exc.action is None and "construction rejected" in str(exc):
            return "unsupported_rules", None, {"id": game_id, "kind": "unsupported_rules", "reason": str(exc)[:200]}
        ruby = ruby_head.get(game_id) or {}
        kind = (
            "ruby_rejects_same"
            if ruby.get("ruby_rejects_at") is not None
            and ruby.get("ruby_rejects_at") == exc.file_id
            and ruby.get("diverges_at") is None
            else "rejected"
        )
        return (
            kind,
            f"{n_players}p {kind}",
            {
                "id": game_id,
                "kind": kind,
                "players": n_players,
                "step": exc.step,
                "file_id": exc.file_id,
                "action": exc.action,
                "reason": str(exc)[:300],
            },
        )
    recorded = {k: int(v) for k, v in (game.get("result") or {}).items()}
    recorded = align_recorded_keys(recorded, game, set(report.result))
    if recorded and report.result != recorded:
        names = [p["name"] for p in game.get("players", [])]
        by_seat = {str(names.index(n) + 1): v for n, v in report.result.items() if n in names}
        ruby = ruby_head.get(game_id)
        kind = (
            "recorded_drift"
            if ruby is not None and ruby.get("scores_by_seat") == by_seat
            else "result_mismatch"
        )
        # Aggregates and ids only: scores keyed by seat, never by name.
        rec_by_seat = {str(names.index(n) + 1): v for n, v in recorded.items() if n in names}
        return (
            kind,
            f"{n_players}p {kind}",
            {"id": game_id, "kind": kind, "players": n_players, "engine": by_seat, "recorded": rec_by_seat},
        )
    return "outcome_match", f"{n_players}p outcome_match", None


def _audit_star(args):
    return audit_one(*args)


def audit(
    title: str,
    paths: list[str],
    ruby_head: dict,
    skip_optional_rules: bool,
    jobs: int = 1,
) -> tuple[Counter, list[dict], Counter]:
    counts: Counter[str] = Counter()
    by_players: Counter[str] = Counter()
    failures: list[dict] = []
    work = [(title, p, ruby_head, skip_optional_rules) for p in paths]
    if jobs > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(jobs) as pool:
            results = pool.map(_audit_star, work, chunksize=4)
    else:
        results = map(_audit_star, work)
    for kind, bp, failure in results:
        counts[kind] += 1
        if bp:
            by_players[bp] += 1
        if failure:
            failures.append(failure)
    return counts, failures, by_players


def main(argv: list[str] | None = None, default_title: str = "1822") -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default=default_title)
    ap.add_argument("--glob", default=None, help="default human_games/<title>/*.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--ruby-results", default=None,
                    help="default tests/g<title>_ruby_head_results.json when it exists")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (pin with taskset)")
    ap.add_argument("--skip-optional-rules", action="store_true",
                    help="skip games with optional rules (1867's grid_market is unimplemented)")
    args = ap.parse_args(argv)
    glob = args.glob or f"human_games/{args.title}/*.json"
    ruby_results = args.ruby_results
    if ruby_results is None:
        ruby_results = str(Path(__file__).resolve().parent / f"g{args.title}_ruby_head_results.json")
    ruby_head: dict = {}
    if ruby_results and Path(ruby_results).exists():
        ruby_head = json.load(open(ruby_results)).get("games", {})

    paths = sorted(globmod.glob(glob))[: args.limit]
    try:
        counts, failures, by_players = audit(
            args.title, paths, ruby_head, args.skip_optional_rules, jobs=args.jobs
        )
    except TitleUnsupported as exc:
        print(f"UNSUPPORTED: {exc}")
        return 2

    print(f"scanned {len(paths)} files: {dict(counts)}")
    if by_players:
        print("by player count:", dict(sorted(by_players.items())))
    if args.json_out:
        json.dump(
            {"counts": dict(counts), "by_players": dict(by_players), "failures": failures},
            open(args.json_out, "w"),
            indent=1,
        )
        print(f"wrote {args.json_out}")
    bad = counts["rejected"] + counts["result_mismatch"] + counts["unparseable"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
