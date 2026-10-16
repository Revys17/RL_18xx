"""Rust-vs-Ruby per-action state diff for recorded 1867 games.

The 1867 port has no Python oracle: rules are validated against the REAL
Ruby engine (tobymao/18xx) in the ruby:3.2-slim docker image. This script
replays each game in BOTH engines, snapshots the state after every FILE
action id (corp price/cash/trains, player cash by seat, bank, round type),
and reports per game:

  * the FIRST file id where the two states differ (and the action there),
  * where either engine rejects the record (Ruby HEAD rejects some old
    games on its own — e.g. the pre-tobymao/18xx#9655 merger order),
  * whether the final results agree (Ruby's result vs ours, by seat), and
    whether both flag game end at the same action.

Setup (once; read-only use of the clone, never contact 18xx.games):
    git clone --depth 1 https://github.com/tobymao/18xx.git logs/ruby/18xx
    docker image inspect ruby:3.2-slim   # `docker pull ruby:3.2-slim` if missing

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_ruby_diff.py \\
        --corpus-dir human_games/1867 --audit logs/g1867/audit.json   # its failing games
    ... scripts/g1867_ruby_diff.py --corpus-dir ... 20229 24683          # explicit ids
    ... --results-json logs/g1867/ruby_results.json   # Ruby-HEAD scores for the audit

Mind the id pitfall: Ruby dumps are keyed by FILE action ids; the Rust side
tags every filtered action with its originating file id (auto_actions
inherit the parent's) — never compare by the renumbered filtered index.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def ensure_gems(gems: Path) -> None:
    gems.mkdir(parents=True, exist_ok=True)
    if any(gems.glob("gems/require_all-*")):
        return
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{gems}:/usr/local/bundle", "ruby:3.2-slim",
         "gem", "install", "require_all"],
        check=True, capture_output=True,
    )


def ruby_dump(game_path: Path, out: Path, ruby_repo: Path, gems: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    subprocess.run(
        ["docker", "run", "--rm", "-e", "ENGINE_LOG_LEVEL=3",
         "-v", f"{gems}:/usr/local/bundle",
         "-v", f"{ruby_repo.resolve()}:/18xx:ro",
         "-v", f"{game_path.resolve().parent}:/games:ro",
         "-v", f"{script_dir}:/probe:ro",
         "-v", f"{out.parent.resolve()}:/out",
         "ruby:3.2-slim", "ruby", "/probe/g1867_ruby_dump.rb",
         f"/games/{game_path.name}", f"/out/{out.name}"],
        check=False, capture_output=True,
    )


def rust_dump(game: dict) -> list[dict]:
    import engine_rs

    from rl18xx.agent.alphazero.pretraining import filter_actions

    raw = json.loads(json.dumps(game["actions"]))
    for a in raw:
        a["_fid"] = a["id"]
        for au in a.get("auto_actions") or []:
            au["_fid"] = a["id"]
    acts = filter_actions(raw)
    seat = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    rust = engine_rs.BaseGame.new_titled(game["title"], {s: f"Player {s}" for s in seat.values()})
    rows = []
    for i, action in enumerate(acts):
        fid = action.pop("_fid")
        action = dict(action)
        if action.get("entity_type") == "player":
            action["entity"] = action["user"] = seat[action["entity"]]
        try:
            rust.process_action(action)
        except BaseException as exc:  # noqa: BLE001 — incl. pyo3 panics
            rows.append({"id": fid, "exception": str(exc)[:300]})
            break
        if i + 1 < len(acts) and acts[i + 1].get("_fid") == fid:
            continue
        rt, _, step, _ = rust.debug_state()
        rows.append({
            "id": fid, "round": rt, "step": step, "phase": rust.phase.name,
            "corps": {c.sym: {"p": c.share_price.price if c.share_price else None, "cash": c.cash,
                              "fl": c.floated, "tr": [t.name for t in c.trains]}
                      for c in rust.corporations},
            "pcash": {str(p.id): p.cash for p in rust.players},
            "finished": bool(rust.finished), "bank": rust.bank.cash,
        })
    rows.append({"final": True, "finished": bool(rust.finished),
                 "result": {str(k): v for k, v in rust.result().items()}})
    return rows


def split(rows):
    states, final, exc = {}, None, None
    for d in rows:
        if d.get("final"):
            final = d
        elif "exception" in d:
            exc = d
        else:
            states[d["id"]] = d
    return states, final, exc


def diffs(a: dict, b: dict) -> list[str]:
    out = []
    if a["round"] != b["round"]:
        out.append(f"round rs={a['round']} rb={b['round']}")
    for sym, ca in a["corps"].items():
        cb = b["corps"].get(sym)
        if cb is None:
            continue
        if ca["p"] != cb["p"] and (ca["fl"] or cb["fl"]):
            out.append(f"{sym} price rs={ca['p']} rb={cb['p']}")
        if ca["cash"] != cb["cash"]:
            out.append(f"{sym} cash rs={ca['cash']} rb={cb['cash']}")
    for s, v in a["pcash"].items():
        vb = b["pcash"].get(s)
        if vb is not None and vb != v:
            out.append(f"seat{s} cash rs={v} rb={vb}")
    if a["bank"] != b["bank"]:
        out.append(f"bank rs={a['bank']} rb={b['bank']}")
    return out


def first_finished(states: dict):
    return next((i for i in sorted(states) if states[i].get("finished")), None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="*", help="game ids (looked up in --corpus-dir) or JSON paths")
    ap.add_argument("--corpus-dir", default="human_games/1867")
    ap.add_argument("--audit", help="g1867_corpus_audit --json output: diff its failing games")
    ap.add_argument("--all-replayable", type=int, metavar="N",
                    help="every finished, rule-free game among the first N corpus files")
    ap.add_argument("--ruby-repo", default="logs/ruby/18xx")
    ap.add_argument("--gems", default="logs/ruby/gems", help="persistent gem dir (require_all)")
    ap.add_argument("--work", default="logs/ruby/dumps", help="where Ruby dumps are cached")
    ap.add_argument("--refresh", action="store_true", help="re-run cached Ruby dumps")
    ap.add_argument("--jobs", type=int, default=6, help="parallel Ruby containers")
    ap.add_argument("--results-json", help="write Ruby-HEAD final scores (by seat) for full replays")
    args = ap.parse_args()

    corpus = Path(args.corpus_dir)
    paths: list[Path] = []
    for g in args.games:
        p = Path(g)
        paths.append(p if p.suffix == ".json" else corpus / f"{g}.json")
    if args.audit:
        audit = json.load(open(args.audit))
        paths += [corpus / f"{f['id']}.json" for f in audit["failures"]]
    if args.all_replayable:
        for p in sorted(corpus.glob("*.json"))[: args.all_replayable]:
            d = json.load(open(p))
            if d.get("status") in ("finished", "archived") and not d.get("settings", {}).get("optional_rules"):
                paths.append(p)
    if not paths:
        ap.error("no games")

    ruby_repo, gems, work = Path(args.ruby_repo), Path(args.gems).resolve(), Path(args.work)
    if not (ruby_repo / "lib" / "engine.rb").exists():
        ap.error(f"no tobymao/18xx clone at {ruby_repo} (see the module docstring)")
    ensure_gems(gems)

    def rb(p: Path) -> Path:
        out = work / f"{p.stem}.rb.jsonl"
        if args.refresh or not out.exists() or out.stat().st_size == 0:
            ruby_dump(p, out, ruby_repo, gems)
        return out

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rb_files = list(ex.map(rb, paths))

    results = {}
    summary = {"no_divergence": 0, "diverges": 0, "ruby_rejects": 0, "result_differs": 0}
    for p, rbf in zip(paths, rb_files):
        game = json.load(open(p))
        rs_states, rs_final, rs_exc = split(rust_dump(game))
        rb_rows = [json.loads(line) for line in open(rbf)] if rbf.exists() else []
        rb_states, rb_final, rb_exc = split(rb_rows)
        first = None
        for fid in sorted(set(rs_states) & set(rb_states)):
            d = diffs(rs_states[fid], rb_states[fid])
            if d:
                first = (fid, d)
                break
        tags = []
        if rs_exc:
            tags.append(f"rust rejects @{rs_exc['id']}")
        if rb_exc:
            tags.append(f"ruby rejects @{rb_exc['id']}: {rb_exc['exception'][:70]}")
            summary["ruby_rejects"] += 1
        if rb_final and not rb_exc:
            rb_res = {str(k): v for k, v in rb_final["result"].items()}
            if rs_final and not rs_exc and rb_res != rs_final["result"]:
                tags.append(f"RESULT ours={rs_final['result']} ruby={rb_res}")
                summary["result_differs"] += 1
            if rs_final and not rs_exc:
                a, b = first_finished(rs_states), first_finished(rb_states)
                if a != b:
                    tags.append(f"finish rs@{a} rb@{b} ({rb_final.get('game_end_reason') or '-'})")
                results[p.stem] = {"scores_by_seat": rb_res, "finished": rb_final["finished"],
                                   "end_reason": rb_final.get("game_end_reason") or ""}
        if first:
            summary["diverges"] += 1
            fid, d = first
            act = next((a for a in game["actions"] if a["id"] == fid), {})
            act = {k: v for k, v in act.items() if k not in ("created_at", "user", "entity_type", "auto_actions")}
            print(f"{p.stem} DIVERGES @{fid} {json.dumps(act)[:140]} | {'; '.join(d[:4])} | {' | '.join(tags)}")
        else:
            summary["no_divergence"] += 1
            print(f"{p.stem} no divergence | {' | '.join(tags)}")
    print(summary)
    if args.results_json:
        head = subprocess.run(["git", "-C", str(ruby_repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        json.dump({"ruby_head": head.stdout.strip(), "games": results}, open(args.results_json, "w"),
                  indent=0, sort_keys=True)
        print(f"wrote {args.results_json} ({len(results)} full Ruby replays)")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(main())
