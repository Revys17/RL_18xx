#!/usr/bin/env python3
"""View a single PlayoutTrace JSONL file (Phase 1 of MCTS improvements plan).

Reads ``traces/{iteration}/{game_id}.jsonl`` produced by
``MCTSPlayer.dump_traces`` and prints:

1. The header (config snapshot + players).
2. Per-traced-move sections showing each playout's descent path with
   PW grandchild markers and forced-chain lengths.
3. Aggregate statistics: avg leaf depth, PW grandchild rate, distribution
   of leaf-Q values, terminal-leaf rate, expansion rate.

Usage:
    uv run python scripts/view_trace.py traces/0/<game_id>.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> tuple[dict, list[dict]]:
    header: dict = {}
    traces: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and obj.get("kind") == "header":
                header = obj
                continue
            traces.append(obj)
    return header, traces


def _format_step(action: int, was_pw: bool, forced_len: int) -> str:
    marker = "*" if was_pw else " "
    chain = f" (+{forced_len} forced)" if forced_len > 0 else ""
    return f"{marker}{action}{chain}"


def _render_header(header: dict) -> str:
    lines = ["=== Trace header ==="]
    lines.append(f"iteration            : {header.get('iteration')}")
    lines.append(f"game_idx_in_iteration: {header.get('game_idx_in_iteration')}")
    lines.append(f"game_id              : {header.get('game_id')}")
    lines.append(f"num_traces           : {header.get('num_traces')}")
    players = header.get("players") or []
    if players:
        lines.append("players              : " + ", ".join(f"{p['id']}={p.get('name', p['id'])}" for p in players))
    cfg = header.get("trace_config") or {}
    if cfg:
        lines.append(
            "trace_config         : "
            f"rate={cfg.get('trace_game_rate')}, "
            f"every_n={cfg.get('trace_every_n_moves')}, "
            f"per_move={cfg.get('traces_per_move')}"
        )
    return "\n".join(lines)


def _render_moves(traces: list[dict]) -> str:
    by_move: dict[int, list[dict]] = defaultdict(list)
    for t in traces:
        by_move[int(t["move_idx"])].append(t)

    out: list[str] = ["", "=== Per-move trace summary ==="]
    for move_idx in sorted(by_move):
        bucket = by_move[move_idx]
        out.append(f"\n-- move {move_idx} ({len(bucket)} traced playouts) --")
        for tr in bucket:
            ap = tr.get("action_path") or []
            pw = tr.get("pw_grandchild_path") or []
            fc = tr.get("forced_chain_lengths") or []
            steps = [
                _format_step(int(a), bool(pw[i]) if i < len(pw) else False, int(fc[i]) if i < len(fc) else 0)
                for i, a in enumerate(ap)
            ]
            term = " TERM" if tr.get("leaf_terminal") else ""
            exp = " EXP" if tr.get("expansion_occurred") else ""
            q = tr.get("leaf_q_perspective", 0.0)
            depth = tr.get("leaf_depth", len(ap))
            ent = tr.get("leaf_prior_entropy", 0.0)
            out.append(
                f"  depth={depth:3d} Q_persp={q:+.3f} entropy={ent:.3f}{term}{exp}"
                f"   path: {' -> '.join(steps) if steps else '(root expansion)'}"
            )
    return "\n".join(out)


def _render_aggregate(traces: list[dict]) -> str:
    if not traces:
        return "\n=== Aggregate ===\n(no traces)"
    depths = [int(t.get("leaf_depth", 0)) for t in traces]
    qs = [float(t.get("leaf_q_perspective", 0.0)) for t in traces]
    entropies = [float(t.get("leaf_prior_entropy", 0.0)) for t in traces]
    forced_totals = [sum(int(x) for x in (t.get("forced_chain_lengths") or [])) for t in traces]
    pw_steps = sum(sum(1 for x in (t.get("pw_grandchild_path") or []) if x) for t in traces)
    total_steps = sum(len(t.get("action_path") or []) for t in traces) or 1
    terminal_rate = sum(1 for t in traces if t.get("leaf_terminal")) / len(traces)
    expansion_rate = sum(1 for t in traces if t.get("expansion_occurred")) / len(traces)

    def _q(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return f"{values[0]:.3f}"
        return (
            f"min={min(values):.3f} "
            f"p25={statistics.quantiles(values, n=4)[0]:.3f} "
            f"med={statistics.median(values):.3f} "
            f"p75={statistics.quantiles(values, n=4)[2]:.3f} "
            f"max={max(values):.3f}"
        )

    lines = ["", "=== Aggregate ==="]
    lines.append(f"traces                 : {len(traces)}")
    lines.append(f"avg leaf depth         : {sum(depths)/len(depths):.2f}  (max {max(depths)})")
    lines.append(f"leaf Q (perspective)   : {_q(qs)}")
    lines.append(f"leaf prior entropy     : {_q(entropies)}")
    lines.append(f"PW grandchild rate     : {pw_steps/total_steps:.3f}  ({pw_steps}/{total_steps} steps)")
    lines.append(f"avg forced chain total : {sum(forced_totals)/len(forced_totals):.2f}")
    lines.append(f"terminal-leaf rate     : {terminal_rate:.3f}")
    lines.append(f"expansion rate         : {expansion_rate:.3f}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="JSONL trace file produced by MCTSPlayer.dump_traces")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Trace file not found: {path}", file=sys.stderr)
        return 2

    header, traces = _load(path)
    print(_render_header(header))
    print(_render_moves(traces))
    print(_render_aggregate(traces))
    return 0


if __name__ == "__main__":
    sys.exit(main())
