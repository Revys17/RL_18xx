"""Compare engine-rs's shadow keys with Ruby's (docs/SITE_ENGINE_RS.md §8.2) position by position.

    python3 scripts/site_engine/compare_keys.py <rust dir: <id>.keys.json> <ruby dir: <id>.ruby.json> [--examples N]

Positions are matched by actionId; every differing field path is counted (corps.PRR.cash -> corps.*.cash).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}" if path else k, out)
    elif a != b:
        out.append((path, a, b))


def generic(path):
    p = re.sub(r"^(corps|companies|hexes)\.[^.]+", r"\1.*", path)
    return re.sub(r"^players\.\d+", "players.*", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rust")
    ap.add_argument("ruby")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    counts = Counter()
    fields = Counter()
    examples = {}
    for rf in sorted(Path(args.rust).glob("*.keys.json")):
        gid = rf.name.split(".")[0]
        yf = Path(args.ruby) / f"{gid}.ruby.json"
        if not yf.exists():
            counts["no ruby file"] += 1
            continue
        rust = json.loads(rf.read_text())
        ruby = json.loads(yf.read_text())
        if "error" in ruby:
            counts["ruby error"] += 1
            examples.setdefault("ruby error", []).append(f"{gid}: {ruby['error'][:200]}")
            continue
        counts["games"] += 1
        if bool(rust.get("stopped")) != bool(ruby.get("stopped")):
            counts["stop differs"] += 1
            examples.setdefault("stop", []).append(f"{gid}: rust {rust.get('stopped')} ruby {ruby.get('stopped')}")
        rp = {p["actionId"]: p["state"] for p in rust["positions"]}
        yp = {p["actionId"]: p["state"] for p in ruby["positions"]}
        if list(rp) != list(yp):
            counts["position lists differ"] += 1
            examples.setdefault("positions", []).append(f"{gid}: rust {len(rp)} ruby {len(yp)}")
        game_bad = False
        for aid, rs in rp.items():
            ys = yp.get(aid)
            if ys is None:
                continue
            counts["positions"] += 1
            # players are a list in seat order: compare by position in Ruby's order
            out = []
            diff({**rs, "players": {str(i): p for i, p in enumerate(rs["players"])}},
                 {**ys, "players": {str(i): p for i, p in enumerate(ys["players"])}}, "", out)
            if out:
                counts["positions differ"] += 1
                game_bad = True
                for path, a, b in out:
                    g = generic(path)
                    fields[g] += 1
                    ex = examples.setdefault(g, [])
                    if len(ex) < args.examples:
                        ex.append(f"{gid}@{aid}: {path}: rust {json.dumps(a)[:150]} ruby {json.dumps(b)[:150]}")
            else:
                counts["positions equal"] += 1
        counts["games differ" if game_bad else "games equal"] += 1
    for k, v in sorted(counts.items()):
        print(f"{k:24s} {v}")
    print("fields:")
    for k, v in fields.most_common():
        print(f"  {v:7d}  {k}")
    for k, ex in examples.items():
        for e in ex:
            print(f"  e.g. {k}: {e}")
    return 0 if not fields and not counts.get("stop differs") else 1


if __name__ == "__main__":
    sys.exit(main())
