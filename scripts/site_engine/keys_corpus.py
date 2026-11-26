"""Run `site_engine keys` over a corpus (docs/SITE_ENGINE_RS.md §8.2) and check what Rust alone can: every game
replays, every stored run's revenues equal Rust's own pricing (the key's last_action), the final key agrees with the
site's stored result / acting, and the output is deterministic (two runs give the same digest). Writes one
`<id>.keys.json` per game when --out is given (for B's ruby_keys.rb comparison).

    python3 scripts/site_engine/keys_corpus.py <site_engine> <corpus dir>... [--out DIR] [--twice]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--twice", action="store_true")
    args = ap.parse_args()
    files = sorted((p for d in args.dirs for p in Path(d).glob("*.json")), key=lambda p: (len(p.stem), p.stem))
    counts = Counter()
    examples = {}
    digest = hashlib.sha256()
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for f in files:
        game = json.loads(f.read_text())
        inp = json.dumps({"game": game}).encode()
        r = subprocess.run([args.bin, "keys"], input=inp, capture_output=True)
        if r.returncode != 0:
            counts[f"exit {r.returncode}"] += 1
            examples.setdefault(f"exit {r.returncode}", f"{f.stem}: {r.stdout[:200]!r} {r.stderr[-300:]!r}")
            continue
        if args.twice:
            r2 = subprocess.run([args.bin, "keys"], input=inp, capture_output=True)
            a = json.loads(r.stdout)
            b = json.loads(r2.stdout)
            a.pop("ms", None)
            b.pop("ms", None)
            if a != b:
                counts["nondeterministic"] += 1
        d = json.loads(r.stdout)
        d.pop("ms", None)
        digest.update(json.dumps(d, sort_keys=True).encode())
        if out:
            (out / f"{f.stem}.keys.json").write_text(json.dumps(d))
        counts["games"] += 1
        if d.get("stopped"):
            counts["stopped"] += 1
            examples.setdefault("stopped", f"{f.stem}: {d['stopped']}")
        counts["positions"] += len(d["positions"])
        by_id = {a.get("id"): a for a in game.get("actions", [])}
        for pos in d["positions"]:
            la = pos["state"]["last_action"]
            if la and "revenues" in la:
                a = by_id.get(pos["actionId"], {})
                stored = [r.get("revenue") for r in a.get("routes", [])]
                counts["runs"] += 1
                if stored == la["revenues"]:
                    counts["runs equal"] += 1
                else:
                    counts["runs differ"] += 1
                    examples.setdefault(f"run differs {len(examples)}", f"{f.stem}@{pos['actionId']}: stored {stored} rust {la['revenues']} nodes {[r.get('nodes') for r in a.get('routes', [])]}")
        last = d["positions"][-1]["state"]
        if game.get("status") == "finished" and not d.get("stopped"):
            counts["finished"] += 1
            site = {str(k): v for k, v in (game.get("result") or {}).items()}
            if last["finished"] and site == last["result"]:
                counts["finished: result equal"] += 1
            else:
                counts["finished: result differs"] += 1
                examples.setdefault("result", f"{f.stem}: site {site} rust {last['result']} finished={last['finished']}")
    for k, v in sorted(counts.items()):
        print(f"{k:28s} {v}")
    print("digest", digest.hexdigest()[:16])
    for k, v in examples.items():
        print(f"  e.g. {k}: {v}")
    return 0 if not counts.get("stopped") and not counts.get("runs differ") else 1


if __name__ == "__main__":
    sys.exit(main())
