"""Fetch the 1830 corpus of docs/SITE_ENGINE_RS.md §8.1 into $RUST1/corpus (read-only, one request at a time).

    python3 scripts/site_engine/fetch_corpus.py dev  [--from 1] [--to N]   # :9301 GET /api/game/<id>  -> corpus/g9301/
    python3 scripts/site_engine/fetch_corpus.py prod [--from 1] [--to N]   # :9292 bot API, read-only  -> corpus/prod/

prod uses the bot API's read-only GET /api/bot/game/<id> (its `game` key) with the token from
~/.config/18xx/local_site_bot_token (sent as a header, never printed); its files hold real users' names: mode 600,
never printed or committed, deleted when the checks are done. Keeps 1830 games only; stops after --misses
consecutive missing ids. Prints counts only.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RUST1 = Path(os.environ.get("RUST1", "/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/rust1"))


def get(url, headers):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (404, 400, 403):
            return None
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", choices=["dev", "prod"])
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", type=int, default=100000)
    ap.add_argument("--misses", type=int, default=150)
    ap.add_argument("--pause", type=float, default=0.05)
    args = ap.parse_args()
    if args.site == "dev":
        base, out, headers = "http://127.0.0.1:9301/api/game/", RUST1 / "corpus" / "g9301", {}
    else:
        token = Path.home().joinpath(".config/18xx/local_site_bot_token").read_text().strip()
        base, out = "http://127.0.0.1:9292/api/bot/game/", RUST1 / "corpus" / "prod"
        headers = {"Authorization": f"Bearer {token}"}
    out.mkdir(parents=True, exist_ok=True)
    if args.site == "prod":
        os.chmod(out, 0o700)
    kept = other = missing = 0
    misses = 0
    gid = args.start
    while gid <= args.to and misses < args.misses:
        data = get(f"{base}{gid}", headers)
        time.sleep(args.pause if args.site == "dev" else max(args.pause, 0.2))
        if data is None:
            missing += 1
            misses += 1
            gid += 1
            continue
        misses = 0
        game = data.get("game", data) if args.site == "prod" else data
        if game.get("title") != "1830" or game.get("status") == "new":
            other += 1
            gid += 1
            continue
        path = out / f"{gid}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if args.site == "prod" else 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(game, f)
        kept += 1
        gid += 1
    print(f"{args.site}: kept {kept} 1830 games, {other} other, {missing} missing ids, last id tried {gid - 1}", file=sys.stderr)


if __name__ == "__main__":
    main()
