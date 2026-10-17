"""Vendor a recorded 18xx.games game as an anonymized test fixture.

Players become "Player N" with ids 1..N (seat order) — every player entity,
`user` field, result key and auction/ability reference is remapped — chat
(`message` actions) and timestamps are dropped, and only the fields the
replay harness and Ruby need are kept. Undo/redo structure is preserved
(action ids are untouched), so Ruby's filtered_actions and ours see the same
stream.

    uv run python scripts/vendor_fixture.py IN.json tests/fixtures/1822/OUT.json
"""

from __future__ import annotations

import json
import sys

KEEP_TOP = ("id", "title", "status", "settings", "players", "result", "actions", "game_end_reason", "turn", "round")


def anonymize(game: dict) -> dict:
    seat = {p["id"]: i + 1 for i, p in enumerate(game["players"])}

    def fix_action(a: dict) -> dict:
        a = {k: v for k, v in a.items() if k not in ("created_at", "user")}
        if a.get("entity_type") == "player" and a.get("entity") in seat:
            a["entity"] = seat[a["entity"]]
        if a.get("auto_actions"):
            a["auto_actions"] = [fix_action(x) for x in a["auto_actions"] if x.get("type") != "message"]
        return a

    out = {k: game[k] for k in KEEP_TOP if k in game}
    out["id"] = str(game.get("id"))
    out["players"] = [{"id": seat[p["id"]], "name": f"Player {seat[p['id']]}"} for p in game["players"]]
    by_name = {p["name"]: seat[p["id"]] for p in game["players"]}
    result = {}
    for k, v in (game.get("result") or {}).items():
        if str(k).isdigit() and int(k) in seat:
            result[str(seat[int(k)])] = v
        elif k in by_name:
            result[str(by_name[k])] = v
        else:
            result[k] = v
    out["result"] = result
    settings = dict(game.get("settings") or {})
    settings.pop("unlisted", None)
    out["settings"] = settings
    out["actions"] = [fix_action(a) for a in game["actions"] if a.get("type") != "message"]
    return out


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    game = json.load(open(src))
    out = anonymize(game)
    with open(dst, "w") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {dst}: {len(out['actions'])} actions, {len(out['players'])} players")
    return 0


if __name__ == "__main__":
    sys.exit(main())
