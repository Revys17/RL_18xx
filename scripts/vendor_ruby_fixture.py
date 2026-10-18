"""Vendor tobymao/18xx spec fixtures into tests/fixtures/<title>/, anonymized
the way tests/fixtures/1822 is: player ids renumbered 1..N in seat order
(names "Player N"), `user` / `created_at` dropped, message and `log` texts
blanked (a consent log can name a site account even when the fixture's player
names are already anonymized; the action stays, an undo can target it), so are
program_disable `reason` texts (they quote account names), only the keys the
replay tools read kept.

    python scripts/vendor_ruby_fixture.py logs/ruby/18xx/public/fixtures/1822PNW tests/fixtures/1822PNW
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEEP = ("id", "title", "status", "settings", "players", "result", "actions", "game_end_reason", "turn", "round")


def vendor(src: Path, dst: Path) -> None:
    g = json.load(open(src))
    seat = {p["id"]: i + 1 for i, p in enumerate(g["players"])}

    def fix(a: dict) -> dict:
        a = {k: v for k, v in a.items() if k not in ("user", "created_at")}
        if a.get("entity_type") == "player" and a.get("entity") in seat:
            a["entity"] = seat[a["entity"]]
        if a.get("type") in ("message", "log"):
            a["message"] = ""
        if "reason" in a:
            # program_disable reasons quote site accounts ("<name> bought on
            # corporation X and is unsecure")
            a["reason"] = ""
        if "auto_actions" in a:
            a["auto_actions"] = [fix(x) for x in a["auto_actions"]]
        return a

    out = {k: g[k] for k in KEEP if k in g}
    out["players"] = [{"id": seat[p["id"]], "name": f"Player {seat[p['id']]}"} for p in g["players"]]
    if isinstance(g.get("result"), dict):
        out["result"] = {str(seat.get(int(k) if str(k).isdigit() else k, k)): v for k, v in g["result"].items()}
    out["actions"] = [fix(a) for a in g["actions"]]
    dst.write_text(json.dumps(out, separators=(",", ":")))


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.json")):
        vendor(p, dst / p.name)
        print(dst / p.name)


if __name__ == "__main__":
    main()
