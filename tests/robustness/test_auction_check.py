"""scripts/robustness/auction_check.py: paired openings and the report (no networks)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "robustness"))
sys.path.insert(0, str(ROOT / "scripts"))

import auction_check as ac  # noqa: E402

STARTS = [{"id": str(i), "actions": [{"type": "pass", "entity": 1}] * (3 + i)} for i in range(5)]


def test_openings_are_paired_and_stay_in_the_auction():
    for g in range(20):
        a = ac.opening(g, STARTS, 6, 1830)
        b = ac.opening(g, STARTS, 6, 1830)
        assert a[:2] == b[:2]  # every run plays game g from the same opening
        line = next(s for s in STARTS if s["id"] == a[0])
        assert len(a[1]) <= len(line["actions"]) - 1  # never the auction's last action


def _record(g, cand, points, cfwin, sales):
    return {
        "g": g,
        "candidate": cand,
        "termination": "auction_end",
        "cash": [100, 200, 300, 400],
        "auction": sales,
        "values": {"critic1ep": {"win": cfwin, "score": cfwin}},
        "playouts": {"points": points, "win": points},
        "seconds": 1.0,
    }


def test_report_pairs_runs_by_game(tmp_path):
    ctrl = [_record(g, [g % 4 + 1], [0.25] * 4, [0.25] * 4, {"DH": [1, 120]}) for g in range(4)]
    cand = []
    for g in range(4):
        pts = [0.25] * 4
        pts[g % 4] = 0.35  # the candidate seat does 0.1 better than the same seat in the control
        cand.append(_record(g, [g % 4 + 1], pts, [0.25] * 4, {"DH": [g % 4 + 1, 80]}))
    for name, recs in (("control", ctrl), ("cand", cand)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (tmp_path / f"{name}.settings.json").write_text(json.dumps({"candidate": name}))
    text = ac.report(tmp_path, "control")
    line = next(x for x in text.splitlines() if "playout points" in x and "|" in x and "cand" not in x)
    assert "+0.1000" in line and "(4)" in line
    dh = [x for x in text.splitlines() if x.strip().startswith("DH")]
    assert any("80.0 (4)" in x for x in dh)  # the candidate bought DH at 80 in every game
    assert ac._judges({"termination": "auction_stall"}) == {}
