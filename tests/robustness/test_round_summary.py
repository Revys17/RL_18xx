"""The arms' per-round summary (scripts/robustness/round_summary.py): one line per round and the rounds side by side,
from whatever of a round's eval / M1 / M2 / M3 outputs exist."""

import gzip
import json
import shutil
import sys
from pathlib import Path

ROBUST = Path(__file__).resolve().parents[2] / "scripts" / "robustness"
sys.path.insert(0, str(ROBUST))

import regression_set as rs  # noqa: E402
import round_summary as summary  # noqa: E402

BASE = ROBUST / "data" / "m1_baselines"


def _match(path: Path, score: float, fair: float = 0.25):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps([{"A_score": score, "A_score_se": 0.02, "A_score_equal": fair,
                                                    "games": 400, "A_worth_share": 0.3, "mean_decisions": 600}]))


def _m1(path: Path, baseline: str, limit: int):
    with gzip.open(BASE / f"{baseline}.jsonl.gz", "rt") as f:
        lines = f.readlines()[:limit]
    path.write_text("".join(lines))
    return [json.loads(line) for line in lines]


def _m2(path: Path, lbd_a: float):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({
        "label": "x", "games": 50, "rates": {"looted_broke_dump": {"all": 0.2, "A": lbd_a, "B": 0.1},
                                             "handover_cash300": {"all": 1.0, "A": 0.3}},
        "auction": {"CA": {"n": 10, "mean": 250.0, "contested": 0.9, "by_group": {"A": {"n": 4, "mean": 240.0,
                                                                                        "contested": 1.0}}}},
    }))


def test_rounds_line_and_status_table(tmp_path):
    out = tmp_path / "eval"
    for u, score in ((100, 0.31), (200, 0.35)):
        _match(out / f"u{u}_4p_1v3_sl", score)
        _match(out / f"u{u}_2p_1v1_gen6", score + 0.2, fair=0.5)
        _m2(out / f"u{u}_m2_eval", 0.05 if u == 100 else 0.02)
        _match(out / f"u{u}_m3" / "sl10_t1", score - 0.05)
        (out / f"u{u}_m3" / "sl10_t1.m2").mkdir()  # M3's own M2 output: not a member
    policy = _m1(out / "u100_m1_policy.jsonl", "gen6_u2400_policy_only", 59)
    searched = _m1(out / "u200_m1_search_score.jsonl", "gen6_u2400_critic2400_score_r400", 20)
    _m1(out / "u200_m1_policy.jsonl", "sl10_policy_only", 59)
    summary.main([str(out), "--update", "100", "--label", "robust_A0"])
    summary.main([str(out), "--update", "200", "--label", "robust_A0"])
    rounds = (out / "rounds.txt").read_text().splitlines()
    assert len(rounds) == 2 and rounds[0].startswith("u100 robust_A0") and rounds[1].startswith("u200")
    assert "4p_1v3_sl 0.310" in rounds[0] and "M3 sl10@1 0.260" in rounds[0] and "M2 eval (50 games)" in rounds[0]
    r100 = json.loads((out / "round_u100.json").read_text())
    assert set(r100["evals"]) == {"4p_1v3_sl", "2p_1v1_gen6"} and set(r100["m3"]) == {"sl10_t1"}
    assert r100["m1"]["policy"]["pass"] == rs.summarize(policy)["all"]["pass"] and "search_score" not in r100["m1"]
    r200 = json.loads((out / "round_u200.json").read_text())
    assert r200["m1"]["search_score"]["search_chose_bad"] == rs.summarize(searched)["all"]["search_chose_bad"]
    assert r200["m2"]["eval"]["rates"]["looted_broke_dump|A"] == 0.02
    status = (out / "robust_status.txt").read_text()
    head = status.splitlines()[1]
    assert head.split("|")[-2].strip() == "u100" and head.split("|")[-1].strip() == "u200"
    row = next(line for line in status.splitlines() if line.startswith("policy alone (argmax): pass "))
    assert row.split("|")[-2].strip() == f"{rs.summarize(policy)['all']['pass']:.2f}"
    assert "gen6 0.27 / SL10 0.71" in row  # the M1 references (data/m1_baselines)
    lbd = next(line for line in status.splitlines() if line.startswith("2 looted-broke dumps onto A"))
    assert lbd.split("|")[-2].strip() == "0.05" and "human" in lbd  # the M2 reference (data/behaviour_baselines)
    # --status-only rewrites the files from the round jsons; a round without anything still gets a column
    shutil.rmtree(out / "u200_m3")
    summary.main([str(out), "--update", "300"])
    assert len((out / "rounds.txt").read_text().splitlines()) == 3
    summary.main([str(out), "--status-only"])
    assert "u300" in (out / "robust_status.txt").read_text().splitlines()[1]
