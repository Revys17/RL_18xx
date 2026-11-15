"""One robustness-arm eval round in one line, and the arm's rounds side by side (docs/robustness_measurement.md
"In the arms' evals").

``scripts/launch/2026-11/bridge/robust_arm.sh <arm> evals`` writes, per round ``u<U>`` under the arm's eval directory:

- ``u<U>_<players>p_<a>v<b>_<opponent>/summary.json``: pg_generic_1830.sh's policy-only matches;
- ``u<U>_m1_policy.jsonl`` / ``u<U>_m1_search_score.jsonl``: M1 (``regression_set.py score``) of the newest learner
  alone (policy argmax) and searched with the run's critic on its score head (the bots' settings);
- ``u<U>_m3/<member>/summary.json``: M3, the learner alone against three copies of each panel member
  (``exploiter_panel.sh``; the arm's loop exploiters as extra members);
- ``u<U>_m2_eval/summary.json`` / ``u<U>_m2_pg/summary.json``: M2 (``behaviour_stats.py run``) on the round's saved
  4-player eval games (the matches' and M3's; side A = the arm) and on the run's saved 4-player policy-gradient games
  since the previous round (A = the learner).

This script reads whatever of that exists for a round (a missing piece is a dash), writes ``round_u<U>.json``,
rewrites ``rounds.txt`` (one line per round, like gen6's ``outfamily.txt``) and ``robust_status.txt`` (one column per
round, with the references: M1 of gen6 / SL ckpt 10 from ``data/m1_baselines``, M2 of the human corpus and gen6 from
``data/behaviour_baselines.json``).

    uv run python -P scripts/robustness/round_summary.py <eval dir> --update 100 [--label robust_A0_20261115]
    uv run python -P scripts/robustness/round_summary.py <eval dir> --status-only
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

M1_RUNS = (("policy", "policy alone (argmax)"), ("search_score", "searched, critic #score"))
# M2 rows: (metric, group) -- A = the arm's seats; "all" = per game. The weakness numbers are
# docs/bot_weaknesses_2026_11.md's.
M2_ROWS = [
    ("looted_broke_dump", "A", "2 looted-broke dumps onto A (per A seat-game)"),
    ("looted_broke_dump", "B", "2 looted-broke dumps onto B (A's tactic when A dumps)"),
    ("broke_dump_victim_bankrupt", "A", "2 A bankrupt after a broke dump onto it"),
    ("handover_cash300", "A", "3 A hands over a >= $300 treasury"),
    ("buy_into_trainless_broke", "A", "4 A buys into a trainless broke company"),
    ("took_trainless_broke", "A", "4 A takes over a trainless broke company"),
    ("trainless_turn", "A", "5 A's companies end a turn trainless"),
    ("suitcase2", "A", "5 A suitcases (2+ ORs)"),
    ("bankrupt", "A", "A goes bankrupt"),
]
M2_PRIVATES = ("CA", "MH", "DH", "CS")
M2_SHORT = {"looted_broke_dump": "lbd", "broke_dump_victim_bankrupt": "bdvb", "handover_cash300": "h300",
            "buy_into_trainless_broke": "bitb", "took_trainless_broke": "ttb", "trainless_turn": "tt",
            "suitcase2": "sc2", "bankrupt": "bkr"}
M3_SHORT = {"sl10_t1": "sl10@1", "sl10_t05": "sl10@.5", "gx2_bankrupt": "gx2b", "gx2_rust": "gx2r", "gen6": "gen6"}


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list:
    if not path.exists():
        return []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def _match(summary) -> Optional[dict]:
    s = summary[0] if isinstance(summary, list) and summary else summary
    if not isinstance(s, dict) or "A_score" not in s:
        return None
    return {"score": s["A_score"], "se": s.get("A_score_se"), "fair": s.get("A_score_equal"), "games": s.get("games")}


def m1_summary(results: list) -> Optional[dict]:
    if not results:
        return None
    from regression_set import summarize

    s = summarize(results)
    out = {"n": s["all"]["n"], "pass": s["all"]["pass"], "p_bad": s["all"]["p_bad"],
           "by_weakness": {w: r["pass"] for w, r in s.items() if w != "all"}}
    if "search_chose_bad" in s["all"]:
        out["search_chose_bad"] = s["all"]["search_chose_bad"]
        out["search_share_bad"] = s["all"]["search_share_bad"]
    return out


def m2_summary(summary: Optional[dict]) -> Optional[dict]:
    if not summary:
        return None
    rates = summary.get("rates") or {}
    out = {"games": summary.get("games", 0),
           "rates": {f"{m}|{g}": (rates.get(m) or {}).get(g, 0.0) for m, g, _ in M2_ROWS},
           "auction": {}}
    for co in M2_PRIVATES:
        a = (summary.get("auction") or {}).get(co)
        if a:
            row = {"n": a["n"], "mean": a["mean"], "contested": a["contested"]}
            ga = (a.get("by_group") or {}).get("A")
            if ga:
                row["A"] = {"n": ga["n"], "mean": ga["mean"], "contested": ga["contested"]}
            out["auction"][co] = row
    return out


def collect(eval_out: Path, update: int) -> dict:
    """Everything of round ``u<update>`` under ``eval_out``."""
    prefix = f"u{update}_"
    rnd = {"update": update, "evals": {}, "m1": {}, "m2": {}, "m3": {}}
    for d in sorted(eval_out.glob(f"{prefix}*")):
        name = d.name[len(prefix):]
        if not d.is_dir() or name.startswith(("m1", "m2", "m3")):
            continue
        match = _match(_json(d / "summary.json"))
        if match:
            rnd["evals"][name] = match
    for key, _ in M1_RUNS:
        s = m1_summary(_jsonl(eval_out / f"{prefix}m1_{key}.jsonl"))
        if s:
            rnd["m1"][key] = s
    for key in ("eval", "pg"):
        s = m2_summary(_json(eval_out / f"{prefix}m2_{key}" / "summary.json"))
        if s:
            rnd["m2"][key] = s
    m3 = eval_out / f"{prefix}m3"
    if m3.is_dir():
        for d in sorted(p for p in m3.iterdir() if p.is_dir() and not p.name.endswith(".m2")):
            match = _match(_json(d / "summary.json"))
            if match:
                rnd["m3"][d.name] = match
    return rnd


def _f(x, digits=2, pct=False) -> str:
    if x is None:
        return "-"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:.{digits}f}"


def round_line(rnd: dict, label: str = "") -> str:
    parts = [f"u{rnd['update']}" + (f" {label}" if label else "")]
    if rnd["evals"]:
        parts.append("evals " + " ".join(f"{k} {_f(v['score'], 3)}" for k, v in rnd["evals"].items()))
    m1 = []
    for key, _ in M1_RUNS:
        s = rnd["m1"].get(key)
        if s:
            cell = f"{key} pass {_f(s['pass'])} p(bad) {_f(s['p_bad'])}"
            if "search_chose_bad" in s:
                cell += f" chose bad {_f(s['search_chose_bad'])}"
            cell += " [" + " ".join(f"{w[:4]} {_f(v)}" for w, v in s["by_weakness"].items()) + "]"
            m1.append(cell)
    if m1:
        parts.append("M1 " + "; ".join(m1))
    for key in ("eval", "pg"):
        s = rnd["m2"].get(key)
        if s:
            cells = [f"{M2_SHORT[k.split('|')[0]]}|{k.split('|')[1]} {_f(v)}" for k, v in s["rates"].items()]
            cells += [f"{co} ${a['mean']:.0f} c{_f(a['contested'], pct=True)}" for co, a in s["auction"].items()
                      if co in ("CA", "MH")]
            parts.append(f"M2 {key} ({s['games']} games) " + " ".join(cells))
    if rnd["m3"]:
        parts.append("M3 " + " ".join(f"{M3_SHORT.get(k, k)} {_f(v['score'], 3)}" for k, v in rnd["m3"].items()))
    return " | ".join(parts)


# ------------------------------------------------------------------------------------------- references
def m1_references() -> dict:
    base = HERE / "data" / "m1_baselines"
    out = {}
    for name, file in (("gen6 policy", "gen6_u2400_policy_only"), ("SL10 policy", "sl10_policy_only"),
                       ("gen6+critic #score", "gen6_u2400_critic2400_score_r400")):
        s = m1_summary(_jsonl(base / f"{file}.jsonl.gz"))
        if s:
            out[name] = s
    return out


def m2_references() -> dict:
    try:
        data = json.loads((HERE / "data" / "behaviour_baselines.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for name, group in (("human", "human"), ("gen6", "A")):
        b = data.get(name)
        if not b:
            continue
        rates = b.get("rates") or {}
        # the human corpus has one group (every seat "human"): its per seat-game rate stands for both sides
        out[name] = {f"{m}|{g}": (rates.get(m) or {}).get(group if g == "A" or name == "human" else "B")
                     for m, g, _ in M2_ROWS}
        out[name]["auction"] = {co: (b.get("auction") or {}).get(co) for co in M2_PRIVATES}
    return out


def status_table(rounds: list, label: str = "") -> str:
    """Rows: metrics; columns: the references, then the rounds."""
    cols = [f"u{r['update']}" for r in rounds]
    rows = []  # (section, metric, reference, [cells])

    def add(section, metric, ref, cells):
        rows.append((section, metric, ref, cells))

    names = []
    for r in rounds:
        for k in r["evals"]:
            if k not in names:
                names.append(k)
    for k in names:
        fair = next((r["evals"][k]["fair"] for r in rounds if k in r["evals"]), None)
        add("evals", k, f"fair {_f(fair)}", [_f(r["evals"].get(k, {}).get("score"), 3) for r in rounds])
    refs1 = m1_references()
    for key, desc in M1_RUNS:
        ref_name = "gen6+critic #score" if key == "search_score" else "gen6 policy"
        ref = refs1.get(ref_name)
        sl = refs1.get("SL10 policy") if key == "policy" else None
        ref_txt = (f"gen6 {_f(ref['pass'])}" if ref else "") + (f" / SL10 {_f(sl['pass'])}" if sl else "")
        add("M1", f"{desc}: pass", ref_txt, [_f((r["m1"].get(key) or {}).get("pass")) for r in rounds])
        ref_txt = (f"gen6 {_f(ref['p_bad'])}" if ref else "") + (f" / SL10 {_f(sl['p_bad'])}" if sl else "")
        add("M1", f"{desc}: p(bad)", ref_txt, [_f((r["m1"].get(key) or {}).get("p_bad")) for r in rounds])
        if key == "search_score":
            add("M1", f"{desc}: chose bad", f"gen6 {_f(ref['search_chose_bad'])}" if ref else "",
                [_f((r["m1"].get(key) or {}).get("search_chose_bad")) for r in rounds])
        weak = list((ref or {}).get("by_weakness") or {})
        for r in rounds:
            for w in (r["m1"].get(key) or {}).get("by_weakness", {}):
                if w not in weak:
                    weak.append(w)
        for w in weak:
            add("M1", f"{desc}: pass {w}", f"gen6 {_f((ref or {}).get('by_weakness', {}).get(w))}",
                [_f((r["m1"].get(key) or {}).get("by_weakness", {}).get(w)) for r in rounds])
    refs2 = m2_references()
    for key, title in (("eval", "M2 eval games (4p)"), ("pg", "M2 PG games (4p)")):
        add(title, "games", "", [str((r["m2"].get(key) or {}).get("games", "-")) for r in rounds])
        for m, g, desc in M2_ROWS:
            k = f"{m}|{g}"
            ref = " / ".join(f"{n} {_f(v.get(k))}" for n, v in refs2.items())
            add(title, desc, ref, [_f((r["m2"].get(key) or {}).get("rates", {}).get(k)) for r in rounds])
        for co in M2_PRIVATES:
            ref = " / ".join(
                f"{n} ${v['auction'][co]['mean']:.0f} c{_f(v['auction'][co]['contested'], pct=True)}"
                for n, v in refs2.items() if v.get("auction", {}).get(co))
            cells = []
            for r in rounds:
                a = (r["m2"].get(key) or {}).get("auction", {}).get(co)
                if not a:
                    cells.append("-")
                    continue
                cell = f"${a['mean']:.0f} c{_f(a['contested'], pct=True)}"
                if a.get("A"):
                    cell += f" (A ${a['A']['mean']:.0f})"
                cells.append(cell)
            add(title, f"1 {co}: price, contested", ref, cells)
    members = []
    for r in rounds:
        for k in r["m3"]:
            if k not in members:
                members.append(k)
    for k in members:
        add("M3 (1v3, fair 0.25)", f"vs {k}", "", [_f(r["m3"].get(k, {}).get("score"), 3) for r in rounds])

    widths = [max(len(x) for x in [c] + [row[3][i] for row in rows]) for i, c in enumerate(cols)]
    w_metric = max([len(m) for _, m, _, _ in rows] + [6])
    w_ref = max([len(ref) for _, _, ref, _ in rows] + [9])
    lines = [f"robustness status{(' ' + label) if label else ''}: {len(rounds)} round(s)",
             f"{'metric':{w_metric}} | {'reference':{w_ref}} | "
             + " | ".join(f"{c:>{w}}" for c, w in zip(cols, widths))]
    section = None
    for s, m, ref, cells in rows:
        if s != section:
            lines.append(f"-- {s}")
            section = s
        lines.append(f"{m:{w_metric}} | {ref:{w_ref}} | " + " | ".join(f"{c:>{w}}" for c, w in zip(cells, widths)))
    return "\n".join(lines)


def write_status(eval_out: Path, label: str = "") -> list:
    rounds = []
    for p in sorted(eval_out.glob("round_u*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        r = _json(p)
        if r:
            rounds.append(r)
    label = label or next((r["label"] for r in reversed(rounds) if r.get("label")), "")
    (eval_out / "rounds.txt").write_text("".join(round_line(r, r.get("label") or label) + "\n" for r in rounds))
    (eval_out / "robust_status.txt").write_text(status_table(rounds, label) + "\n")
    return rounds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_out", help="the arm's eval directory (robust_arm.sh's EVAL_OUT)")
    parser.add_argument("--update", type=int, default=None, help="the round to collect (u<U>)")
    parser.add_argument("--label", default="", help="e.g. the run's name")
    parser.add_argument("--status-only", action="store_true", help="rewrite rounds.txt / robust_status.txt only")
    args = parser.parse_args(argv)
    eval_out = Path(args.eval_out)
    if not args.status_only:
        if args.update is None:
            raise SystemExit("--update (or --status-only)")
        rnd = collect(eval_out, args.update)
        rnd["label"] = args.label
        (eval_out / f"round_u{args.update}.json").write_text(json.dumps(rnd, indent=1) + "\n")
        print(round_line(rnd, args.label))
    write_status(eval_out, args.label)


if __name__ == "__main__":
    main()
