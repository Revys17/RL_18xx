"""Caleb's openers: the bots against the user's auction openings, played to the end of the auction.

M1's "contest" positions (``regression_set.py``) ask the bots for one move right after Caleb's opening bids. This
plays the rest of the auction from his openings and checks whether any bot ever bids on the privates he opened on.
Weakness 1 is inconsistency (docs/bot_weaknesses_2026_11.md "1."): bots that pay far over face for some privates
should not let Caleb take others at face (docs/robustness_measurement.md "Caleb's openers").

* **Openings**: the 11 four-player games in which Caleb (site user ``a``) opened with bids on CA / MH
  (``data/caleb_games``; game 8 opened differently and game 12 had 6 players). Each game gives Caleb's seat and his
  opening bids in order (``OPENERS``), e.g. game 21: seat 1, MH $115 then CA $165. Caleb's seat places these bids on
  his first turns. A scripted bid that is no longer legal (a bot bid on that private first) is dropped and that
  turn is played by the fallback. After the script, ``--caleb`` (default: the generic SL policy at T 1.0, a human
  imitation) plays his seat to the end of the auction, sampled.
* **Bots**: every other seat plays ``--bots``. Searched by default (``--search``: the live bots' settings, the score
  head, 400 readouts, priors at T 1.0, with a fresh tree for every decision as in M1 and ``auction_check.py``; the
  bots keep their trees between moves). With ``--policy-only`` the bots play the policy alone, sampled at
  ``--policy-temperature``. Real auction rules, the rule against train shuffling on.
* **Repeats**: auction ``g`` uses opening ``g mod 11`` and seed ``--seed + g``. A searched bot is deterministic, so
  its repeats differ only through the sampled moves of Caleb's seat.
* **Measures**: for every private Caleb opened on, whether any bot bid on it before it sold (contested), who bought
  it and at what price; for every private, its price, bidders and buyer; and each seat's cash plus the face value of
  its privates when the auction ends (what each side kept).

    uv run python -P scripts/robustness/caleb_openers.py run --name gen6_search \\
        --bots "$GEN6+$GEN6_CRITIC#score" --repeats 10
    uv run python -P scripts/robustness/caleb_openers.py run --name u900_policy --policy-only \\
        --bots generic:<run>/learner/900.pth --repeats 40
    uv run python -P scripts/robustness/caleb_openers.py report
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MAIN = Path(os.environ.get("RL18XX_MAIN", "/home/revys/workspace/RL_18xx"))  # checkpoints, logs
LR1 = Path("/home/revys/longrun/work/lr1")
GEN6 = LR1 / "model_checkpoints_pg/gen6_20261110/learner/2400.pth"
GEN6_CRITIC = LR1 / "model_checkpoints_pg/gen6_20261110/critic/2400.pth"
SL = MAIN / "model_checkpoints_generic/bridge_1830_check_20261113/final.pth"
LIVE_SEARCH = "@400/0.05:mean,dedupe,par=8,ptemp=1.0"  # the live bots since 2026-11-19 (score head, priors at T 1)
OUT = MAIN / "logs/robustness/caleb_openers"
NUM_PLAYERS = 4
FACE = {"SV": 20, "CS": 40, "DH": 70, "MH": 110, "CA": 160, "BO": 220}
COMPANIES = ("SV", "CS", "DH", "MH", "CA", "BO")

# Site game -> (Caleb's player id, his opening bids in order): his bids on his first turns, up to his first turn that
# was not a bid on a private still waiting for the cheaper ones (a pass, or buying the cheapest private at face).
OPENERS = {
    5: (1, [("CA", 165), ("MH", 115)]),
    7: (3, [("CA", 165)]),
    10: (1, [("CA", 165), ("MH", 115)]),
    11: (1, [("CA", 165), ("MH", 115)]),
    13: (1, [("DH", 75), ("CA", 165), ("MH", 115)]),
    14: (1, [("CA", 165)]),
    15: (1, [("CA", 165), ("MH", 115)]),
    18: (2, [("CA", 165)]),
    19: (2, [("MH", 115), ("CA", 165)]),
    20: (3, [("MH", 115)]),
    21: (1, [("MH", 115), ("CA", 165)]),
}
GAMES = sorted(OPENERS)


# ------------------------------------------------------------------------------------------------- players


class PolicyMover:
    """A policy alone, sampled at ``temperature`` (prices from its price head): ``generic:<ckpt>`` or a legacy
    checkpoint."""

    def __init__(self, spec: str, temperature: float):
        from rl18xx.agent.alphazero.generic_policy import GENERIC_PREFIX

        self.spec = spec
        self.temperature = float(temperature)
        if spec.startswith(GENERIC_PREFIX):
            from rl18xx.agent.alphazero.generic_policy import GenericPolicy

            self.generic = GenericPolicy(spec[len(GENERIC_PREFIX):], "cpu")
            self.legacy = None
        else:
            from rl18xx.agent.alphazero.policy_gradient import _load

            self.generic = None
            self.legacy = _load(Path(spec)).eval()

    def move(self, game, legal: list, rng) -> tuple:
        import torch

        if self.generic is not None:
            out = self.generic.evaluate([game], [legal])[0]
            return self.generic.choose(game, out, self.temperature, rng)
        from rl18xx.agent.alphazero.mcts import _rust_encode
        from rl18xx.agent.alphazero.policy_selfplay import _sample_price, tempered

        with torch.no_grad():
            probs, _, _ = self.legacy.run_many_encoded([_rust_encode(game)], legal_indices=[legal])
        price = getattr(self.legacy, "last_price_components", None)
        rows = price["price_logits"].float().cpu().numpy()[0] if price is not None else None
        p = tempered(probs[0].float().cpu().numpy()[legal], self.temperature)
        choice = int(legal[int(rng.choice(len(legal), p=p))])
        return choice, _sample_price(game, choice, rows, rng, 0.0)


_W: dict = {}  # the worker's players


def _init_worker(args: dict) -> None:
    import torch

    torch.set_num_threads(int(args["threads"]))
    from regression_set import load_system

    _W["args"] = args
    _W["caleb"] = PolicyMover(args["caleb"], args["caleb_temperature"])
    if args["policy_only"]:
        _W["bots"] = PolicyMover(args["bots"], args["policy_temperature"])
    else:
        _W["bots"] = load_system(args["bots"], args["search"])


# ------------------------------------------------------------------------------------------------- one auction


def _forced_price(game, index: int) -> Optional[int]:
    pr = game._game.price_range_for_index(int(index))
    return None if pr is None else int(pr[0])


def _in_auction(game) -> bool:
    return (not game.finished) and game._game.round.round_type == "Auction"


def _scripted(game, pid: int, company: str, price: int) -> Optional[int]:
    """The legal index that bids ``price`` on ``company`` for player ``pid``, or None when that bid is not legal."""
    from regression_set import match_indices

    action = {"type": "bid", "entity": pid, "entity_type": "player", "company": company, "price": int(price)}
    found = match_indices(game, action)
    return int(found[0]) if found else None


def play(g: int) -> dict:
    from regression_set import acting_player

    from rl18xx.agent.alphazero.policy_selfplay import has_price_choice
    from rl18xx.agent.alphazero.start_positions import new_game
    from rl18xx.rust_adapter import RustGameAdapter

    a = _W["args"]
    gid = GAMES[g % len(GAMES)]
    caleb, script = OPENERS[gid]
    script = list(script)
    rng = np.random.default_rng(a["seed"] + g)
    game = new_game(NUM_PLAYERS, auction_unlock=False, no_buyback_rule=True)
    record = {"g": g, "opening": gid, "caleb": caleb, "script": [list(s) for s in script], "opened": [],
              "dropped": []}
    decisions = defaultdict(int)
    total = 0
    t0 = time.time()
    try:
        while _in_auction(game):
            rs = game._game
            legal = [int(i) for i in rs.factored_legal_indices()]
            if not legal:
                raise RuntimeError("no legal move in the auction")
            if len(legal) == 1 and not has_price_choice(rs, legal[0]):
                rs.apply_action_index(legal[0], _forced_price(game, legal[0]))
                continue
            if total >= a["max_decisions"]:
                record["termination"] = "auction_stall"
                break
            pid = acting_player(game)
            total += 1
            decisions[pid] += 1
            if pid == caleb:
                index = None
                while script and index is None:
                    company, price = script.pop(0)
                    index = _scripted(game, pid, company, price)
                    (record["opened"] if index is not None else record["dropped"]).append([company, price])
                if index is not None:
                    rs.apply_action_index(index, int(record["opened"][-1][1]))
                else:
                    index, price = _W["caleb"].move(game, legal, rng)
                    rs.apply_action_index(int(index), price)
            elif a["policy_only"]:
                index, price = _W["bots"].move(game, legal, rng)
                rs.apply_action_index(int(index), price)
            else:
                player = _W["bots"].grown_player(game)
                tree = player._rust_player
                player.play_move(int(tree.pick_best_action()))  # at the root's most visited price
                game = RustGameAdapter(tree.root_game_object()).pickle_clone()
        else:
            record["termination"] = "auction_end"
    except Exception as e:  # an engine error must not end the run
        record.update(termination="error", error=f"{type(e).__name__}: {e}")
        return record
    record["decisions"] = {str(k): v for k, v in decisions.items()}
    record["bids"] = [[int(x["entity"]), x["company"], int(x.get("price", 0))]
                      for x in game.raw_actions if x.get("type") == "bid"]
    record["passes"] = sum(1 for x in game.raw_actions if x.get("type") == "pass")
    if record["termination"] == "auction_end":
        players = sorted(game.players, key=lambda p: int(p.id))
        record["cash"] = [int(p.cash) for p in players]
        pids = {int(p.id) for p in players}
        record["owned"] = {}
        for c in game.companies:  # privates held by players (the B&O's private stays its buyer's)
            try:
                owner = int(c.owner.id)
            except (AttributeError, TypeError, ValueError):
                continue
            if owner in pids:
                record["owned"][c.sym] = owner
    record["seconds"] = round(time.time() - t0, 1)
    return record


# ------------------------------------------------------------------------------------------------- analysis


def sales(record: dict) -> dict:
    """``{private: {"buyer", "price", "bidders", "bids"}}``: the highest bid wins (the waterfall's rule); a private
    nobody bid on (the SV handed over at $0) is absent unless ``owned`` names its owner (price 0)."""
    out = {}
    for pid, company, price in record.get("bids", []):
        s = out.setdefault(company, {"buyer": pid, "price": price, "bidders": [], "bids": 0})
        s["bids"] += 1
        if pid not in s["bidders"]:
            s["bidders"].append(pid)
        if price > s["price"]:
            s["buyer"], s["price"] = pid, price
    for company, owner in (record.get("owned") or {}).items():
        if company not in out:
            out[company] = {"buyer": owner, "price": 0, "bidders": [], "bids": 0}
    return out


def opener_rows(record: dict) -> list:
    """Per private Caleb opened on: contested by a bot (any bot bid on it after his opening bid), buyer and price."""
    rows = []
    s = sales(record)
    caleb = record["caleb"]
    for company, price in record["opened"]:
        bids = [(pid, c, p) for pid, c, p in record["bids"] if c == company]
        after = bids[[i for i, (pid, c, p) in enumerate(bids) if pid == caleb and p == price][0] + 1:]
        info = s.get(company, {})
        rows.append({
            "company": company, "opener": price, "contested": any(pid != caleb for pid, _, _ in after),
            "caleb_bought": info.get("buyer") == caleb, "price": info.get("price"), "game": record["opening"],
        })
    return rows


def _mean_se(xs) -> tuple:
    x = np.asarray([float(v) for v in xs], dtype=np.float64)
    if len(x) == 0:
        return math.nan, math.nan
    return float(x.mean()), (float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan)


def _fmt(m, se, pct=False) -> str:
    if math.isnan(m):
        return "-"
    if pct:
        return f"{m:.0%}" + ("" if math.isnan(se) else f" +- {se:.0%}")
    return f"{m:.0f}" + ("" if math.isnan(se) else f" +- {se:.0f}")


def summarize(records: list) -> dict:
    ended = [r for r in records if r.get("termination") == "auction_end"]
    out = {"auctions": len(records), "ended": len(ended),
           "stalls": sum(r.get("termination") == "auction_stall" for r in records),
           "errors": sum(r.get("termination") == "error" for r in records)}
    openers = [row for r in ended for row in opener_rows(r)]
    by_company = defaultdict(list)
    for row in openers:
        by_company[row["company"]].append(row)
    out["openers"] = {}
    for company, rows in sorted(by_company.items()):
        unc = [x["price"] for x in rows if not x["contested"]]
        con = [x["price"] for x in rows if x["contested"]]
        out["openers"][company] = {
            "n": len(rows), "contested": _mean_se(x["contested"] for x in rows),
            "caleb_bought": _mean_se(x["caleb_bought"] for x in rows),
            "price": _mean_se(x["price"] for x in rows), "price_uncontested": _mean_se(unc),
            "price_contested": _mean_se(con),
            "by_game": {str(gid): float(np.mean([x["contested"] for x in rows if x["game"] == gid]))
                        for gid in sorted({x["game"] for x in rows})},
        }
    out["all_openers_contested"] = _mean_se(x["contested"] for x in openers)
    # scripted bids a bot made illegal by bidding on that private first (no opening by Caleb to contest)
    dropped = defaultdict(int)
    scripted = defaultdict(int)
    for r in ended:
        for company, _ in r["script"]:
            scripted[company] += 1
        for company, _ in r["dropped"]:
            dropped[company] += 1
    out["preempted"] = {c: [dropped[c], scripted[c]] for c in sorted(scripted)}
    # every private: the bots' purchases (price over face, 2+ bidders) and Caleb's
    priv = {}
    for company in COMPANIES:
        bot, cal = [], []
        for r in ended:
            s = sales(r).get(company)
            if s is None:
                continue
            row = {"over": s["price"] - FACE[company], "contested": len(s["bidders"]) >= 2}
            (cal if s["buyer"] == r["caleb"] else bot).append(row)
        priv[company] = {
            "caleb_share": len(cal) / max(1, len(cal) + len(bot)),
            "bot_over": _mean_se(x["over"] for x in bot), "bot_contested": _mean_se(x["contested"] for x in bot),
            "caleb_over": _mean_se(x["over"] for x in cal), "caleb_contested": _mean_se(x["contested"] for x in cal),
        }
    out["privates"] = priv
    # the inconsistency: Caleb keeps an opened private at his price while the bots pay $30+ over face for another
    bargains = []
    for r in ended:
        kept = any(not x["contested"] and x["caleb_bought"] for x in opener_rows(r))
        s = sales(r)
        overpaid = any(v["buyer"] != r["caleb"] and v["price"] - FACE[c] >= 30 for c, v in s.items())
        bargains.append(kept and overpaid)
    out["bargain_and_overpay"] = _mean_se(bargains)
    # what each side kept: cash + the face value of its privates when the auction ended
    kept_c, kept_b = [], []
    for r in ended:
        worth = {pid: r["cash"][pid - 1] for pid in range(1, NUM_PLAYERS + 1)}
        for company, owner in (r.get("owned") or {}).items():
            worth[owner] += FACE[company]
        kept_c.append(worth[r["caleb"]])
        kept_b.extend(v for pid, v in worth.items() if pid != r["caleb"])
    out["kept_caleb"] = _mean_se(kept_c)
    out["kept_bot"] = _mean_se(kept_b)
    secs = [r["seconds"] for r in records if "seconds" in r]
    out["seconds"] = float(np.mean(secs)) if secs else math.nan
    return out


def format_summary(name: str, s: dict, settings: dict) -> str:
    bots = settings.get("bots", "?")
    mode = f"policy-only T {settings.get('policy_temperature')}" if settings.get("policy_only") else \
        f"searched {settings.get('search')}"
    lines = [f"== {name}: {s['ended']} auctions (stalls {s['stalls']}, errors {s['errors']}); bots {bots} ({mode})"]
    m, se = s["all_openers_contested"]
    lines.append(f"  Caleb's openers contested by a bot: {_fmt(m, se, True)}")
    lines.append("  bots bid first (Caleb's scripted bid no longer legal): "
                 + ", ".join(f"{c} {d}/{n}" for c, (d, n) in s["preempted"].items()))
    lines.append("  opener   n   contested     Caleb buys    price            uncontested   contested")
    for company, o in s["openers"].items():
        contested, bought = _fmt(*o["contested"], pct=True), _fmt(*o["caleb_bought"], pct=True)
        lines.append(
            f"  {company:<6} {o['n']:>4}   {contested:<12}  {bought:<12}"
            f"  {_fmt(*o['price']):<15}  {_fmt(*o['price_uncontested']):<12}  {_fmt(*o['price_contested'])}"
        )
    for company, o in s["openers"].items():
        lines.append(f"    {company} contested by opening game: "
                     + ", ".join(f"{g} {v:.0%}" for g, v in o["by_game"].items()))
    lines.append("  private  Caleb's share   bots: over face, 2+ bidders     Caleb: over face, 2+ bidders")
    for company, p in s["privates"].items():
        bot_c, caleb_c = _fmt(*p["bot_contested"], pct=True), _fmt(*p["caleb_contested"], pct=True)
        lines.append(
            f"  {company:<7}  {p['caleb_share']:>12.0%}   ${_fmt(*p['bot_over']):<10} {bot_c:<14}"
            f"   ${_fmt(*p['caleb_over']):<10} {caleb_c}"
        )
    m, se = s["bargain_and_overpay"]
    lines.append(
        f"  auctions where Caleb keeps an opener at his price while a bot pays $30+ over face: {_fmt(m, se, True)}"
    )
    lines.append(f"  cash + private face at the end of the auction: Caleb {_fmt(*s['kept_caleb'])}, "
                 f"a bot {_fmt(*s['kept_bot'])}")
    lines.append(f"  {s['seconds']:.0f} s per auction (worker time)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- run / report


def cmd_run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["g"] for line in path.read_text().splitlines() if line.strip()}
    n = args.repeats * len(GAMES)
    todo = [g for g in range(n) if g not in done]
    settings = {
        "bots": args.bots, "policy_only": args.policy_only, "policy_temperature": args.policy_temperature,
        "search": args.search, "caleb": args.caleb, "caleb_temperature": args.caleb_temperature, "seed": args.seed,
        "max_decisions": args.max_decisions, "threads": args.threads, "repeats": args.repeats,
    }
    (out / f"{args.name}.settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{args.name}: {len(todo)} auctions to play ({len(done)} done), {args.workers} workers", flush=True)
    if not todo:
        return
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(settings,)) as pool, path.open("a") as f:
        for k, record in enumerate(pool.imap_unordered(play, todo), 1):
            f.write(json.dumps(record) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"{args.name}: {k}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)


def report(out: Path, names: Optional[list] = None) -> str:
    paths = sorted(out.glob("*.jsonl"))
    if names:
        paths = [p for p in paths if p.stem in names]
    blocks = []
    for p in paths:
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        sp = p.with_suffix(".settings.json")
        settings = json.loads(sp.read_text()) if sp.exists() else {}
        s = summarize(records)
        (out / f"{p.stem}.summary.json").write_text(json.dumps(s, indent=2) + "\n")
        blocks.append(format_summary(p.stem, s, settings))
    return "\n".join(blocks) if blocks else f"no runs in {out}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--bots", default=f"{GEN6}+{GEN6_CRITIC}#score",
                   help="regression_set.py --model syntax; with --policy-only a policy checkpoint (generic:<ckpt>)")
    r.add_argument("--search", default=LIVE_SEARCH)
    r.add_argument("--policy-only", action="store_true")
    r.add_argument("--policy-temperature", type=float, default=0.5)
    r.add_argument("--caleb", default=f"generic:{SL}", help="Caleb's seat after his opening bids (a policy)")
    r.add_argument("--caleb-temperature", type=float, default=1.0)
    r.add_argument("--repeats", type=int, default=10, help="Auctions per opening (11 openings)")
    r.add_argument("--seed", type=int, default=1830)
    r.add_argument("--max-decisions", type=int, default=150, help="Auction decisions before a stall")
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--out", default=str(OUT))
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("names", nargs="*")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.cmd == "run":
        cmd_run(args)
    else:
        print(report(Path(args.out), args.names))


if __name__ == "__main__":
    main()
