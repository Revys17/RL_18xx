"""Auction check: an auction system against the live bots' auction system in real 4-player waterfall auctions.

The bots search every auction decision with their play-auction networks (gen6 u2400 + its critic, the score head,
400 readouts, the advisor's PUCT, priors at T 0.5). This script plays auctions in which a candidate system holds
``--candidate-seats`` seats and the bots' system the others, and judges each finished auction at the first Stock Round
(docs/robustness_measurement.md "The auction check"):

* **Openings**: a random human 4-player auction (``human_games/start_positions_1830_4p_v2.jsonl``, 1,945 games) cut
  after its first 0..``--prefix`` actions, so games differ and contested privates come up as they do between humans.
  Game ``g`` of every run uses the same opening and the same candidate seats (seed ``--seed`` + g): runs pair up.
* **Play**: every decision searched with a fresh tree (as M1; the bots keep theirs between moves, at the same total
  budget), the most visited move at its most visited price; a forced move is played. Real auction rules, the rule
  against train shuffling on (``autoplay.bot_position``). An auction still running after ``--max-decisions``
  searched decisions is a stall.
* **Judges** at the first Stock Round, per seat: ``--playouts`` games of the rest played by ``--playout-policy``
  alone (every seat, softmax(log p / ``--playout-temperature``), prices from its price head) -> mean placement points
  (4p 1 / .5 / .2 / 0) and win share; and value networks' win / score estimates: gen6's critic (the bots' value),
  the human-game critic (``critic1ep``), lr1's v4 and value_v2 (ckpt 26, the advisor's win estimates). A side's own
  value favours it: the play-outs and the third-party values are the neutral judges.
* **Prices**: what each side paid per private, against the human medians.

Runs: ``run --name <run> --candidate <spec> [--candidate-search <search>]``, one JSONL record per auction in
``<out>/<run>.jsonl`` (resumable). A run with ``--candidate`` = the baseline is the control for the paired
differences. ``report --out <dir> [--control <run>]`` prints the comparison. Specs as ``regression_set.py --model``
(legacy ``<policy>+<value>[#score]``, generic ``generic:<policy>[+generic:<value>][#score]``, mixed
``<legacy policy>+generic:<value>[#score]``); searches in eval_head_to_head's ``@<n>/<c>:options`` syntax.

    uv run python -P scripts/robustness/auction_check.py run --name control --games 400
    uv run python -P scripts/robustness/auction_check.py run --name gen6_critic1ep --games 400 \\
        --candidate "$GEN6+generic:$CRITIC1EP#score"
    uv run python -P scripts/robustness/auction_check.py report
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

MAIN = Path(os.environ.get("RL18XX_MAIN", "/home/revys/workspace/RL_18xx"))  # checkpoints, data, logs
LR1 = Path("/home/revys/longrun/work/lr1")
GEN6 = LR1 / "model_checkpoints_pg/gen6_20261110/learner/2400.pth"
GEN6_CRITIC = LR1 / "model_checkpoints_pg/gen6_20261110/critic/2400.pth"
V4 = LR1 / "model_checkpoints_value/lr1/promoted/4.pth"
VALUE_V2 = Path(
    "/home/revys/workspace/RL_18xx/.claude/worktrees/gpu-experiments-training-93d287/model_checkpoints_value/"
    "AlphaZeroTransformer/20261004_134558_804974562/26.pth"
)
CRITIC1EP = MAIN / "model_checkpoints_generic/bridge_1830_critic1ep_20261118/final.pth"
STARTS = MAIN / "human_games/start_positions_1830_4p_v2.jsonl"
BASELINE = f"{GEN6}+{GEN6_CRITIC}#score"
BOT_SEARCH = "@400/0.05:mean,dedupe,par=8,ptemp=0.5"  # the live bots (advisor SEARCH_OVERRIDES, gen6's T 0.5)
OUT = MAIN / "logs/robustness/auction_check"
HUMAN_MEDIAN = {"CS": 45, "DH": 80, "MH": 135, "CA": 210, "BO": 220}  # 1,945 human 4p auctions (eval_auction.py)
COMPANIES = ("SV", "CS", "DH", "MH", "CA", "BO")
NUM_PLAYERS = 4

# ------------------------------------------------------------------------------------------------- one auction

_W: dict = {}  # the worker's systems and judges


def _init_worker(args: dict) -> None:
    import torch

    torch.set_num_threads(int(args["threads"]))
    from regression_set import load_system

    from rl18xx.agent.alphazero.game_network import GenericNetwork
    from rl18xx.agent.alphazero.policy_gradient import _load

    baseline = load_system(args["baseline"], args["baseline_search"])
    same = (args["candidate"], args["candidate_search"]) == (args["baseline"], args["baseline_search"])
    _W.update(
        args=args,
        baseline=baseline,
        candidate=baseline if same else load_system(args["candidate"], args["candidate_search"]),
        playout=_load(Path(args["playout_policy"])).eval(),
        legacy_judges={name: _load(Path(p)).eval() for name, p in (("gen6_critic", GEN6_CRITIC), ("v4", V4),
                                                                  ("value_v2", VALUE_V2))},
        generic_judges={"critic1ep": GenericNetwork(str(CRITIC1EP), "cpu")},
        starts=[json.loads(line) for line in STARTS.read_text().splitlines() if line.strip()],
    )


def opening(g: int, starts: list, prefix: int, seed: int) -> tuple:
    """``(human game id, its first k actions, the candidate's player ids)`` of game ``g``."""
    rng = np.random.default_rng(seed + g)
    line = starts[int(rng.integers(len(starts)))]
    k = int(rng.integers(0, prefix + 1))
    k = min(k, max(0, len(line["actions"]) - 1))
    return line["id"], line["actions"][:k], rng


def new_auction(actions: list):
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game

    game = new_game(NUM_PLAYERS, auction_unlock=False, no_buyback_rule=True)
    return apply_actions(game, actions)


def _in_auction(game) -> bool:
    return (not game.finished) and game._game.round.round_type == "Auction"


def _forced_price(game, index: int) -> Optional[int]:
    pr = game._game.price_range_for_index(int(index))
    return None if pr is None else int(pr[0])


def play_auction(game, systems: dict, max_decisions: int) -> tuple:
    """Play ``game``'s auction to its end: ``(game, termination, searched decisions per player id)``."""
    from regression_set import acting_player

    from rl18xx.agent.alphazero.policy_selfplay import has_price_choice
    from rl18xx.rust_adapter import RustGameAdapter

    searched = defaultdict(int)
    total = 0
    while _in_auction(game):
        rs = game._game
        legal = [int(i) for i in rs.factored_legal_indices()]
        if not legal:
            raise RuntimeError("no legal move in the auction")
        if len(legal) == 1 and not has_price_choice(rs, legal[0]):
            rs.apply_action_index(legal[0], _forced_price(game, legal[0]))
            continue
        if total >= max_decisions:
            return game, "auction_stall", dict(searched)
        pid = acting_player(game)
        player = systems[pid].grown_player(game)
        tree = player._rust_player
        player.play_move(int(tree.pick_best_action()))  # at the root's most visited price
        game = RustGameAdapter(tree.root_game_object()).pickle_clone()
        searched[pid] += 1
        total += 1
    return game, "auction_end", dict(searched)


def _legacy_values(model, game) -> tuple:
    """``(win shares, score)`` per player id order of a legacy value network."""
    import torch

    from rl18xx.agent.alphazero.mcts import _rust_encode, unrotate_value

    e = _rust_encode(game)
    n = int(e[7])
    with torch.no_grad():
        win, score = model.run_values_and_scores_encoded([e])
    win = np.clip(unrotate_value(win[0].float().cpu().numpy(), e[6], n)[:n], 0.0, None)
    score = unrotate_value(score[0].float().cpu().numpy(), e[6], n)[:n]
    win = win / win.sum() if win.sum() > 0 else np.full(n, 1.0 / n)
    return [float(x) for x in win], [float(x) for x in score]


def judge_values(game) -> dict:
    out = {}
    for name, model in _W["legacy_judges"].items():
        win, score = _legacy_values(model, game)
        out[name] = {"win": win, "score": score}
    n = len(game.players)
    for name, net in _W["generic_judges"].items():
        o = net.forward([game])[0]  # absolute seat order, padded to the model's seats
        out[name] = {"win": [float(x) for x in o.win[:n]], "score": [float(x) for x in o.score[:n]]}
    return out


def playouts(game, k: int, temperature: float, max_decisions: int, rng) -> dict:
    """``k`` policy-only games from ``game``, in lockstep on the worker's play-out policy: per player id order the
    mean placement points and win share, and the terminations."""
    import torch

    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.policy_gradient import placement_points
    from rl18xx.agent.alphazero.policy_selfplay import _sample_price, has_price_choice, tempered
    from rl18xx.agent.alphazero.self_play import _compute_net_worth

    model = _W["playout"]
    games = [game.pickle_clone() for _ in range(k)]
    decisions = [0] * k
    terminations = [None] * k
    rows = [None] * k  # each game's last price-head row
    while True:
        batch = []
        for i, g in enumerate(games):
            if terminations[i]:
                continue
            rs = g._game
            while True:
                if g.finished:
                    terminations[i] = "finished"
                    break
                if decisions[i] >= max_decisions:
                    g.end_game()
                    terminations[i] = "max_length"
                    break
                legal = [int(x) for x in rs.factored_legal_indices()]
                if len(legal) > 1 or (len(legal) == 1 and has_price_choice(rs, legal[0])):
                    batch.append((i, legal))
                    break
                rs.apply_action_index(legal[0], _sample_price(g, legal[0], rows[i], rng, 0.0))
        if not batch:
            break
        encoded = [_rust_encode(games[i]) for i, _ in batch]
        with torch.no_grad():
            probs, _, _ = model.run_many_encoded(encoded, legal_indices=[legal for _, legal in batch])
        price = getattr(model, "last_price_components", None)
        price_logits = price["price_logits"].float().cpu().numpy() if price is not None else None
        for b, (i, legal) in enumerate(batch):
            p = tempered(probs[b].float().cpu().numpy()[legal], temperature)
            choice = legal[int(rng.choice(len(legal), p=p))]
            rows[i] = price_logits[b] if price_logits is not None else None
            games[i]._game.apply_action_index(choice, _sample_price(games[i], choice, rows[i], rng, 0.0))
            decisions[i] += 1
    points, wins = [], []
    for g in games:
        worth = _compute_net_worth(g)
        w = np.array([float(worth[pid]) for pid in sorted(worth)])
        points.append(placement_points(w))
        top = w == w.max()
        wins.append(top / top.sum())
    return {
        "points": [float(x) for x in np.mean(points, axis=0)],
        "win": [float(x) for x in np.mean(wins, axis=0)],
        "terminations": {t: terminations.count(t) for t in set(terminations)},
        "decisions": float(np.mean(decisions)),
    }


def run_game(g: int) -> dict:
    from rl18xx.agent.alphazero.search_learning import auction_sales

    a = _W["args"]
    t0 = time.time()
    gid, actions, rng = opening(g, _W["starts"], a["prefix"], a["seed"])
    seats = a["candidate_seats"]
    first = g % NUM_PLAYERS
    candidate = sorted(((first + j) % NUM_PLAYERS) + 1 for j in range(seats))
    systems = {pid: (_W["candidate"] if pid in candidate else _W["baseline"]) for pid in range(1, NUM_PLAYERS + 1)}
    record = {"g": g, "opening": gid, "prefix": len(actions), "candidate": candidate}
    try:
        game, termination, searched = play_auction(new_auction(actions), systems, a["max_decisions"])
    except Exception as e:  # an engine error must not end the run
        return {**record, "termination": "error", "error": f"{type(e).__name__}: {e}"}
    record.update(termination=termination, searched=searched, auction=auction_sales(game.raw_actions))
    if termination == "auction_end":
        record["cash"] = [int(p.cash) for p in sorted(game.players, key=lambda p: int(p.id))]
        record["values"] = judge_values(game)
        if a["playouts"]:
            record["playouts"] = playouts(game, a["playouts"], a["playout_temperature"], a["playout_max_decisions"], rng)
    record["seconds"] = round(time.time() - t0, 1)
    return record


# ------------------------------------------------------------------------------------------------- run / report


def cmd_run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["g"] for line in path.read_text().splitlines() if line.strip()}
    todo = [g for g in range(args.games) if g not in done]
    settings = {
        "candidate": args.candidate, "candidate_search": args.candidate_search, "baseline": args.baseline,
        "baseline_search": args.baseline_search, "candidate_seats": args.candidate_seats, "prefix": args.prefix,
        "seed": args.seed, "max_decisions": args.max_decisions, "playouts": args.playouts,
        "playout_policy": args.playout_policy, "playout_temperature": args.playout_temperature,
        "playout_max_decisions": args.playout_max_decisions, "threads": args.threads,
    }  # fmt: skip
    (out / f"{args.name}.settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{args.name}: {len(todo)} auctions to play ({len(done)} done), {args.workers} workers", flush=True)
    if not todo:
        return
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(settings,)) as pool, path.open("a") as f:
        for n, record in enumerate(pool.imap_unordered(run_game, todo), 1):
            f.write(json.dumps(record) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{args.name}: {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)


def _mean_se(x) -> tuple:
    x = np.asarray(list(x), dtype=np.float64)
    if len(x) == 0:
        return math.nan, math.nan
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan


def _judges(record: dict) -> dict:
    """Per judge the candidate seats' summed estimate (``None`` where the record has no such judge)."""
    if record.get("termination") != "auction_end":
        return {}
    seats = [pid - 1 for pid in record["candidate"]]
    out = {}
    if "playouts" in record:
        out["playout points"] = sum(record["playouts"]["points"][s] for s in seats)
        out["playout win"] = sum(record["playouts"]["win"][s] for s in seats)
    for name, v in record.get("values", {}).items():
        out[f"{name} win"] = sum(v["win"][s] for s in seats)
        out[f"{name} score"] = sum(v["score"][s] for s in seats)
    return out


def load_run(path: Path) -> dict:
    return {r["g"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def report(out: Path, control: str) -> str:
    runs = {p.stem: load_run(p) for p in sorted(out.glob("*.jsonl"))}
    if not runs:
        return f"no runs in {out}"
    lines = []
    ctrl = runs.get(control)
    for name, recs in runs.items():
        ended = [r for r in recs.values() if r.get("termination") == "auction_end"]
        stalls = sum(1 for r in recs.values() if r.get("termination") == "auction_stall")
        errors = sum(1 for r in recs.values() if r.get("termination") == "error")
        settings_path = out / f"{name}.settings.json"
        cand = json.loads(settings_path.read_text())["candidate"] if settings_path.exists() else "?"
        lines.append(f"== {name}: {len(ended)} auctions (stalls {stalls}, errors {errors}); candidate {cand}")
        fair = None
        if ended:
            fair = len(ended[0]["candidate"]) / NUM_PLAYERS
        judged = defaultdict(list)
        paired = defaultdict(list)
        for r in ended:
            mine = _judges(r)
            for k, v in mine.items():
                judged[k].append(v)
            c = ctrl.get(r["g"]) if ctrl is not None and name != control else None
            if c is not None and c.get("termination") == "auction_end" and c["candidate"] == r["candidate"]:
                theirs = _judges(c)
                for k in mine:
                    if k in theirs:
                        paired[k].append(mine[k] - theirs[k])
        lines.append(f"  judge (candidate seats' sum; fair {fair})    mean +- se    | paired vs {control}: diff +- se (n)")
        for k in sorted(judged):
            m, se = _mean_se(judged[k])
            row = f"  {k:<28} {m:7.4f} +- {se:.4f}"
            if paired.get(k):
                dm, dse = _mean_se(paired[k])
                row += f"   | {dm:+.4f} +- {dse:.4f} ({len(paired[k])})"
            lines.append(row)
        # prices: per private, the candidate seats' and the other seats' mean price and the candidate's share
        prices = defaultdict(lambda: {"cand": [], "base": []})
        for r in ended:
            for company, (buyer, price) in (r.get("auction") or {}).items():
                prices[company]["cand" if int(buyer) in r["candidate"] else "base"].append(price)
        lines.append("  private  candidate mean (n)   others mean (n)   human median")
        for company in COMPANIES:
            p = prices.get(company)
            if not p:
                continue
            cm = f"{np.mean(p['cand']):6.1f} ({len(p['cand'])})" if p["cand"] else "     - (0)"
            bm = f"{np.mean(p['base']):6.1f} ({len(p['base'])})" if p["base"] else "     - (0)"
            lines.append(f"  {company:<7}  {cm:<18}   {bm:<16}  {HUMAN_MEDIAN.get(company, '-')}")
        cash = [r["cash"][pid - 1] for r in ended for pid in r["candidate"]]
        if cash:
            lines.append(f"  candidate seats' cash at the first Stock Round: mean {np.mean(cash):.1f}")
        secs = [r["seconds"] for r in recs.values() if "seconds" in r]
        if secs:
            lines.append(f"  {np.mean(secs):.0f} s per auction (worker time)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--candidate", default=BASELINE)
    r.add_argument("--candidate-search", default=BOT_SEARCH)
    r.add_argument("--baseline", default=BASELINE)
    r.add_argument("--baseline-search", default=BOT_SEARCH)
    r.add_argument("--candidate-seats", type=int, default=1, choices=(1, 2, 3))
    r.add_argument("--games", type=int, default=400)
    r.add_argument("--prefix", type=int, default=6, help="Human opening actions: uniform 0..N")
    r.add_argument("--seed", type=int, default=1830)
    r.add_argument("--max-decisions", type=int, default=120, help="Searched auction decisions before a stall")
    r.add_argument("--playouts", type=int, default=8, help="Policy-only games of the rest per auction (0: none)")
    r.add_argument("--playout-policy", default=str(GEN6))
    r.add_argument("--playout-temperature", type=float, default=0.5)
    r.add_argument("--playout-max-decisions", type=int, default=1500)
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--out", default=str(OUT))
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--control", default="control")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.cmd == "run":
        cmd_run(args)
    else:
        print(report(Path(args.out), args.control))


if __name__ == "__main__":
    main()
