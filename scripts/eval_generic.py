"""Policy-only head-to-head between two title-generic checkpoints, for any title with an RL layout (RL bridge
milestone C2; ``docs/rl_model.md`` "Policy gradient for the generic model").

``scripts/eval_policy_only.py`` plays 1830 only (the legacy encoder routes moves, the legacy start code makes
the starts) but can seat legacy checkpoints; this one plays generic checkpoints only, on any title: starts from
``generic_play.replayed_start`` (seeded by the start index, so every seat arrangement of a start shares it; the
opening with ``--auction-start``), legal moves from ``generic_play.title_play``, moves and prices from each
side's network exactly as the generic policy-gradient workers draw them (temperature in log space, the price
cell from the head over the legal range, a price inside it). Each side runs in the pool workers (CPU) or, with
``--server``, on a generic inference server (the GPU when there is one). Side A holds ``--a-seats`` of
``--num-players`` seats in every arrangement; equal strength scores ``a_seats / players``.

**Searched sides** (``docs/rl_model.md`` "Searching 1867"): a side given in ``eval_head_to_head.py``'s player grammar
-- ``generic:<policy>[+generic:<critic>][#win|#score|#blend=<w>][@<readouts>][/<c_puct_init>][:<option>,...]`` (a
``+``, ``#`` or ``:`` part; ``:sample=<t>`` is the policy alone at temperature t) -- searches every decision with the
Rust tree over the title's generator moves (``generic_search.GenericSearchPlayer``): priors and price cells from the
policy, values from the critic's head (``#score``: placement points, finished games backed up as placement points
when the critic's run trained a placement reward), the most visited move played (``--softpick N``: visit-sampled for
a side's first N decisions). All of a worker's searches run in lockstep (``lockstep.LockstepSearcher``, one request
per round and network: ``--games-per-task`` games' leaves together), on generic inference servers with ``--server``
(one per checkpoint of a searched side: its policy and its critic), else in the workers. A side keeps its tree while
the game's position is its root (its own consecutive decisions). Which side decides a position is the deciding
player's (``engine_rs.mcts_decider_id``: the encoder's mover, except 1867's stock-round home token, which the opening
bidder chooses -- the encoder names nobody there and seat 1 used to choose for every side).

    uv run python scripts/eval_generic.py --title 1867 --num-players 4 --games 480 --server --games-per-task 8 \
        --match "generic:$R/learner/2000.pth+generic:$R/critic/2000.pth#score@400/0.05:mean,dedupe,par=8,ptemp=0.5" \
                "$R/learner/2000.pth@0.5"

Optional rules (1822's 1822+ / tax-haven combinations, ``titles.TitleSpec.rules_mix``): ``--rules <name>`` plays
every game under one combination (``base``, ``plus``, ``plus+nr+ss+th``, ...), ``--rules-mix <name>:<weight>,...``
draws each start's combination (seeded by the start index, like the start: every seat arrangement of a start plays
the same rules); the default is the title's mix (1822: the human games' combinations; other titles: base rules).
Records carry the combination (``rules``); the summary splits decisions and scores by it (``by_rules``).

All-pass openings (the 1822 family): a game whose opening Stock Round ends with nothing bought, sold or bid ends by
Ruby's rule with every seat tied (``generic_play.all_pass_opening``); it is not scored but replayed from a fresh
draw of its start index (``<seed>:<start>:r<k>``; at most ``generic_play.MAX_ALL_PASS_REDRAWS``, then it counts
with termination ``all_pass``), so ties do not dilute the score. The summary counts them (``all_pass_redrawn``);
``--keep-all-pass`` scores them as they are.

    uv run python scripts/eval_generic.py --title 1830 --match <a.pth>@0.5 <b.pth>@0.5 --num-players 4 \\
        --a-seats 1 --games 400 --auction-start --out logs/eval/<name>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_player(spec: str) -> tuple:
    """``(checkpoint path, temperature)`` of ``<checkpoint>[@<temperature>]`` (a ``generic:`` prefix is allowed)."""
    from rl18xx.agent.alphazero.generic_policy import GENERIC_PREFIX

    if spec.startswith(GENERIC_PREFIX):
        spec = spec[len(GENERIC_PREFIX) :]
    path, _, temperature = spec.partition("@")
    if not Path(path).is_file():
        raise SystemExit(f"checkpoint {path!r} not found")
    return str(Path(path)), float(temperature) if temperature else 1.0


def is_search_spec(spec: str) -> bool:
    """Whether a side is given in ``eval_head_to_head.py``'s grammar (a ``+``, ``#`` or ``:`` part)."""
    from rl18xx.agent.alphazero.generic_policy import GENERIC_PREFIX

    body = spec.replace("+" + GENERIC_PREFIX, "+")
    body = body[len(GENERIC_PREFIX) :] if body.startswith(GENERIC_PREFIX) else body
    return any(c in body for c in "+#:")


def parse_side(spec: str, readouts: int) -> dict:
    """A side: ``{"kind": "policy", "path", "temperature"}`` or ``{"kind": "search", "policy", "value",
    "value_head", "overrides"}`` (``generic_search.parse_search_spec``)."""
    if not is_search_spec(spec):
        path, temperature = parse_player(spec)
        return {"kind": "policy", "path": path, "temperature": temperature}
    from rl18xx.agent.alphazero.generic_search import parse_search_spec

    try:
        s = parse_search_spec(spec, default_readouts=readouts)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"{spec!r}: {e}")
    if s.policy_only:
        if s.value:
            raise SystemExit(f"{spec!r}: sample=<t> plays the policy alone (no value part)")
        return {"kind": "policy", "path": s.policy, "temperature": s.overrides["policy_temperature"]}
    return {"kind": "search", "policy": s.policy, "value": s.value, "value_head": s.value_head,
            "overrides": s.overrides}  # fmt: skip


def side_paths(side: dict) -> list:
    if side["kind"] == "policy":
        return [side["path"]]
    return [p for p in (side["policy"], side["value"]) if p]


_LOCAL: dict = {}


def _policy(path: str):
    """A side's in-process server (``LocalGenericClient`` on the checkpoint, CPU), one per process."""
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_inference import LocalGenericClient, LocalGenericServer

    if path not in _LOCAL:
        model, _ = mg.load_checkpoint(path, "cpu")
        _LOCAL[path] = LocalGenericClient(LocalGenericServer(model.eval()))
    return _LOCAL[path]


def _search_setup(settings: dict) -> tuple:
    """``(LockstepSearcher, {side: SearchSpec})`` of the searched sides (None, {} without any): each side's
    evaluator on its generic servers (``--server``) or in-process clients."""
    from rl18xx.agent.alphazero import generic_pg as gpg
    from rl18xx.agent.alphazero.generic_search import SearchSpec
    from rl18xx.agent.alphazero.lockstep import GenericEvaluator, LockstepSearcher

    specs, evaluators = {}, {}
    for side, sp in (settings.get("search") or {}).items():
        specs[side] = SearchSpec(sp["policy"], sp["value"], sp["value_head"], dict(sp["overrides"]))
        if settings["server"]:
            policy, value = gpg._CLIENTS[f"{side}:policy"], gpg._CLIENTS.get(f"{side}:value")
            evaluators[side] = GenericEvaluator(policy, value, sp["value_head"])
        else:
            value = _policy(sp["value"]) if sp["value"] else None
            evaluators[side] = GenericEvaluator(_policy(sp["policy"]), value, sp["value_head"], compress=False)
    return (LockstepSearcher(evaluators) if evaluators else None), specs


def decider_seat(raw, seating) -> int:
    """The seating index of the player who decides ``raw``'s position (``engine_rs.mcts_decider_id``; nobody: the
    first player, as the encoder's seat 0)."""
    import engine_rs

    decider = engine_rs.mcts_decider_id(raw)
    return list(seating).index(int(raw.players[0].id if decider is None else decider))


def play_games(game_indices: list, settings: dict) -> list:
    """Pool task: play the given games concurrently; one record per game."""
    import torch

    from rl18xx.agent.alphazero import generic_pg as gpg
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_inference import slice_positions
    from rl18xx.agent.alphazero.generic_play import ALL_PASS, MAX_ALL_PASS_REDRAWS, all_pass_opening, replayed_start
    from rl18xx.agent.alphazero.generic_play import title_play
    from rl18xx.agent.alphazero.generic_search import GenericSearchPlayer
    from rl18xx.agent.alphazero.generic_tokens import encode_batch
    from rl18xx.agent.alphazero.titles import draw_rules, rules_name

    torch.set_num_threads(1)
    rng = np.random.default_rng()
    searcher, search_specs = _search_setup(settings)
    title = settings["title"]
    play = title_play(title)
    seatings = [tuple(a) for a in settings["arrangements"]]
    n = len(seatings[0])

    def new_game(idx: int, redraws: int = 0):
        # the start index seeds the start (every arrangement of a start shares it); a redrawn all-pass opening
        # takes the k-th redraw of that start index
        key = f"{settings['seed']}:{idx // len(seatings)}" + (f":r{redraws}" if redraws else "")
        start_rng = random.Random(key)
        rules = draw_rules(settings.get("rules_mix") or {"base": 1.0}, start_rng, title)
        start, raw, _ = replayed_start(
            title, n, settings["start_positions"], settings["random_start_fraction"], rng=start_rng,
            auction_fraction=1.0 if settings["auction_start"] else 0.0, auction_unlock=False, optional_rules=rules,
        )  # fmt: skip
        seats = list(seatings[idx % len(seatings)])
        g = gpg._Game(raw=raw, uid=str(idx), opponent="", seat_models=seats, start=start.label)
        g.seed, g.idx = start.seed, idx
        g.rules = rules_name(title, start.optional_rules)
        # the seating order is the new game's (``titles.new_game`` numbers the seats in order), not raw.players after
        # the start's actions: the 1822 family reorders its players every Stock Round, and saved games replay from
        # the fresh game in this order
        g.seating = sorted(int(p.id) for p in raw.players)
        g.redraws = redraws
        g.trees = {}  # searched side -> its GenericSearchPlayer in this game
        g.searched = defaultdict(int)  # searched side -> decisions it searched
        return g

    games = [new_game(idx) for idx in game_indices]
    redraw_all_pass = settings.get("redraw_all_pass", True)

    def client(side):
        if settings["server"]:
            return gpg._CLIENTS[side]
        return _policy(settings["paths"][side])

    finished, active = [], list(games)
    while active:
        pending = []
        queue = list(active)
        while queue:
            g = queue.pop(0)
            legal = gpg._advance(g, play, settings)
            if legal is not None:
                pending.append((g, legal))
            elif redraw_all_pass and all_pass_opening(title, g.raw):
                if g.redraws < MAX_ALL_PASS_REDRAWS:
                    queue.append(new_game(g.idx, g.redraws + 1))  # every seat tied: not scored, replayed
                else:
                    g.termination = ALL_PASS
                    finished.append(g)
            else:
                finished.append(g)
        active = [g for g, _ in pending]
        if not pending:
            break
        by_side = defaultdict(list)
        for i, (g, _) in enumerate(pending):
            by_side[g.seat_models[decider_seat(g.raw, g.seating)]].append(i)
        for side, rows in by_side.items():
            for i in rows:  # a searched side's tree is kept only for its own next decision (~75 MB at 400 readouts)
                for other, player in pending[i][0].trees.items():
                    if other != side:
                        player.release()
        searched = [(side, i) for side in search_specs for i in by_side.pop(side, [])]
        policy_rows = sorted(i for rows in by_side.values() for i in rows)
        enc = encode_batch([pending[i][0].raw for i in policy_rows]) if policy_rows else None
        local = {i: j for j, i in enumerate(policy_rows)}
        sent = {}
        for side, rows in by_side.items():
            arrays = mg.add_legal_inputs(slice_positions(enc, [local[i] for i in rows]), [pending[i][1] for i in rows])
            sent[side] = (client(side).send_inputs(arrays, price_eps=settings["price_eps"]), rows, arrays)
        searches = []
        for side, i in searched:
            g, _ = pending[i]
            player = g.trees.get(side)
            if player is None:
                player = g.trees[side] = GenericSearchPlayer(
                    search_specs[side].config(n, softpick_move_cutoff=settings.get("softpick", 0))
                )
            player.follow(g.raw)
            searches.append((side, i, player.start_search(key=side)))
        if searches:
            searcher.run([s for _, _, s in searches if s is not None])
        for side, (request, rows, arrays) in sent.items():
            reply = client(side).receive_outputs(request)
            offsets = arrays["legal_offsets"]
            price_of_row = {int(r): j for j, r in enumerate(arrays["price_rows"])}
            for j, i in enumerate(rows):
                g, legal = pending[i]
                lo, hi = int(offsets[j]), int(offsets[j + 1])
                p = np.exp(gpg.tempered_logp(reply.logp[lo:hi], settings["temperatures"][side]))
                k = int(rng.choice(len(p), p=p / p.sum()))
                if play.open_price(legal, k):
                    cell = int(reply.price_cell[price_of_row[lo + k]])
                    price = int(legal.price_cells(k).sample_price(cell, rng))  # on the layout's grid
                else:
                    price = play.fixed_price(g.raw, legal, k)
                g.auction_decisions += g.raw.round.round_type == "Auction"
                if gpg.apply_move(g, play, legal, k, price):  # (a refused move ends the game: "rejected_move")
                    g.decisions += 1
        for side, i, _ in searches:
            g, legal = pending[i]
            player = g.trees[side]
            g.auction_decisions += g.raw.round.round_type == "Auction"
            actions = player.play(player.pick_move())
            try:
                for action in actions:  # the move at its price, then the forced moves after it
                    g.raw.process_action(dict(action))
            except RuntimeError as e:
                g.rejected = {"wire": actions[0] if actions else None, "error": str(e)[:300],
                              "action_count": len(g.raw.raw_actions)}  # fmt: skip
                g.termination = "rejected_move"
                continue
            g.decisions += 1
            g.searched[side] += 1

    records = []
    for g in finished:
        worth, win_share, fractions = gpg.game_outcome(g.raw, g.seating)
        record = {
            "idx": g.idx,
            "seats": list(g.seat_models),
            "win_share": win_share.tolist(),
            "worth_share": fractions.tolist(),
            "decisions": g.decisions,
            "auction_decisions": g.auction_decisions,
            "termination": g.termination,
            "start": g.start,
            "rules": g.rules,
            "optional_rules": list(getattr(g.raw, "optional_rules", None) or []),
            "all_pass_redrawn": g.redraws,
        }
        if g.trees:
            record["search"] = {
                side: {"decisions": g.searched[side], "kept_trees": player.kept} for side, player in g.trees.items()
            }
        if g.rejected:
            record["rejected_move"] = g.rejected
        if g.idx < settings.get("save_games", 0) or g.rejected:
            record["net_worth"] = worth.tolist()
            record["seed"] = g.seed
            record["players"] = gpg.seating_players(g.raw, g.seating)
            record["raw_actions"] = [dict(a) for a in g.raw.raw_actions]
        records.append(record)
    return records


def run_match(a_spec: str, b_spec: str, args) -> tuple:
    from eval_policy_only import arrangements, summarize

    from rl18xx.agent.alphazero import generic_pg as gpg
    from rl18xx.agent.alphazero.generic_inference import start_generic_inference_server
    from rl18xx.agent.alphazero.inference_server import wait_checking_servers

    sides = {"A": parse_side(a_spec, args.readouts), "B": parse_side(b_spec, args.readouts)}
    for side in sides.values():
        for path in side_paths(side):
            gpg.check_generic_checkpoint(path, args.title)
    policy_sides = {name: s for name, s in sides.items() if s["kind"] == "policy"}
    search_sides = {name: s for name, s in sides.items() if s["kind"] == "search"}
    handles = {}
    if args.server:
        servers = {name: s["path"] for name, s in policy_sides.items()}
        for name, s in search_sides.items():
            servers[f"{name}:policy"] = s["policy"]
            if s["value"]:
                servers[f"{name}:value"] = s["value"]
        handles = {
            key: start_generic_inference_server(num_workers=args.workers, checkpoint_path=path, batch_size=1024)
            for key, path in servers.items()
        }
    settings = {
        "title": args.title,
        "paths": {name: s["path"] for name, s in policy_sides.items()},
        "temperatures": {name: s["temperature"] for name, s in policy_sides.items()},
        "search": search_sides,
        "softpick": args.softpick + args.softpick % 2,
        "server": bool(args.server),
        "seed": args.seed,
        "start_positions": args.start_positions,
        "random_start_fraction": args.random_start_fraction,
        "auction_start": args.auction_start,
        "max_decisions": args.max_decisions,
        "auction_stall_decisions": args.auction_stall_decisions,
        "price_eps": args.price_eps,
        "save_games": args.save_games,
        "arrangements": arrangements(args.a_seats, args.num_players),
        "rules_mix": dict(args.rules_mix),
        "redraw_all_pass": not args.keep_all_pass,
    }
    seatings = settings["arrangements"]
    num_games = math.ceil(args.games / len(seatings)) * len(seatings)
    chunks = [list(range(i, min(i + args.games_per_task, num_games))) for i in range(0, num_games, args.games_per_task)]
    queues = {side: (h.request_q, h.reply_qs, h.ticket_q) for side, h in handles.items()}
    records, started = [], time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=gpg.worker_init, initargs=(queues,)) as pool:
            futures = {pool.submit(play_games, chunk, settings) for chunk in chunks}
            while futures:
                done, futures = wait_checking_servers(futures, handles, pool)
                for future in done:
                    records.extend(future.result())
    finally:
        for handle in handles.values():
            handle.shutdown()
    summary = summarize(a_spec, b_spec, records)
    summary["title"] = args.title
    summary["rules_mix"] = dict(args.rules_mix)
    summary["by_rules"] = by_rules(records, args.a_seats / args.num_players)
    summary["all_pass_redrawn"] = int(sum(r.get("all_pass_redrawn", 0) for r in records))
    summary["seconds"] = round(time.time() - started, 1)
    return summary, records


def by_rules(records: list, equal: float) -> dict:
    """Per rules combination: games, decisions (median / p90 / max), terminations and side A's score (equal =
    ``equal``)."""
    out: dict = {}
    for name in sorted({r.get("rules", "base") for r in records}):
        rs = [r for r in records if r.get("rules", "base") == name]
        decisions = np.array([r["decisions"] for r in rs])
        a_score = [sum(w for w, side in zip(r["win_share"], r["seats"]) if side == "A") for r in rs]
        terminations: dict = {}
        for r in rs:
            terminations[r["termination"]] = terminations.get(r["termination"], 0) + 1
        out[name] = {
            "games": len(rs),
            "decisions_median": float(np.median(decisions)),
            "decisions_p90": float(np.percentile(decisions, 90)),
            "decisions_max": int(decisions.max()),
            "terminations": terminations,
            "A_score": float(np.mean(a_score)),
            "A_score_equal": equal,
        }
    return out


def save_game_logs(out: Path, match_index: int, a: str, b: str, title: str, records: list) -> None:
    from rl18xx.agent.alphazero.game_records import GAME_RECORD_VERSION, save_game_record

    labels = {"A": a, "B": b}
    for r in sorted(records, key=lambda r: r["idx"]):
        actions = r.pop("raw_actions", None)
        if actions is None:
            continue
        name = f"m{match_index}_{r['idx']}"
        r["seat_labels"] = [labels[side] for side in r["seats"]]
        game = {
            "id": name,
            "title": title,
            "description": f"eval_generic {a} vs {b}, game {r['idx']}",
            "players": [{"id": pid, "name": pname} for pid, pname in r.pop("players")],
            "settings": {"optional_rules": list(r.get("optional_rules") or []), "seed": r.get("seed") or 0},
            "actions": actions,
            "rl18xx": {"version": GAME_RECORD_VERSION, "kind": "eval_generic", "match": f"{a} vs {b}", **r},
        }
        r["game_file"] = save_game_record(out, name, game)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", default="1830")
    parser.add_argument("--match", nargs=2, action="append", metavar=("A", "B"), required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--a-seats", type=int, default=None, help="Side A's seats (default half the table)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--games-per-task", type=int, default=30)
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--price-eps", type=float, default=0.0)
    parser.add_argument("--start-positions", default=None, help="SR1 start file (default: the title's for the count)")
    parser.add_argument("--random-start-fraction", type=float, default=0.2)
    parser.add_argument("--auction-start", action="store_true", help="Every game from the opening (real rules)")
    parser.add_argument("--auction-stall-decisions", type=int, default=150)
    parser.add_argument(
        "--keep-all-pass", action="store_true",
        help="The 1822 family: score games whose opening Stock Round ends with nothing bought, sold or bid (every "
        "seat tied) instead of replaying them from a fresh draw",
    )
    rules = parser.add_mutually_exclusive_group()
    rules.add_argument(
        "--rules", default=None, help="Every game under this rules combination (base, plus, plus+nr, ...)"
    )
    rules.add_argument(
        "--rules-mix", default=None,
        help="<name>:<weight>,... each start draws its rules from (default: the title's mix, "
        "titles.TitleSpec.rules_mix)",
    )
    parser.add_argument("--server", action="store_true", help="Each side on a generic inference server")
    parser.add_argument(
        "--readouts", type=int, default=200, help="A searched side's budget per decision when its spec has no @<n>"
    )
    parser.add_argument(
        "--softpick", type=int, default=0,
        help="A searched side samples its first N decisions in a game from the root visits (even; default 0: always "
        "the most visited move)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--save-games", type=int, default=0)
    args = parser.parse_args()
    from rl18xx.agent.alphazero.titles import title_spec

    spec = title_spec(args.title)
    spec.check_players(args.num_players)
    try:
        args.rules_mix = spec.parse_rules_mix(args.rules if args.rules is not None else args.rules_mix)
    except ValueError as e:
        parser.error(str(e))
    if args.a_seats is None:
        args.a_seats = args.num_players // 2
    if args.save_games and not args.out:
        parser.error("--save-games needs --out")
    if args.start_positions is None and not args.auction_start:
        args.start_positions = spec.start_positions(args.num_players)
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    summaries, all_records = [], []
    for match_index, (a, b) in enumerate(args.match):
        summary, records = run_match(a, b, args)
        summaries.append(summary)
        if args.out and (args.save_games or any(r.get("rejected_move") for r in records)):
            save_game_logs(Path(args.out), match_index, a, b, args.title, records)
        all_records.extend({"match": f"{a} vs {b}", **r} for r in records)
        print(
            f"[{args.title} {summary['players']}p {summary['a_seats']}v{summary['players'] - summary['a_seats']}] "
            f"{a} vs {b}: {summary['A_score']:.3f} ± {summary['A_score_se']:.3f} over {summary['games']} games, "
            f"equal = {summary['A_score_equal']:.2f} (worth share {summary['A_worth_share']:.3f}, "
            f"{summary['seconds']:.0f}s; all-pass openings redrawn: {summary['all_pass_redrawn']})",
            flush=True,
        )
        if len(summary["by_rules"]) > 1 or list(summary["by_rules"]) != ["base"]:
            for name, r in summary["by_rules"].items():
                print(
                    f"    {name:16} {r['games']:4} games, decisions median {r['decisions_median']:.0f} / max "
                    f"{r['decisions_max']}, {r['terminations']}, A {r['A_score']:.3f}",
                    flush=True,
                )
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        with (out / "games.jsonl").open("w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
