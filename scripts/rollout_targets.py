"""Rollout value targets (AlphaGo's value-net recipe) vs outcome targets, offline.

Instead of each position's single game outcome, a position's value target is the mean result of K fast
policy-only continuations from it (the policy at a temperature, prices from its price head, the rule against
train shuffling on, the evals' 1000-decision limit). Subcommands:

    rollouts   positions of saved search-learning games (search_data shards; one decision position past the
               auction per game, uniform over the game's decisions) rolled out K times each; per task a
               ``tasks/<id>.npz``: the root encoding (canonical), the game's own outcome, and every
               rollout's win share / net worth / length / termination
    sib-roots  sibling truth-set roots: decision positions of eval games (one per game)
    siblings   each root's children (all legal moves, or the top-8 by prior + 4 random), each child rolled
               out K times with common random numbers (rollout r of every child draws from seed (s, root, r))
    testset    eval collections -> 16 decision positions per game (fixed seed), canonical encodings + outcome
    games      whole policy-only games from the generators' starts (auction / mid-game / Stock Round 1 mix, the
               first decisions explored at T 1, as the search generators play them), saved as search_data shards
               (no searches) for ``free``; stops at a budget of policy decisions. ``--num-players`` (default 4;
               2 for 1830's two-player game) picks the start files too (human_games/*_1830_<N>p_v2.jsonl)
    free       outcome-labelled positions of every saved game (16 per game), as memory-mappable .npy files (one
               player count per directory: the targets have a column per seat)
    train      a value arm: warm start, AdamW, the value trainer's loss (win CE + placement MSE), a share of
               every batch from the anchor, a weight average; at most ~1.5 passes
    preds      value predictions of models on the test sets (and sibling children)

Run with the deployment's interpreter and PYTHONPATH at this checkout (same engine_rs):
    PYTHONPATH=<worktree> /home/revys/longrun/deploy/longrun-2026-11g/.venv/bin/python scripts/rollout_targets.py ...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("rollout_targets")
NUM_PLAYERS = 4  # the default player count; everything else follows each game's own count
SERVER = "policy"
DEFAULT_POLICY = "/home/revys/longrun/work/lr1/model_checkpoints_search/azf/learner/600.pth"
DEFAULT_DATA = "/home/revys/longrun/work/lr1/saved_games/azf"


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def walk_decisions(rs, actions: list, stop: int | None = None):
    """Replay ``actions`` on engine ``rs`` (live), yielding ``(i, is_decision, decisions_before)`` before each
    action ``i`` (up to ``stop`` exclusive, then once more at ``stop`` without applying anything):
    ``is_decision`` = past the auction with more than one legal move (the value trainer's positions);
    ``decisions_before`` counts the positions past the auction where the mover had a choice (more than one
    legal move, or one whose price is open) -- what the evals' decision limit counts."""
    from rl18xx.agent.alphazero.policy_selfplay import has_price_choice

    n = len(actions) if stop is None else stop
    decisions = 0
    for i in range(n + 1):
        if rs.finished:
            yield i, False, decisions
            return
        auction = rs.round.round_type == "Auction"
        legal = [] if auction else rs.factored_legal_indices()
        is_decision = (not auction) and len(legal) > 1
        yield i, is_decision, decisions
        if i == n:
            return
        if not auction and (len(legal) > 1 or (len(legal) == 1 and has_price_choice(rs, legal[0]))):
            decisions += 1
        rs.process_action(actions[i])


def saved_game_root(game, seed: int):
    """``(rs, index, decisions_before, n_actions)``: one decision position of a saved game, uniform over its
    decisions past the auction, drawn by ``(crc32(uid), seed)``; None if it has none."""
    from rl18xx.agent.alphazero.start_positions import new_game

    meta = game.meta
    rule = bool(meta.get("no_buyback_rule", False))
    actions = game.actions

    def fresh():
        return new_game(int(meta["num_players"]), bool(meta["auction_unlock"]), no_buyback_rule=rule)._game

    rs = fresh()
    cands = [(i, d) for i, dec, d in walk_decisions(rs, actions) if dec]
    if not cands:
        return None
    rng = np.random.default_rng([zlib.crc32(meta["uid"].encode()), int(seed)])
    i, d = cands[int(rng.integers(len(cands)))]
    rs = fresh()
    for _ in walk_decisions(rs, actions, stop=i):
        pass
    return rs, i, d, len(actions)


def eval_game_roots(actions: list, game_id: str, k: int, seed: int, unlock: bool = False,
                    num_players: int = NUM_PLAYERS):
    """``[(index, decisions_before, encoding)]`` for ``k`` decision positions of an eval game (from game start,
    the rule on), drawn by ``(crc32(game_id), seed)``; plus the number of actions."""
    from rl18xx.agent.alphazero.start_positions import new_game

    rs = new_game(num_players, unlock)._game
    cands = [(i, d) for i, dec, d in walk_decisions(rs, actions) if dec]
    rng = np.random.default_rng([zlib.crc32(game_id.encode()), int(seed)])
    picked = sorted(rng.choice(len(cands), size=min(k, len(cands)), replace=False).tolist()) if cands else []
    wanted = {cands[j][0]: cands[j][1] for j in picked}
    rs = new_game(num_players, unlock)._game
    out = []
    for i, action in enumerate(actions):
        if i in wanted:
            state, nodes, _, rotation, _ = rs.encode_canonical_py()
            out.append((i, wanted[i], np.asarray(state, np.float32), np.asarray(nodes).astype(np.float16),
                        int(rotation)))
        if i >= max(wanted, default=-1):
            break
        rs.process_action(action)
    return out, len(actions)


def canon(x: np.ndarray, rotation: int) -> np.ndarray:
    """Absolute seat order -> canonical frame (the mover first): ``out[j] = x[(j + rotation) % n]``, ``n`` the
    seats (the last axis)."""
    x = np.asarray(x)
    n = x.shape[-1]
    idx = (np.arange(n) + int(rotation)) % n
    return x[..., idx]


# ---------------------------------------------------------------------------
# Rollouts (pool workers)
# ---------------------------------------------------------------------------


def _new_rollout(rs, decisions: int, rng, uid: str):
    from rl18xx.rust_adapter import RustGameAdapter

    return SimpleNamespace(game=RustGameAdapter(rs.pickle_clone()), uid=uid, decisions=int(decisions),
                           start_decisions=int(decisions), price_row=None, termination=None, rng=rng)


def play_out(games: list, settings: dict) -> None:
    """Play every game to its end in lockstep through the worker's policy server: moves sampled from the
    policy at ``settings['temperature']`` (at ``g.explore_temperature`` while ``g.decisions`` is below
    ``g.explore_until``, for games that have them), prices from its price head; ``g.termination`` is set
    (finished, max_length, error)."""
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.policy_selfplay import _advance

    client = pg._CLIENTS[SERVER]
    temperature = float(settings["temperature"])
    active = list(games)
    while active:
        pending = []
        for g in active:
            try:
                legal = _advance(g, settings, g.rng)
            except Exception as e:  # an engine error: the rollout is dropped
                LOGGER.warning(f"{g.uid}: {e!r}")
                g.termination = "error"
                continue
            if legal is not None:
                pending.append((g, legal))
        active = [g for g, _ in pending]
        if not pending:
            break
        encoded = [_rust_encode(g.game) for g, _ in pending]
        probs, _, _ = client.receive(client.send(encoded, legal_indices=[legal for _, legal in pending]))
        price = client.last_price_components
        price_logits = price["price_logits"].numpy() if price is not None else None
        for j, (g, legal) in enumerate(pending):
            p = probs[j].numpy()[legal].astype(np.float64)
            t = g.explore_temperature if g.decisions < getattr(g, "explore_until", 0) else temperature
            if t != 1.0:
                p = np.power(np.clip(p, 1e-12, None), 1.0 / t)
            p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
            choice = legal[int(g.rng.choice(len(legal), p=p))]
            g.price_row = price_logits[j] if price_logits is not None else None
            try:
                value, _, _ = pg.choose_price(g.game, choice, g.price_row, g.rng, 0.0)
                g.game._game.apply_action_index(choice, value)
            except Exception as e:
                LOGGER.warning(f"{g.uid}: {e!r}")
                g.termination = "error"
                active.remove(g)
                continue
            g.decisions += 1


def rollout_result(g) -> tuple:
    """``(win share, net worth, rollout decisions, termination code)``; NaNs for an errored rollout."""
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.self_play import _compute_net_worth

    codes = {"finished": 0, "max_length": 1, "error": 2}
    if g.termination == "error":
        n = len(g.game._game.players)
        return np.full(n, np.nan), np.full(n, np.nan), g.decisions - g.start_decisions, 2
    win, _ = pg.outcome(g.game)
    nw = _compute_net_worth(g.game)
    worth = np.array([float(nw[pid]) for pid in sorted(nw)], dtype=np.float64)
    return win, worth, g.decisions - g.start_decisions, codes.get(g.termination, 3)


_SHARD_CACHE: dict = {}


def _saved_games(path: str) -> list:
    from rl18xx.agent.alphazero import search_data as sd

    if path not in _SHARD_CACHE:
        _SHARD_CACHE.clear()
        _SHARD_CACHE[path] = sd.read_shard(path)
    return _SHARD_CACHE[path]


def rollout_task(task: dict) -> dict:
    """Pool task: positions of saved games, each rolled out ``K`` times."""
    t0 = time.time()
    K, seed = int(task["K"]), int(task["seed"])
    settings = {"temperature": task["temperature"], "max_decisions": task["max_decisions"], "price_eps": 0.0}
    roots, games = [], []
    for pos_id, path, slot in task["items"]:
        game = _saved_games(path)[slot]
        try:
            found = saved_game_root(game, seed)
        except Exception as e:
            LOGGER.warning(f"{game.uid}: not replayed ({e!r})")
            continue
        if found is None:
            continue
        rs, index, decisions, n = found
        state, nodes, _, rotation, _ = rs.encode_canonical_py()
        roots.append(dict(pos_id=pos_id, uid=game.uid, index=index, n_actions=n, decisions=decisions,
                          rotation=int(rotation), state=np.asarray(state, np.float32),
                          nodes=np.asarray(nodes).astype(np.float16),
                          win=np.asarray(game.meta["win_share"], np.float32),
                          worth=np.asarray(game.meta["net_worth"], np.float64),
                          rollouts=[_new_rollout(rs, decisions, np.random.default_rng([seed, pos_id, r]),
                                                 f"{game.uid}@{index}#{r}") for r in range(K)]))
        games.extend(roots[-1]["rollouts"])
    t1 = time.time()
    play_out(games, settings)
    out = {"pos_id": [], "uid": [], "index": [], "n_actions": [], "decisions": [], "rotation": [], "game_states": [],
           "nodes": [], "win": [], "worth": [], "r_win": [], "r_worth": [], "r_len": [], "r_term": []}
    for r in roots:
        res = [rollout_result(g) for g in r["rollouts"]]
        for key, src in (("pos_id", "pos_id"), ("uid", "uid"), ("index", "index"), ("n_actions", "n_actions"),
                         ("decisions", "decisions"), ("rotation", "rotation"), ("game_states", "state"),
                         ("nodes", "nodes"), ("win", "win"), ("worth", "worth")):
            out[key].append(r[src])
        out["r_win"].append(np.stack([x[0] for x in res]))
        out["r_worth"].append(np.stack([x[1] for x in res]))
        out["r_len"].append(np.array([x[2] for x in res]))
        out["r_term"].append(np.array([x[3] for x in res]))
    arrays = {k: np.asarray(v) for k, v in out.items()} if roots else {}
    return {"arrays": arrays, "replay_seconds": t1 - t0, "seconds": time.time() - t0,
            "decisions": int(sum(g.decisions - g.start_decisions for g in games)), "rollouts": len(games)}


def sibling_task(task: dict) -> dict:
    """Pool task: sibling roots (eval-game positions), their children, each child rolled out ``K`` times with
    common random numbers across children."""
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero import price_pmf
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.policy_selfplay import _advance
    from rl18xx.agent.alphazero.start_positions import new_game
    from rl18xx.rust_adapter import RustGameAdapter

    t0 = time.time()
    K, seed, max_children = int(task["K"]), int(task["seed"]), int(task["max_children"])
    settings = {"temperature": task["temperature"], "max_decisions": task["max_decisions"], "price_eps": 0.0}
    client = pg._CLIENTS[SERVER]
    parents = []
    for item in task["items"]:
        pos_id, game_file, index, game_id = item[:4]
        players = int(item[4]) if len(item) > 4 else NUM_PLAYERS  # roots files before 2p support: 4
        actions = json.loads(Path(game_file).read_text())
        actions = actions.get("actions", []) if isinstance(actions, dict) else actions
        rs = new_game(players)._game
        decisions = 0
        for i, dec, d in walk_decisions(rs, actions, stop=index):
            decisions = d
        parents.append(dict(pos_id=pos_id, game_id=game_id, index=index, n_actions=len(actions), decisions=decisions,
                            rs=rs, legal=[int(x) for x in rs.factored_legal_indices()]))
    if not parents:
        return {"arrays": {}, "seconds": 0.0, "decisions": 0, "rollouts": 0}
    encoded = [_rust_encode(RustGameAdapter(p["rs"])) for p in parents]
    probs, _, _ = client.receive(client.send(encoded, legal_indices=[p["legal"] for p in parents]))
    price = client.last_price_components
    price_logits = price["price_logits"].numpy() if price is not None else None
    games, children = [], []
    for j, p in enumerate(parents):
        legal = p["legal"]
        prior = probs[j].numpy()[legal].astype(np.float64)
        prior = prior / prior.sum() if prior.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
        rng = np.random.default_rng([seed, p["pos_id"], 10**6])
        keep = list(range(len(legal)))
        if len(legal) > max_children:
            top = [int(k) for k in np.argsort(-prior)[: max_children - 4]]
            rest = [k for k in range(len(legal)) if k not in set(top)]
            keep = sorted(top + [int(k) for k in rng.choice(rest, size=4, replace=False)])
        state, nodes, _, rotation, _ = p["rs"].encode_canonical_py()
        p.update(rotation=int(rotation), state=np.asarray(state, np.float32), nodes=np.asarray(nodes).astype(np.float16))
        row = price_logits[j] if price_logits is not None else None
        for k in keep:
            ix = legal[k]
            c = p["rs"].pickle_clone()
            # An open price: the price head's most likely cell, a price inside it (seeded per child).
            pr, cell_price = c.price_range_for_index(int(ix)), None
            if pr is not None:
                slot = c.price_head_slot_for_index(int(ix))
                if pr[0] == pr[1] or slot is None:
                    cell_price = int(pr[0])
                else:
                    si, lo, hi = slot
                    cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[si], int(lo), int(hi))
                    cp = cells.cell_probs(row[si] if row is not None else None)
                    cp = np.where(cells.nonempty, cp, -1.0)
                    cell_price = int(cells.sample_price(int(np.argmax(cp)), np.random.default_rng([seed, p["pos_id"], k])))
            child = dict(pos_id=p["pos_id"], move=ix, prior=float(prior[k]), price=-1 if cell_price is None else cell_price)
            try:
                c.apply_action_index(int(ix), cell_price)
                g0 = _new_rollout(c, p["decisions"] + 1, np.random.default_rng([seed, p["pos_id"], 0]), "probe")
                g0.game = RustGameAdapter(c)
                legal_after = _advance(g0, settings, g0.rng)  # forced follow-ups, as the search collapses them
            except Exception as e:
                LOGGER.warning(f"{p['game_id']}@{p['index']} child {ix}: {e!r}")
                continue
            child["terminal"] = legal_after is None
            if legal_after is None:
                win, worth, _, _ = rollout_result(g0)
                child.update(rotation=0, state=np.zeros_like(p["state"]), nodes=np.zeros_like(p["nodes"]),
                             rollouts=[], fixed=(win, worth))
            else:
                s2, n2, _, r2, _ = c.encode_canonical_py()
                child.update(rotation=int(r2), state=np.asarray(s2, np.float32), nodes=np.asarray(n2).astype(np.float16),
                             rollouts=[_new_rollout(c, g0.decisions, np.random.default_rng([seed, p["pos_id"], r]),
                                                    f"{p['game_id']}@{p['index']}:{ix}#{r}") for r in range(K)])
                games.extend(child["rollouts"])
            children.append(child)
    play_out(games, settings)
    out = {"c_pos_id": [], "c_move": [], "c_prior": [], "c_price": [], "c_terminal": [], "c_rotation": [],
           "c_states": [], "c_nodes": [], "c_win": [], "c_worth": [], "c_len": [], "c_term": []}
    for ch in children:
        if ch["terminal"]:
            win, worth = ch["fixed"]
            res = [(win, worth, 0, 0)] * K
        else:
            res = [rollout_result(g) for g in ch["rollouts"]]
        for key, src in (("c_pos_id", "pos_id"), ("c_move", "move"), ("c_prior", "prior"), ("c_price", "price"),
                         ("c_terminal", "terminal"), ("c_rotation", "rotation"), ("c_states", "state"),
                         ("c_nodes", "nodes")):
            out[key].append(ch[src])
        out["c_win"].append(np.stack([x[0] for x in res]))
        out["c_worth"].append(np.stack([x[1] for x in res]))
        out["c_len"].append(np.array([x[2] for x in res]))
        out["c_term"].append(np.array([x[3] for x in res]))
    arrays = {k: np.asarray(v) for k, v in out.items()} if children else {}
    for key in ("pos_id", "index", "n_actions", "decisions", "rotation"):
        arrays[f"p_{key}"] = np.asarray([p[key] for p in parents])
    arrays["p_game_id"] = np.asarray([p["game_id"] for p in parents])
    arrays["p_states"] = np.stack([p["state"] for p in parents])
    arrays["p_nodes"] = np.stack([p["nodes"] for p in parents])
    arrays["p_legal_count"] = np.asarray([len(p["legal"]) for p in parents])
    return {"arrays": arrays, "seconds": time.time() - t0,
            "decisions": int(sum(g.decisions - g.start_decisions for g in games)), "rollouts": len(games)}


# ---------------------------------------------------------------------------
# Driver: a pool of workers on one policy inference server
# ---------------------------------------------------------------------------


def run_pool(fn, tasks: list, out_dir: Path, policy: str, workers: int, replicas: int, deadline: float | None,
             budget: int | None = None):
    """Run ``fn`` over ``tasks`` (dicts with an ``id``), each result's arrays saved to
    ``out_dir/tasks/<id>.npz`` (tasks already there are skipped); stops submitting at ``deadline``, or once this
    run's finished tasks hold ``budget`` decisions."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.inference_server import start_inference_server, wait_checking_servers

    multiprocessing.set_start_method("spawn", force=True)
    (out_dir / "tasks").mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (out_dir / "tasks" / f"{t['id']:06d}.npz").exists()]
    LOGGER.info(f"{len(tasks)} tasks, {len(todo)} to do")
    if not todo:
        return
    handle = start_inference_server(num_workers=workers, model_factory=pg.load_server_model, checkpoint_path=policy,
                                    batch_size=1024, autocast_device=None, replicas=replicas)
    queues = {SERVER: (handle.request_q, handle.reply_qs, handle.ticket_q)}
    started, done_n, rollouts, decisions, worker_s = time.time(), 0, 0, 0, 0.0
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=pg.worker_init, initargs=(queues,)) as pool:
            it = iter(todo)
            futures = {}
            for t in it:
                futures[pool.submit(fn, t)] = t
                if len(futures) >= 2 * workers:
                    break
            while futures:
                done, _ = wait_checking_servers(set(futures), {SERVER: handle}, pool)
                for f in done:
                    t = futures.pop(f)
                    res = f.result()
                    if res["arrays"]:
                        tmp = out_dir / "tasks" / f"{t['id']:06d}.tmp.npz"
                        np.savez(tmp, **res["arrays"])
                        os.replace(tmp, out_dir / "tasks" / f"{t['id']:06d}.npz")
                    done_n += 1
                    rollouts += res["rollouts"]
                    decisions += res["decisions"]
                    worker_s += res["seconds"]
                    if (deadline is None or time.time() < deadline) and (budget is None or decisions < budget):
                        nxt = next(it, None)
                        if nxt is not None:
                            futures[pool.submit(fn, nxt)] = nxt
                el = time.time() - started
                if done:
                    LOGGER.info(f"{done_n}/{len(todo)} tasks, {rollouts} rollouts ({rollouts / el * 3600:.0f}/h), "
                                f"{decisions / el:.0f} decisions/s, {el / 60:.1f} min")
    finally:
        handle.shutdown()
    status = {"tasks_done": done_n, "rollouts": rollouts, "decisions": decisions, "seconds": time.time() - started}
    (out_dir / f"run_{int(started)}.json").write_text(json.dumps(status, indent=1))


def cmd_rollouts(args):
    from rl18xx.agent.alphazero import search_data as sd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sd.shard_paths(args.data)
    # Every game (path, slot), then a seeded sample of games; tasks grouped by shard.
    index = []
    for p in paths:
        n = len(sd.read_shard(p)) if args.games_per_shard is None else args.games_per_shard
        index.extend((str(p), s) for s in range(n))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(index))[args.skip : args.skip + args.games]
    chosen = sorted(order.tolist(), key=lambda g: index[g])
    items = [(int(g), index[g][0], index[g][1]) for g in chosen]
    tasks = []
    for b in range(0, len(items), args.positions_per_task):
        tasks.append({"id": b // args.positions_per_task, "items": items[b : b + args.positions_per_task],
                      "K": args.K, "seed": args.seed, "temperature": args.temperature, "max_decisions": args.max_decisions})
    # Shuffle task order so a partial run is a uniform sample.
    perm = np.random.default_rng(args.seed + 1).permutation(len(tasks))
    tasks = [tasks[i] for i in perm]
    (out / "config.json").write_text(json.dumps({**vars(args), "games_total": len(index), "tasks": len(tasks)}, indent=1))
    deadline = time.time() + args.hours * 3600 if args.hours else None
    run_pool(rollout_task, tasks, out, args.policy, args.workers, args.replicas, deadline)


def games_task(task: dict) -> dict:
    """Pool task: whole policy-only games ``task['games']`` (game numbers), each from the start a generator would
    draw -- ``replayed_start`` with the task's mix, seeded by ``(seed, game number)``, so every policy given the
    same seed plays the same starts -- the first ``explore_decisions`` decisions counted from the real game start
    at ``explore_temperature`` (as ``search_learning --explore-from-game-start``), the rest at ``temperature``;
    written as one search_data shard ``task['shards']/<id>.sld`` (no searches, no rows: ``free`` reads them)."""
    import random

    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero import search_data as sd
    from rl18xx.agent.alphazero.search_learning import start_decisions
    from rl18xx.agent.alphazero.self_play import _compute_net_worth
    from rl18xx.agent.alphazero.start_positions import AUCTION_START, replayed_start

    t0 = time.time()
    seed = int(task["seed"])
    players = int(task.get("num_players", NUM_PLAYERS))
    settings = {"temperature": task["temperature"], "max_decisions": task["max_decisions"], "price_eps": 0.0}
    games = []
    for i in task["games"]:
        start, game, _ = replayed_start(
            players, task["start_positions"], task["random_start_fraction"], rng=random.Random(f"{seed}:{i}"),
            midgame_path=task["midgame_positions"], midgame_fraction=task["midgame_fraction"],
            auction_fraction=task["auction_fraction"])
        explored = start_decisions(start.actions, players)
        games.append(SimpleNamespace(
            game=game, uid=f"{seed}-{i:07d}", decisions=0, start_decisions=0, price_row=None, termination=None,
            rng=np.random.default_rng([seed, i]), start=start.label, start_actions=len(start.actions),
            explore_until=int(task["explore_decisions"]) - explored, explore_temperature=task["explore_temperature"]))
    play_out(games, settings)
    empty = {name: np.zeros(0, dtype=dtype) for name, (dtype, _) in sd.COLUMNS.items()}
    empty["search_value"] = np.zeros((0, players), dtype=np.float32)
    empty["price_cells"] = np.zeros((0, task["price_cells"]), dtype=np.float32)
    packed = []
    for g in games:
        if g.termination == "error":
            continue
        win, _ = pg.outcome(g.game)
        nw = _compute_net_worth(g.game)
        worth = np.array([float(nw[pid]) for pid in sorted(nw)], dtype=np.float64)
        total = float(worth.clip(min=0).sum())
        fractions = worth.clip(min=0) / total if total > 0 else np.full(players, 1.0 / players)
        actions = list(g.game._game.raw_actions)
        meta = {
            "uid": g.uid, "start": g.start, "start_kind": g.start.split(":")[0] if g.start else None,
            "start_actions": g.start_actions, "auction_unlock": g.start == AUCTION_START,
            "no_buyback_rule": bool(g.game._game.no_buyback_rule), "num_players": players,
            "termination": g.termination, "decisions": int(g.decisions), "searched": 0,
            "seat_servers": ["policy"] * players, "seat_members": ["main"] * players, "actions": len(actions),
            "win_share": [float(x) for x in win], "fractions": [float(x) for x in fractions],
            "net_worth": [float(x) for x in worth], "score_target": None, "aux_seat": [], "aux_global": [],
            "value_version": None,
        }  # fmt: skip
        packed.append({"meta": meta, "actions": json.dumps(actions, separators=(",", ":")).encode(),
                       "columns": dict(empty)})
    if packed:
        sd.write_shard(Path(task["shards"]) / f"{task['id']:06d}{sd.SHARD_SUFFIX}", packed, task["config"])
    decisions = int(sum(g.decisions for g in games))
    return {"arrays": {"games": np.array([len(packed)]), "decisions": np.array([decisions])},
            "seconds": time.time() - t0, "decisions": decisions, "rollouts": len(packed)}


def cmd_games(args):
    from rl18xx.agent.alphazero import price_pmf
    from rl18xx.agent.alphazero.start_positions import (
        check_start_files,
        default_midgame_positions,
        default_start_positions,
    )

    # The player count's start files unless given (4 players: the files the loop has always used).
    args.start_positions = args.start_positions or default_start_positions(args.num_players)
    args.midgame_positions = args.midgame_positions or default_midgame_positions(args.num_players)
    if not 2 <= args.num_players <= 6:
        raise SystemExit(f"--num-players must be 2-6, got {args.num_players}")
    check_start_files(args.num_players, args.start_positions, args.random_start_fraction, args.midgame_positions,
                      args.midgame_fraction)
    out = Path(args.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "kind": "policy_games"}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    common = {key: getattr(args, key) for key in (
        "seed", "temperature", "max_decisions", "explore_decisions", "explore_temperature", "start_positions",
        "random_start_fraction", "midgame_positions", "midgame_fraction", "auction_fraction",
        "num_players")}  # fmt: skip
    common.update(shards=str(out / "shards"), config=config, price_cells=price_pmf.NUM_CELLS)
    tasks = [{"id": b // args.games_per_task, "games": list(range(b, min(args.games, b + args.games_per_task))),
              **common} for b in range(0, args.games, args.games_per_task)]  # in game order: a prefix is paired
    done = [np.load(f) for f in sorted((out / "tasks").glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    spent = int(sum(int(z["decisions"][0]) for z in done))
    LOGGER.info(f"{len(done)} tasks done before ({spent} decisions of the {args.decisions} budget)")
    if spent < args.decisions:
        run_pool(games_task, tasks, out, args.policy, args.workers, args.replicas, None, args.decisions - spent)


def cmd_sib_roots(args):
    """One decision position per eval game (fixed seed), the games spread over the collections."""
    from rl18xx.agent.alphazero import value_refit as vr

    rng = np.random.default_rng(args.seed)
    games = []
    for c in args.collection:
        for gid, actions, record in vr.collection_games(Path(c)):
            games.append((gid, str(Path(c) / "games" / f"{record['game_idx']}.json"), len(actions),
                          len(record["win_share"])))
    pick = sorted(rng.choice(len(games), size=min(args.positions, len(games)), replace=False).tolist())
    roots = []
    for j, g in enumerate(pick):
        gid, path, n, players = games[g]
        actions = json.loads(Path(path).read_text())
        actions = actions.get("actions", []) if isinstance(actions, dict) else actions
        from rl18xx.agent.alphazero.start_positions import new_game

        rs = new_game(players)._game
        cands = [i for i, dec, _ in walk_decisions(rs, actions) if dec]
        if not cands:
            continue
        i = int(np.random.default_rng([zlib.crc32(gid.encode()), args.seed]).choice(cands))
        roots.append({"pos_id": j, "game_id": gid, "game_file": path, "index": i, "n_actions": n,
                      "num_players": players})
    Path(args.out).write_text(json.dumps(roots))
    LOGGER.info(f"{len(roots)} roots -> {args.out}")


def cmd_siblings(args):
    roots = json.loads(Path(args.roots).read_text())
    tasks = []
    for b in range(0, len(roots), args.positions_per_task):
        chunk = roots[b : b + args.positions_per_task]
        tasks.append({"id": b // args.positions_per_task, "K": args.K, "seed": args.seed, "temperature": args.temperature,
                      "max_decisions": args.max_decisions, "max_children": args.max_children,
                      "items": [(r["pos_id"], r["game_file"], r["index"], r["game_id"],
                                 r.get("num_players", NUM_PLAYERS)) for r in chunk]})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    deadline = time.time() + args.hours * 3600 if args.hours else None
    run_pool(sibling_task, tasks, out, args.policy, args.workers, args.replicas, deadline)


# ---------------------------------------------------------------------------
# Test sets and free outcome data (CPU pools, no network)
# ---------------------------------------------------------------------------


def _testset_task(job: tuple) -> dict:
    gid, game_file, record, k, seed = job
    from rl18xx.agent.alphazero.policy_gradient import placement_points

    actions = json.loads(Path(game_file).read_text())
    actions = actions.get("actions", []) if isinstance(actions, dict) else actions
    try:
        rows, n = eval_game_roots(actions, gid, k, seed, num_players=len(record["win_share"]))
    except Exception as e:
        return {"gid": gid, "error": repr(e)}
    win = np.asarray(record["win_share"], np.float32)
    place = placement_points(np.asarray(record["net_worth"], np.float64)).astype(np.float32)
    return {
        "gid": gid,
        "index": np.array([r[0] for r in rows], np.int64),
        "decisions": np.array([r[1] for r in rows], np.int64),
        "game_states": np.stack([r[2] for r in rows]) if rows else None,
        "nodes": np.stack([r[3] for r in rows]) if rows else None,
        "rotation": np.array([r[4] for r in rows], np.int8),
        "stage": np.array([min(2, 3 * r[0] // max(n, 1)) for r in rows], np.int8),
        "win": np.stack([canon(win, r[4]) for r in rows]) if rows else None,
        "placement": np.stack([canon(place, r[4]) for r in rows]) if rows else None,
    }


def cmd_testset(args):
    from multiprocessing import get_context

    from rl18xx.agent.alphazero import value_refit as vr

    jobs = []
    for c in args.collection:
        c = Path(c)
        if args.require_summary and not (c / "summary.json").exists():
            LOGGER.warning(f"{c}: no summary.json, skipped")
            continue
        for line in (c / "games.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            path = c / "games" / f"{record['game_idx']}.json"
            if path.exists():
                jobs.append((f"{c.name}/{record['game_idx']}", str(path), record, args.per_game, args.seed))
    LOGGER.info(f"{len(jobs)} games")
    parts, gids, errors = [], [], []
    with get_context("spawn").Pool(args.workers) as pool:
        for res in pool.imap(_testset_task, jobs, chunksize=4):
            if "error" in res:
                errors.append(res)
                continue
            if res["game_states"] is None:
                continue
            res["game"] = np.full(len(res["index"]), len(gids), np.int32)
            gids.append(res["gid"])
            parts.append(res)
    keys = ("index", "decisions", "game_states", "nodes", "rotation", "stage", "win", "placement", "game")
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "positions.npz", **arrays, game_id=np.asarray(gids))
    (out / "meta.json").write_text(json.dumps({"games": len(gids), "positions": int(len(arrays["win"])),
                                               "errors": errors[:20], "n_errors": len(errors), "args": vars(args)}))
    LOGGER.info(f"{len(gids)} games, {len(arrays['win'])} positions, {len(errors)} errors -> {out}")


def _free_task(job: tuple) -> dict:
    path, slots, k, pass_id = job
    from rl18xx.agent.alphazero import value_trainer as vt
    from rl18xx.agent.alphazero.policy_gradient import placement_points

    games = _saved_games(path)
    out = []
    for slot in slots:
        g = games[slot]
        try:
            pos = vt.game_positions(g.meta, g.actions, k, pass_id)
        except Exception:
            continue
        if pos["game_states"] is None:
            continue
        win = np.asarray(g.meta["win_share"], np.float32)
        place = placement_points(np.asarray(g.meta["net_worth"], np.float64)).astype(np.float32)
        n = len(g.actions)
        out.append({
            "uid": g.uid, "game_states": pos["game_states"], "nodes": pos["nodes"], "rotation": pos["rotation"],
            "index": pos["index"], "stage": np.minimum(2, 3 * pos["index"] // max(n, 1)).astype(np.int8),
            "win": np.stack([canon(win, r) for r in pos["rotation"]]),
            "placement": np.stack([canon(place, r) for r in pos["rotation"]]),
        })
    return {"games": out}


def cmd_free(args):
    """Outcome-labelled positions (value_trainer.game_positions: ``k`` per game, pass ``pass_id``) of every saved
    game, written as .npy (game_states float32, nodes float16, win, placement, game, ...) to memory-map."""
    from multiprocessing import get_context

    from rl18xx.agent.alphazero import search_data as sd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sd.shard_paths(args.data)
    jobs = []
    for p in paths:
        n = len(sd.read_shard(p))
        for b in range(0, n, 32):
            jobs.append((str(p), list(range(b, min(n, b + 32))), args.per_game, args.pass_id))
    est = len(jobs) * 32 * args.per_game
    LOGGER.info(f"{len(jobs)} jobs, <= {est} positions")
    # Allocated at the first game: a state's width is its player count's encoder layout (394 with 4 players).
    mm = {}
    small = {k: [] for k in ("rotation", "index", "stage", "win", "placement", "game")}
    uids, at, t0 = [], 0, time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for j, res in enumerate(pool.imap(_free_task, jobs, chunksize=1)):
            for g in res["games"]:
                if not mm:
                    mm = {
                        "game_states": np.lib.format.open_memmap(
                            out / "game_states.npy", "w+", np.float32, (est, g["game_states"].shape[1])),
                        "nodes": np.lib.format.open_memmap(out / "nodes.npy", "w+", np.float16, (est, 93, 50)),
                    }
                if g["game_states"].shape[1] != mm["game_states"].shape[1]:
                    raise SystemExit(f"{g['uid']}: {g['win'].shape[1]}-player positions among others; one player "
                                     "count per directory")
                n = len(g["index"])
                mm["game_states"][at : at + n] = g["game_states"]
                mm["nodes"][at : at + n] = g["nodes"]
                for k in ("rotation", "index", "stage", "win", "placement"):
                    small[k].append(g[k])
                small["game"].append(np.full(n, len(uids), np.int32))
                uids.append(g["uid"])
                at += n
            if j % 200 == 0:
                LOGGER.info(f"{j}/{len(jobs)} jobs, {len(uids)} games, {at} positions, {time.time() - t0:.0f}s")
    for k, v in mm.items():
        v.flush()
    del mm
    for k, v in small.items():
        np.save(out / f"{k}.npy", np.concatenate(v))
    np.save(out / "uid.npy", np.asarray(uids))
    (out / "meta.json").write_text(json.dumps({"positions": at, "games": len(uids), "allocated": est, "args": vars(args)}))
    LOGGER.info(f"{len(uids)} games, {at} positions -> {out}")


# ---------------------------------------------------------------------------
# Arm data
# ---------------------------------------------------------------------------

VALUE_GATE = Path("/home/revys/longrun/work/lr1/logs/value_gate/data")
LADDER = Path("/home/revys/longrun/work/lr1/logs/value_gate/ladder_lr1_u0")
V4 = "/home/revys/longrun/work/lr1/model_checkpoints_value/lr1/promoted/4.pth"


def val_game(uid: str) -> bool:
    """5% of the azf games: the rollout set's validation positions, out of every arm's training."""
    return zlib.crc32(str(uid).encode()) % 20 == 0


def load_rollouts(d) -> dict:
    files = [f for f in sorted((Path(d) / "tasks").glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    parts = [dict(np.load(f)) for f in files]
    a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(a["pos_id"], kind="stable")
    return {k: v[order] for k, v in a.items()}


def placement_rows(worth: np.ndarray) -> np.ndarray:
    """Placement points per row of net worths (..., players); NaN rows stay NaN."""
    from rl18xx.agent.alphazero.policy_gradient import placement_points

    flat = worth.reshape(-1, worth.shape[-1])
    out = np.full(flat.shape, np.nan)
    for i, w in enumerate(flat):
        if not np.isnan(w).any():
            out[i] = placement_points(w)
    return out.reshape(worth.shape)


def canon_rows(x: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    idx = (np.arange(n)[None, :] + rotation.astype(np.int64)[:, None]) % n
    return np.take_along_axis(x, idx, axis=1)


def rollout_arrays(a: dict, target: str) -> dict:
    """Training arrays of a rollout set under ``target``: ``outcome`` (the game's own result), ``rollout`` (the mean
    of every valid rollout) or ``rollout<k>`` (the mean of the first k); positions without a valid rollout dropped."""
    rot = a["rotation"]
    keep = (~np.isnan(a["r_win"]).any(-1)).any(1)  # the same positions under every target
    if target == "outcome":
        win, place = a["win"].astype(np.float64), placement_rows(a["worth"])
    else:
        k = int(target[len("rollout"):]) if target != "rollout" else a["r_win"].shape[1]
        rw, rp = a["r_win"][:, :k], placement_rows(a["r_worth"][:, :k])
        keep &= (~np.isnan(rw).any(-1)).any(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            win, place = np.nanmean(rw, 1), np.nanmean(rp, 1)
    return {
        "game_states": a["game_states"][keep], "nodes": a["nodes"][keep],
        "win": canon_rows(win[keep], rot[keep]).astype(np.float32),
        "placement": canon_rows(place[keep], rot[keep]).astype(np.float32),
        "uid": a["uid"][keep], "stage": np.minimum(2, 3 * a["index"][keep] // a["n_actions"][keep]).astype(np.int8),
    }  # fmt: skip


def cmd_anchor(args):
    """The value gate's train split (lr1's anchor) as .npy files to memory-map."""
    from rl18xx.agent.alphazero import value_trainer as vt

    arrays = vt.load_fixed(args.data, "train")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    LOGGER.info(f"{len(arrays['win'])} anchor positions -> {out}")


def load_npy_dir(d, mmap=True) -> dict:
    d = Path(d)
    return {p.stem: np.load(p, mmap_mode="r" if mmap else None) for p in d.glob("*.npy")}


class Average:
    """Exponential moving average of the weights after every step (value_trainer.WeightAverage); ``debias``: the
    average starts empty instead of at the warm start (Adam's correction: weight (1 - d) / (1 - d^t) on step t), so a
    run much shorter than the average's horizon is not mostly the start."""

    def __init__(self, model, decay: float, debias: bool):
        import copy

        self.decay, self.debias, self.t = float(decay), bool(debias), 0
        self.source, self.model = model, copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._avg, self._live = list(self.model.parameters()), list(model.parameters())

    def update(self):
        import torch

        self.t += 1
        w = (1 - self.decay) / (1 - self.decay**self.t) if self.debias else (1 - self.decay)
        with torch.no_grad():
            torch._foreach_lerp_(self._avg, self._live, w)

    def averaged(self):
        import torch

        with torch.no_grad():
            for mine, theirs in zip(self.model.buffers(), self.source.buffers()):
                mine.copy_(theirs)
        return self.model


def arm_data(args) -> tuple:
    """``(train arrays, validation arrays or None)`` of an arm."""
    if args.source == "free":
        d = load_npy_dir(args.data)
        uid = d["uid"]
        game = np.asarray(d["game"])
        val = np.array([val_game(u) for u in uid])
        rows = np.flatnonzero(~val[game])
        n = int(json.loads((Path(args.data) / "meta.json").read_text())["positions"])
        rows = rows[rows < n]
        train = {"game_states": d["game_states"], "nodes": d["nodes"], "win": np.asarray(d["win"]),
                 "placement": np.asarray(d["placement"]), "rows": rows}
        return train, None
    a = load_rollouts(args.data)
    if args.pos_ids:  # only these positions (e.g. the ones two rollout sets share)
        keep = np.isin(a["pos_id"], np.load(args.pos_ids))
        a = {k: v[keep] for k, v in a.items()}
    is_val = np.array([val_game(u) for u in a["uid"]])
    train = rollout_arrays({k: v[~is_val] for k, v in a.items()}, args.target)
    train["rows"] = np.arange(len(train["win"]))
    vals = {t: rollout_arrays({k: v[is_val] for k, v in a.items()}, t) for t in ("rollout", "outcome")}
    return train, vals


def cmd_train(args):
    import torch

    from rl18xx.agent.alphazero import value_trainer as vt
    from rl18xx.agent.alphazero.policy_gradient import _load
    from rl18xx.agent.alphazero.value_refit import write_model_checkpoint

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, vals = arm_data(args)
    rows = train.pop("rows")
    anchor = load_npy_dir(args.anchor) if args.anchor_share > 0 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load(args.init).to(device)
    model.eval()
    opt = torch.optim.AdamW([{"params": list(model.parameters()), "lr": args.lr}], weight_decay=0.0)
    avg = Average(model, args.ema, args.debias) if args.ema > 0 else None
    k_anchor = 0 if anchor is None else int(round(args.batch * args.anchor_share))
    k_stream = args.batch - k_anchor
    steps = args.max_steps or int(math.ceil(args.passes * len(rows) / k_stream))
    snaps = sorted({s for s in args.snap if 0 < s < steps} | {steps})
    LOGGER.info(f"arm {out.name}: {len(rows)} positions, {steps} steps ({steps * k_stream / len(rows):.2f} passes), "
                f"batch {k_stream} + {k_anchor} anchor, snapshots {snaps}")
    (out / "config.json").write_text(json.dumps({**vars(args), "positions": int(len(rows)), "steps": steps,
                                                 "snapshots": snaps}, indent=1))

    def draw(order_state, n_total, k):
        got = []
        while k > 0:
            if order_state["at"] >= len(order_state["order"]):
                order_state["order"], order_state["at"] = rng.permutation(n_total), 0
                order_state["passes"] += 1
            take = order_state["order"][order_state["at"] : order_state["at"] + k]
            order_state["at"] += len(take)
            got.append(take)
            k -= len(take)
        return np.sort(np.concatenate(got))

    stream_state = {"order": np.zeros(0, np.int64), "at": 0, "passes": 0}
    anchor_state = {"order": np.zeros(0, np.int64), "at": 0, "passes": 0}
    keys = ("game_states", "nodes", "win", "placement")

    def validate(net) -> dict:
        if not vals:
            return {}
        return {f"val_{t}": float(vt.per_position_ce(net, v, device).mean()) for t, v in vals.items()}

    log = []
    if vals:
        log.append({"step": 0, **validate(model)})
        LOGGER.info(f"step 0 {log[-1]}")
    t0, ces = time.time(), []
    for step in range(1, steps + 1):
        idx = rows[draw(stream_state, len(rows), k_stream)]
        batch = {k: np.asarray(train[k][idx]) for k in keys}
        if k_anchor:
            ai = draw(anchor_state, len(anchor["win"]), k_anchor)
            batch = {k: np.concatenate([batch[k], np.asarray(anchor[k][ai]).astype(batch[k].dtype)]) for k in keys}
        ce, mse = vt.train_step(model, opt, batch, device, args.score_weight)
        if avg is not None:
            avg.update()
        ces.append(ce)
        if step % 500 == 0:
            LOGGER.info(f"step {step}/{steps} train CE {np.mean(ces[-500:]):.4f} {(time.time() - t0) / step * 1000:.0f} ms/step")
        if step in snaps:
            prov = {"rollout_targets": {"arm": out.name, "step": step, "target": args.target, "source": args.source}}
            write_model_checkpoint(model, args.init, out / "raw" / "learner" / f"{step}.pth", prov)
            rec = {"step": step, "seconds": round(time.time() - t0, 1)}
            rec.update({f"raw_{k}": v for k, v in validate(model).items()})
            if avg is not None:
                write_model_checkpoint(avg.averaged(), args.init, out / "learner" / f"{step}.pth", prov)
                rec.update({f"ema_{k}": v for k, v in validate(avg.averaged()).items()})
            log.append(rec)
            LOGGER.info(f"snapshot {rec}")
    (out / "history.json").write_text(json.dumps({"log": log, "train_ce": [float(np.mean(ces[i : i + 500]))
                                                  for i in range(0, len(ces), 500)],
                                                  "seconds": time.time() - t0}, indent=1))
    LOGGER.info(f"done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Predictions and reports
# ---------------------------------------------------------------------------


def load_value_gate(d, split="test") -> dict:
    manifest = json.loads((Path(d) / "manifest.json").read_text())
    wanted = np.array([g["split"] == split for g in manifest["games"]], dtype=bool)
    parts = {k: [] for k in ("game_states", "nodes", "win", "placement", "stage", "game")}
    for name in manifest["shards"]:
        with np.load(Path(d) / "shards" / name) as z:
            r = np.flatnonzero(wanted[z["game"]])
            if len(r):
                for k in parts:
                    parts[k].append(z[k][r])
    return {k: np.concatenate(v) for k, v in parts.items()}


def eval_sets(exp: Path) -> dict:
    """name -> loader of a test set (game_states, nodes, win, placement, stage, game)."""
    def newrule():
        z = np.load(exp / "testsets/newrule/positions.npz")
        return {k: z[k] for k in z.files}

    def rval():
        a = load_rollouts(exp / "rollouts/main")
        is_val = np.array([val_game(u) for u in a["uid"]])
        sub = {k: v[is_val] for k, v in a.items()}
        r, o = rollout_arrays(sub, "rollout"), rollout_arrays(sub, "outcome")
        _, game = np.unique(r["uid"], return_inverse=True)
        return {"game_states": r["game_states"], "nodes": r["nodes"], "win": o["win"], "placement": o["placement"],
                "win_rollout": r["win"], "stage": r["stage"], "game": game}

    return {"newrule": newrule, "fixed": lambda: load_value_gate(VALUE_GATE), "ladder": lambda: load_value_gate(LADDER),
            "rval": rval}


def predict(model, gs, nodes, device, bs=1024, players: int = NUM_PLAYERS) -> np.ndarray:
    """(n, 2 * players): win probabilities (``players``, canonical) and score head (``players``)."""
    import torch

    out = []
    with torch.no_grad():
        for b in range(0, len(gs), bs):
            g = torch.from_numpy(np.ascontiguousarray(gs[b : b + bs])).to(device).float()
            n = torch.from_numpy(np.ascontiguousarray(nodes[b : b + bs])).to(device).float()
            _, logits, score, _ = model(g, n, value_only=True)
            p = torch.softmax(logits.float()[:, :players], 1)
            out.append(torch.cat([p, score.float()[:, :players]], 1).cpu().numpy())
    return np.concatenate(out)


def cmd_preds(args):
    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value

    exp = Path(args.exp)
    pdir = exp / "preds"
    pdir.mkdir(exist_ok=True)
    loaders = eval_sets(exp)
    sets = args.sets or list(loaders)
    models = dict(m.split("=", 1) for m in args.model)
    todo = {name: [s for s in sets if not (pdir / f"{s}.{name}.npy").exists()] for name in models}
    if args.siblings:
        for name in models:
            if not (pdir / f"sib.{name}.npy").exists():
                todo[name].append("sib")
    data = {}
    device = torch.device("cuda")
    for name, path in models.items():
        if not todo[name]:
            continue
        model = load_policy_value(path).to(device)
        for s in todo[name]:
            if s not in data:
                if s == "sib":
                    sib = load_siblings(Path(args.siblings))
                    data[s] = {"game_states": sib["c_states"], "nodes": sib["c_nodes"],
                               "players": sib["c_win"].shape[-1]}
                else:
                    data[s] = loaders[s]()
                    if not (pdir / f"{s}.meta.npz").exists():
                        np.savez(pdir / f"{s}.meta.npz", **{k: v for k, v in data[s].items()
                                                              if k not in ("game_states", "nodes")})
            players = data[s]["win"].shape[1] if "win" in data[s] else data[s].get("players", NUM_PLAYERS)
            np.save(pdir / f"{s}.{name}.npy",
                    predict(model, data[s]["game_states"], data[s]["nodes"], device, players=players))
        LOGGER.info(f"preds {name}: {todo[name]}")
        del model
        torch.cuda.empty_cache()


def clustered(d, g):
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    G, m = len(cnt), d.mean()
    resid = np.bincount(inv, weights=d) - cnt * m
    return float(m), float(np.sqrt(G / (G - 1) * np.sum(resid**2)) / len(d))


def cmd_report(args):
    exp = Path(args.exp)
    pdir = exp / "preds"
    names = args.names
    ref = args.reference
    report = {}
    lines = []
    for s in args.sets:
        meta = dict(np.load(pdir / f"{s}.meta.npz"))
        win, stage, game = meta["win"], meta["stage"], meta["game"]
        preds = {n: np.load(pdir / f"{s}.{n}.npy") for n in names if (pdir / f"{s}.{n}.npy").exists()}
        if ref not in preds:
            continue
        targets = [("outcome", win)] + ([("rollout", meta["win_rollout"])] if "win_rollout" in meta else [])
        for tname, tgt in targets:
            lines.append(f"\n=== {s} vs {tname}: {len(win)} positions, {len(np.unique(game))} games; "
                         f"CE (paired diff vs {ref} +- SE clustered by game); hit = winner hit")
            lines.append(f"{'model':16s} {'CE all':>24s} {'early':>24s} {'mid':>24s} {'late':>24s} "
                         f"{'hit':>6s} {'e':>6s} {'m':>6s} {'l':>6s}")
            k = tgt.shape[1]  # the seats
            ce = {n: -(tgt * np.log(np.clip(p[:, :k], 1e-9, 1))).sum(1) for n, p in preds.items()}
            for n, p in preds.items():
                cells, rec = [], {}
                for st, sm in (("all", np.ones(len(win), bool)), ("early", stage == 0), ("mid", stage == 1),
                               ("late", stage == 2)):
                    d, se = clustered(ce[n][sm] - ce[ref][sm], game[sm])
                    rec[st] = {"ce": float(ce[n][sm].mean()), "diff": d, "se": se}
                    cells.append(f"{ce[n][sm].mean():.4f}({d:+.4f}+-{se:.4f})")
                hit = (np.take_along_axis(win, p[:, :k].argmax(1)[:, None], 1)[:, 0] > 0).astype(float)
                hits = [hit[sm].mean() for sm in (np.ones(len(win), bool), stage == 0, stage == 1, stage == 2)]
                rec["hit"] = [float(h) for h in hits]
                report.setdefault(f"{s}/{tname}", {})[n] = rec
                lines.append(f"{n:16s} " + " ".join(f"{c:>24s}" for c in cells) + " " +
                             " ".join(f"{h:6.3f}" for h in hits))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))


def cmd_noise(args):
    """Target noise of a rollout set, per seat (canonical frame), overall and by stage: the variance of a single
    game outcome across positions, a single rollout's within-position variance, the K-mean's noise variance, the
    rollout value's signal variance, and the game outcome's squared distance from the rollout value (its noise +
    the search-vs-rollout gap), each corrected for the K-mean's own noise."""
    a = load_rollouts(args.data)
    keep = (~np.isnan(a["r_win"]).any(-1)).all(1)
    a = {k: v[keep] for k, v in a.items()}
    rot = a["rotation"]
    players = a["win"].shape[1]
    idx = (np.arange(players)[None, :] + rot[:, None]) % players
    z = np.take_along_axis(a["win"], idx, 1).astype(np.float64)
    rw = np.take_along_axis(a["r_win"], idx[:, None, :], 2).astype(np.float64)
    K = rw.shape[1]
    stage = np.minimum(2, 3 * a["index"] // a["n_actions"])
    term = a["r_term"]
    out, lines = {}, []
    lines.append(f"{len(z)} positions, K={K}; rollout length {a['r_len'].mean():.0f} decisions; terminations "
                 f"{ {int(t): int((term == t).sum()) for t in np.unique(term)} }")
    lines.append(f"{'subset':8s} {'n':>6s} {'Var(z)':>8s} {'s2_w':>8s} {'s2_w/K':>8s} {'signal':>8s} "
                 f"{'E(z-V)^2':>9s} {'noise ratio z/Kmean':>20s} {'SNR 1-roll':>10s} {'SNR K-mean':>10s} {'SNR outcome':>11s}")
    for name, sm in (("all", np.ones(len(z), bool)), ("early", stage == 0), ("mid", stage == 1), ("late", stage == 2)):
        m = rw[sm].mean(1)
        s2w = rw[sm].var(1, ddof=1).mean()
        signal = m.var(0).mean() - s2w / K
        gap = ((z[sm] - m) ** 2).mean() - (rw[sm].var(1, ddof=1) / K).mean()  # E[(z - V_roll)^2] per seat
        row = {"n": int(sm.sum()), "var_outcome": float(z[sm].var(0).mean()), "s2_within": float(s2w),
               "s2_kmean": float(s2w / K), "signal": float(signal), "outcome_gap": float(gap),
               "noise_ratio_outcome_vs_kmean": float(gap / (s2w / K))}
        out[name] = row
        lines.append(f"{name:8s} {row['n']:6d} {row['var_outcome']:8.4f} {s2w:8.4f} {s2w / K:8.4f} {signal:8.4f} "
                     f"{gap:9.4f} {gap / (s2w / K):20.1f} {signal / s2w:10.3f} {signal / (s2w / K):10.3f} "
                     f"{signal / gap:11.3f}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1))


def load_siblings(d: Path) -> dict:
    files = [f for f in sorted((d / "tasks").glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts if k in p]) for k in parts[0]}


def cmd_sib_report(args):
    """Within-root Spearman of each value source's children ranking against the rollout truth (the parent
    mover's mean win share over K common-random-number rollouts), mean over roots (+- SE); top-1 agreement;
    the truth's split-half reliability."""
    from scipy.stats import spearmanr

    exp = Path(args.exp)
    sib = load_siblings(Path(args.siblings))
    mover = dict(zip(sib["p_pos_id"].tolist(), sib["p_rotation"].tolist()))
    stage_of = {int(p): min(2, 3 * int(i) // max(int(n), 1))
                for p, i, n in zip(sib["p_pos_id"], sib["p_index"], sib["p_n_actions"])}
    pos = sib["c_pos_id"]
    m = np.array([mover[int(p)] for p in pos])
    K = sib["c_win"].shape[1]
    share = sib["c_win"][np.arange(len(pos)), :, m]  # (children, K): the parent mover's win share per rollout
    truth = np.nanmean(share, 1)
    half_a, half_b = np.nanmean(share[:, : K // 2], 1), np.nanmean(share[:, K // 2 :], 1)
    rot = sib["c_rotation"]
    terminal = sib["c_terminal"].astype(bool)
    sources = {}
    players = sib["c_win"].shape[-1]
    for n in args.names:
        p = np.load(exp / "preds" / f"sib.{n}.npy")[:, :players]
        v = p[np.arange(len(pos)), (m - rot) % players].astype(np.float64)
        v[terminal] = truth[terminal]
        sources[n] = v
    sources["prior"] = sib["c_prior"].astype(np.float64)
    roots = np.unique(pos)
    rows = {n: [] for n in sources}
    rel, top1 = [], {n: [] for n in sources}
    stages = []
    for r in roots:
        c = np.flatnonzero(pos == r)
        if len(c) < 3 or np.std(truth[c]) < 1e-9:
            continue
        stages.append(stage_of[int(r)])
        if np.std(half_a[c]) > 1e-9 and np.std(half_b[c]) > 1e-9:
            rel.append(spearmanr(half_a[c], half_b[c]).correlation)
        best = truth[c].max()
        for n, v in sources.items():
            rows[n].append(spearmanr(v[c], truth[c]).correlation if np.std(v[c]) > 1e-12 else np.nan)
            top1[n].append(float(truth[c][np.argmax(v[c])] >= best - 1e-9))
    stages = np.array(stages)
    lines = [f"sibling truth set: {len(stages)} roots ({np.bincount(stages, minlength=3).tolist()} early/mid/late), "
             f"{len(pos)} children, K={K}; truth split-half Spearman {np.nanmean(rel):.3f} "
             f"(Spearman-Brown full-K reliability {2 * np.nanmean(rel) / (1 + np.nanmean(rel)):.3f})",
             f"{'source':16s} {'spearman':>16s} {'early':>8s} {'mid':>8s} {'late':>8s} {'top1':>6s}"]
    report = {}
    for n in sources:
        s = np.array(rows[n], dtype=np.float64)
        ok = ~np.isnan(s)
        se = np.nanstd(s, ddof=1) / np.sqrt(ok.sum())
        by = [np.nanmean(s[stages == k]) for k in range(3)]
        report[n] = {"spearman": float(np.nanmean(s)), "se": float(se), "by_stage": [float(x) for x in by],
                     "top1": float(np.mean(top1[n]))}
        lines.append(f"{n:16s} {np.nanmean(s):8.3f}+-{se:.3f} " + " ".join(f"{x:8.3f}" for x in by) +
                     f" {np.mean(top1[n]):6.3f}")
    # paired differences vs the first source
    base = np.array(rows[args.names[0]], dtype=np.float64)
    for n in sources:
        d = np.array(rows[n], dtype=np.float64) - base
        ok = ~np.isnan(d)
        lines.append(f"  {n} - {args.names[0]}: {np.nanmean(d):+.4f} +- {np.nanstd(d, ddof=1) / np.sqrt(ok.sum()):.4f}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rollouts")
    r.add_argument("--data", default=DEFAULT_DATA)
    r.add_argument("--out", required=True)
    r.add_argument("--games", type=int, required=True, help="saved games sampled (one position each)")
    r.add_argument("--skip", type=int, default=0, help="skip the first N of the seeded game order (disjoint sets)")
    r.add_argument("--games-per-shard", type=int, default=None, help="(testing) assume this many games per shard")
    r.add_argument("--K", type=int, default=8)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--policy", default=DEFAULT_POLICY)
    r.add_argument("--workers", type=int, default=24)
    r.add_argument("--replicas", type=int, default=1)
    r.add_argument("--positions-per-task", type=int, default=8)
    r.add_argument("--max-decisions", type=int, default=1000)
    r.add_argument("--seed", type=int, default=20261101)
    r.add_argument("--hours", type=float, default=0.0, help="stop submitting after this long (0: no limit)")

    g = sub.add_parser("games")
    g.add_argument("--out", required=True)
    g.add_argument("--policy", required=True)
    g.add_argument("--decisions", type=int, required=True, help="budget: policy decisions over all games")
    g.add_argument("--games", type=int, default=2_000_000, help="game numbers available (the budget ends first)")
    g.add_argument("--games-per-task", type=int, default=32)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--explore-decisions", type=int, default=120, help="from the real game start, at T explore")
    g.add_argument("--explore-temperature", type=float, default=1.0)
    g.add_argument("--num-players", type=int, default=NUM_PLAYERS, help="players per game (2-6)")
    g.add_argument("--start-positions", default=None,
                   help="default: human_games/start_positions_1830_<N>p_v2.jsonl for --num-players N")
    g.add_argument("--random-start-fraction", type=float, default=0.2)
    g.add_argument("--midgame-positions", default=None,
                   help="default: human_games/midgame_positions_1830_<N>p_v2.jsonl for --num-players N")
    g.add_argument("--midgame-fraction", type=float, default=0.25)
    g.add_argument("--auction-fraction", type=float, default=0.35)
    g.add_argument("--max-decisions", type=int, default=1000)
    g.add_argument("--workers", type=int, default=24)
    g.add_argument("--replicas", type=int, default=1)
    g.add_argument("--seed", type=int, default=20261103)

    s = sub.add_parser("sib-roots")
    s.add_argument("--collection", action="append", required=True)
    s.add_argument("--positions", type=int, default=300)
    s.add_argument("--seed", type=int, default=20261102)
    s.add_argument("--out", required=True)

    s = sub.add_parser("siblings")
    s.add_argument("--roots", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--K", type=int, default=16)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--policy", default=DEFAULT_POLICY)
    s.add_argument("--workers", type=int, default=24)
    s.add_argument("--replicas", type=int, default=1)
    s.add_argument("--positions-per-task", type=int, default=1)
    s.add_argument("--max-children", type=int, default=12)
    s.add_argument("--max-decisions", type=int, default=1000)
    s.add_argument("--seed", type=int, default=20261103)
    s.add_argument("--hours", type=float, default=0.0)

    t = sub.add_parser("testset")
    t.add_argument("--collection", action="append", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--per-game", type=int, default=16)
    t.add_argument("--seed", type=int, default=20261101)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--require-summary", action="store_true")

    f = sub.add_parser("free")
    f.add_argument("--data", default=DEFAULT_DATA)
    f.add_argument("--out", required=True)
    f.add_argument("--per-game", type=int, default=16)
    f.add_argument("--pass-id", type=int, default=1)
    f.add_argument("--workers", type=int, default=8)

    a = sub.add_parser("anchor")
    a.add_argument("--data", default=str(VALUE_GATE))
    a.add_argument("--out", required=True)

    t = sub.add_parser("train")
    t.add_argument("--source", choices=("rollouts", "free"), required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--target", default="outcome", help="outcome | rollout | rollout<k> (rollouts source)")
    t.add_argument("--pos-ids", default=None, help="(rollouts source) a .npy of the positions to keep")
    t.add_argument("--out", required=True)
    t.add_argument("--init", default=V4)
    t.add_argument("--anchor", default=None)
    t.add_argument("--anchor-share", type=float, default=0.5)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--score-weight", type=float, default=1.0)
    t.add_argument("--ema", type=float, default=0.9998)
    t.add_argument("--debias", action="store_true")
    t.add_argument("--passes", type=float, default=1.5)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--snap", type=int, nargs="*", default=[])
    t.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("preds")
    t.add_argument("--exp", required=True)
    t.add_argument("--model", action="append", required=True, help="name=checkpoint spec")
    t.add_argument("--sets", nargs="*", default=None)
    t.add_argument("--siblings", default=None, help="sibling run dir: also predict its children")

    t = sub.add_parser("report")
    t.add_argument("--exp", required=True)
    t.add_argument("--names", nargs="+", required=True)
    t.add_argument("--reference", default="v4")
    t.add_argument("--sets", nargs="+", default=["newrule", "fixed", "ladder", "rval"])
    t.add_argument("--out", default=None)

    t = sub.add_parser("noise")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default=None)

    t = sub.add_parser("sib-report")
    t.add_argument("--exp", required=True)
    t.add_argument("--siblings", required=True)
    t.add_argument("--names", nargs="+", required=True)
    t.add_argument("--out", default=None)

    args = p.parse_args()
    {"rollouts": cmd_rollouts, "games": cmd_games, "sib-roots": cmd_sib_roots, "siblings": cmd_siblings, "testset": cmd_testset,
     "free": cmd_free, "anchor": cmd_anchor, "train": cmd_train, "preds": cmd_preds, "report": cmd_report,
     "sib-report": cmd_sib_report, "noise": cmd_noise}[args.cmd](args)


if __name__ == "__main__":
    main()
