"""Are a search's departures from the policy better moves?

For each ``search_decision_diag.py`` position where a variant played something
other than the policy's top move, both moves are played out ``--rollouts`` times
by policy-only play (every seat the ``--policy`` checkpoint at ``--temperature``,
prices from its price head, as policy-gradient games are played) and the
mover's win share compared. Rollouts of all positions advance in lockstep, so
the network sees large batches. CPU or GPU, in-process model.

    uv run python scripts/search_override_rollouts.py --diag logs/search_diag/run1.jsonl \\
        --policy model_checkpoints_pg/<run>/learner/<n>.pth --variant fix800 --variant b800 \\
        --out logs/search_diag/run1_rollouts.jsonl
    uv run python scripts/search_override_rollouts.py --diag logs/search_diag/run1.jsonl \\
        --summary logs/search_diag/run1_rollouts.jsonl

The rollout policy is a stand-in for perfect play: a move it continues well from
counts as good. Records from diagnostics written before they carried a ``pos``
are matched to their positions by the policy's prior (``--policy`` must then be
the diagnostic's policy).
"""

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_SETTINGS: dict = {}
_MODEL = None


def _init(settings: dict):
    global _MODEL
    import torch

    torch.set_num_threads(settings["threads"])
    from rl18xx.agent.alphazero.composite_model import load_policy_value

    _SETTINGS.update(settings)
    _MODEL = load_policy_value(settings["policy"])


def _forward(games_legal: list):
    """``[(probs over legal, price-head row)]`` and the encoded states."""
    import torch

    from rl18xx.agent.alphazero.mcts import _rust_encode

    encoded = [_rust_encode(game) for game, _ in games_legal]
    with torch.no_grad():
        probs, _, _ = _MODEL.run_many_encoded(encoded, legal_indices=[legal for _, legal in games_legal])
    price = getattr(_MODEL, "last_price_components", None)
    rows = price["price_logits"].numpy() if price is not None else None
    out = []
    for i, (_, legal) in enumerate(games_legal):
        p = np.asarray(probs[i])[legal].astype(np.float64)
        p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
        out.append((p, rows[i] if rows is not None else None))
    return out, encoded


def fingerprint(actions: list) -> tuple:
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game

    game = apply_actions(new_game(4), actions)
    legal = [int(i) for i in game._game.factored_legal_indices()]
    [(p, _)], _ = _forward([(game, legal)])
    return game.round.__class__.__name__, len(legal), legal[int(p.argmax())], float(p.max())


def play_out(job) -> dict:
    """``rollouts`` games from (position, move): the mover's win shares."""
    from rl18xx.agent.alphazero.policy_gradient import outcome
    from rl18xx.agent.alphazero.policy_selfplay import _Game, _advance, _sample_price
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game

    pos_id, actions, move, seed = job
    rng = np.random.default_rng(seed)
    base = apply_actions(new_game(4), actions)
    [(_, price_row)], encoded = _forward([(base, [int(i) for i in base._game.factored_legal_indices()])])
    mover = int(encoded[0][6])
    games = []
    for r in range(_SETTINGS["rollouts"]):
        game = apply_actions(new_game(4), actions)
        game._game.apply_action_index(int(move), _sample_price(game, int(move), price_row, rng, 0.0))
        games.append(_Game(game=game, uid=f"{pos_id}:{move}:{r}", start=""))
    settings = {"max_decisions": _SETTINGS["max_decisions"], "price_eps": 0.0}
    active = games
    while active:
        batch = [(g, legal) for g in active if (legal := _advance(g, settings, rng)) is not None]
        active = [g for g, _ in batch]
        if not batch:
            break
        outs, _ = _forward([(g.game, legal) for g, legal in batch])
        for (g, legal), (p, row) in zip(batch, outs):
            p = np.power(np.clip(p, 1e-12, None), 1.0 / _SETTINGS["temperature"])
            choice = legal[int(rng.choice(len(legal), p=p / p.sum()))]
            g.game._game.apply_action_index(choice, _sample_price(g.game, choice, row, rng, 0.0))
            g.decisions += 1
    return {
        "pos": pos_id,
        "move": int(move),
        "mover": mover,
        "wins": [float(outcome(g.game)[0][mover]) for g in games],
        "terminations": [g.termination for g in games],
    }


def _positions_of(records: list, positions: list, pool) -> list:
    """Each record's position index (fingerprinted when the record has no ``pos``)."""
    if all("pos" in r for r in records):
        return [r["pos"] for r in records]
    index = {}
    for i, (rnd, n, top, pmax) in enumerate(pool.map(fingerprint, positions)):
        index.setdefault((rnd, n, top), []).append((pmax, i))
    out = []
    for r in records:
        candidates = index.get((r["round"], r["n_legal"], r["prior_top"]), [])
        best = min(candidates, key=lambda c: abs(c[0] - r["prior_max"]), default=None)
        out.append(best[1] if best is not None and abs(best[0] - r["prior_max"]) < 1e-4 else None)
    return out


def summarize(records: list, pos_ids: list, rollouts_path: Path, variants=None) -> None:
    wins = {}
    for line in open(rollouts_path):
        r = json.loads(line)
        wins[(r["pos"], r["move"])] = np.asarray(r["wins"])
    variants = variants or [k for k in records[0] if k + "_share" in records[0]]
    print("variant    departures  played out  win(search move) - win(policy move)  better/worse")
    for v in variants:
        diffs, departures = [], 0
        for r, pos in zip(records, pos_ids):
            if r[v] == r["prior_top"]:
                continue
            departures += 1
            if (pos, r[v]) in wins and (pos, r["prior_top"]) in wins:
                diffs.append(wins[(pos, r[v])].mean() - wins[(pos, r["prior_top"])].mean())
        if len(diffs) > 1:
            se = np.std(diffs, ddof=1) / math.sqrt(len(diffs))
            better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
            print(f"{v:10s} {departures:10d} {len(diffs):11d}   {np.mean(diffs):+.3f} +- {se:.3f}{'':19s}{better}/{worse}")
        else:
            print(f"{v:10s} {departures:10d} {len(diffs):11d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diag", type=Path, required=True, help="search_decision_diag.py records")
    parser.add_argument("--positions", type=Path, help="Its positions (default <diag>.positions.json)")
    parser.add_argument("--policy", help="Rollout policy checkpoint (.pth)")
    parser.add_argument("--variant", action="append", default=[], help="Variants whose departures to play out")
    parser.add_argument("--rollouts", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="Torch threads per worker")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path, help="Summarize these rollouts against --diag and exit")
    args = parser.parse_args()
    positions = json.loads((args.positions or Path(f"{args.diag}.positions.json")).read_text())
    records = [json.loads(line) for line in open(args.diag)]
    if args.summary and all("pos" in r for r in records):
        summarize(records, [r["pos"] for r in records], args.summary, args.variant or None)
        return
    if not args.policy or not (args.summary or (args.variant and args.out)):
        parser.error("--policy, --variant and --out are required (--policy also to summarize records without a pos)")
    settings = {
        "policy": str(Path(args.policy).resolve()),
        "threads": args.threads,
        "rollouts": args.rollouts,
        "temperature": args.temperature,
        "max_decisions": args.max_decisions,
    }
    started = time.time()
    with Pool(args.workers, initializer=_init, initargs=(settings,)) as pool:
        pos_ids = _positions_of(records, positions, pool)
        if args.summary:
            summarize(records, pos_ids, args.summary, args.variant or None)
            return
        tasks = set()
        for r, pos in zip(records, pos_ids):
            for v in args.variant:
                if pos is not None and r[v] != r["prior_top"]:
                    tasks.update({(pos, r["prior_top"]), (pos, r[v])})
        print(f"{sum(p is not None for p in pos_ids)}/{len(records)} records placed; {len(tasks)} (position, move) pairs", flush=True)
        jobs = [(pos, positions[pos], move, 1000 * pos + move % 1000) for pos, move in sorted(tasks)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as out:
            for result in pool.imap_unordered(play_out, jobs):
                out.write(json.dumps(result) + "\n")
                out.flush()
    print(f"done in {time.time() - started:.0f}s", flush=True)
    summarize(records, pos_ids, args.out, args.variant)


if __name__ == "__main__":
    main()
