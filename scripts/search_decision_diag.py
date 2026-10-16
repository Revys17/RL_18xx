"""How often does a search setting play something other than the policy's top move?

Positions come from games the player itself plays (16 readouts, moves sampled
from visit counts, from self-play's start positions): every ``--every``-th
decision. At each position every variant searches from scratch and plays its
argmax move; a record holds the policy's top move and prior, each variant's
move, its top move's share of the root visits and the tree size (distinct
evaluated positions + 1). CPU or GPU, in-process models (no inference server).

    uv run python scripts/search_decision_diag.py --player <policy>+<value> --games 4 \\
        --variant b64=@64 --variant fix800=@800:dedupe,par=8,ptemp=0.5 --out logs/search_diag/run1.jsonl
    uv run python scripts/search_decision_diag.py --summary logs/search_diag/run1.jsonl

A variant is ``name=<suffix>``, the suffix in ``eval_head_to_head.py``'s player
syntax (``@<readouts>``, ``/<c_puct_init>``, ``:<options>``). Positions are saved
next to ``--out`` (``<out>.positions.json``) and reused by a rerun. Search moves
these change are what ``scripts/search_override_rollouts.py`` plays out.
(2026-10-16, gen4: with the battery's @64 settings the search left the policy's
top move at 1-2% of positions; @800 at ~3-8%.)
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_SETTINGS: dict = {}
_MODEL = None


def _init(settings: dict):
    global _MODEL
    import torch

    torch.set_num_threads(settings["threads"])
    from rl18xx.agent.alphazero.composite_model import load_policy_value

    _SETTINGS.update(settings)
    _MODEL = load_policy_value(settings["model"])


def _player(overrides: dict):
    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer

    overrides = {"min_readouts": overrides["num_readouts"], **overrides}
    config = SelfPlayConfig(
        network=_MODEL,
        use_inference_server=False,
        softpick_move_cutoff=2,
        dirichlet_noise_weight=0.0,
        enable_resign=False,
        noresign_holdout_rate=0.0,
        player_count_distribution={4: 1.0},
        auction_unlock=True,
        **overrides,
    )
    return RustMCTSPlayer(config)


def _searched(game, overrides: dict):
    from eval_head_to_head import _expand_root

    player = _player(overrides)
    player.initialize_game(game.pickle_clone())
    _expand_root(player)
    return player


def play_positions(seed: int) -> list:
    """Every ``every``-th decision of one game the player plays at 16 readouts."""
    import torch

    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game, sample_start_position

    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    start = sample_start_position(4, _SETTINGS["start_positions"], 0.2, rng=rng)
    game = apply_actions(new_game(4), start.actions)
    out, decisions = [], 0
    with torch.no_grad():
        while not game.finished and decisions < _SETTINGS["max_decisions"]:
            player = _searched(game, {"num_readouts": 16})
            player.decisions = 0  # below the softpick cutoff: sample from the visits
            if len(player._rust_player.legal_action_indices_at_root()) > 1:
                decisions += 1
                if decisions % _SETTINGS["every"] == 0:
                    out.append(list(game.raw_actions))
            player.play_move(player.suggest_move())
            game = player.get_game_state()
    return out


def analyse(job) -> dict:
    import torch

    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game

    pos_id, actions = job
    game = apply_actions(new_game(4), actions)
    record = {"pos": pos_id, "round": game.round.__class__.__name__}
    with torch.no_grad():
        probs, _, _ = _MODEL.run_encoded(_rust_encode(game))
        for name, overrides in _SETTINGS["variants"].items():
            player = _searched(game, overrides)
            player.decisions = 10**9  # past the softpick cutoff: argmax
            tree = player._rust_player
            legal = list(tree.legal_action_indices_at_root())
            if "prior_top" not in record:
                prior = np.asarray(probs)[legal]
                prior = prior / prior.sum()
                record.update(prior_top=int(legal[int(prior.argmax())]), prior_max=float(prior.max()), n_legal=len(legal))
            started = time.time()
            move = player.suggest_move()
            visits = np.asarray(tree.child_n_at_root())[legal]
            record[name] = int(move)
            record[name + "_share"] = float(visits.max() / visits.sum())
            record[name + "_arena"] = int(tree.arena_size())
            record[name + "_s"] = round(time.time() - started, 2)
    return record


def summarize(path: Path, reference=None) -> None:
    records = [json.loads(line) for line in open(path)]
    variants = [k for k in records[0] if k + "_share" in records[0]]
    reference = reference or variants[-1]
    print(f"{len(records)} positions, median {sorted(r['n_legal'] for r in records)[len(records) // 2]} legal moves")
    groups = [("all", records)] + [
        (name, [r for r in records if r["round"] == name]) for name in sorted({r["round"] for r in records})
    ]
    for label, group in groups:
        print(f"\n{label} ({len(group)})  left policy top  agrees w/ {reference}  top visit share  tree  seconds")
        for v in variants:
            mean = lambda f: sum(f(r) for r in group) / len(group)  # noqa: E731
            print(
                f"  {v:10s} {mean(lambda r: r[v] != r['prior_top']):14.3f} {mean(lambda r: r[v] == r[reference]):14.3f}"
                f" {mean(lambda r: r[v + '_share']):14.2f} {mean(lambda r: r[v + '_arena']):7.0f} {mean(lambda r: r[v + '_s']):7.1f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player", help="Checkpoint spec (eval_head_to_head.py syntax, no search suffix)")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=SUFFIX")
    parser.add_argument("--games", type=int, default=4, help="Games to draw positions from")
    parser.add_argument("--every", type=int, default=6, help="Keep every n-th decision")
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--start-positions", default="human_games/start_positions_1830_4p.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="Torch threads per worker")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path, help="Print the summary of a finished run and exit")
    parser.add_argument("--reference", help="Variant the agreement column compares with (default: the last)")
    args = parser.parse_args()
    if args.summary:
        summarize(args.summary, args.reference)
        return
    if not (args.player and args.variant and args.out):
        parser.error("--player, --variant and --out are required")

    from eval_head_to_head import parse_player

    model, variants = None, {}
    for spec in args.variant:
        name, _, suffix = spec.partition("=")
        path, overrides = parse_player(args.player + suffix, default_readouts=64)
        model = model or path
        variants[name] = overrides
    settings = {
        "model": model,
        "variants": variants,
        "threads": args.threads,
        "every": args.every,
        "max_decisions": args.max_decisions,
        "start_positions": args.start_positions,
    }
    positions_path = Path(f"{args.out}.positions.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=_init, initargs=(settings,)) as pool:
        if positions_path.exists():
            positions = json.loads(positions_path.read_text())
        else:
            positions = [p for game in pool.map(play_positions, range(args.games)) for p in game]
            positions_path.write_text(json.dumps(positions))
        print(f"{len(positions)} positions", flush=True)
        with open(args.out, "w") as out:
            for record in pool.imap_unordered(analyse, list(enumerate(positions))):
                out.write(json.dumps(record) + "\n")
                out.flush()
    summarize(args.out, args.reference)


if __name__ == "__main__":
    main()
