"""Search learning's saved games (``main.py search-learning --save-data``, ``search_data``):
what they weigh and what keeping them costs the generation.

``size``: one in-process generator plays a run's search settings (``--run-config``, its
``config.json``) with ``--model`` (a server spec: ``<learner>`` or ``<learner>+<critic>``),
keeps every finished game in ``--out`` and reports the bytes per game (and per part:
action log, searches, rows), the worker's packing time per game, the trainer's add time per
game, the shards' compression time, and what the same rows weigh as pickled ``SearchRow`` s.
The GPU stays hidden unless ``--gpu``.

``ab``: the generation's decisions/s with saving off and on, alternated for ``--rounds``,
on a stand-in network (``search_test_utils.HashNet``, a hash of the position) so the
generation itself, not a forward pass, is the cost; both generators are seeded alike, so they
play nearly the same games.

    uv run python scripts/bench_search_data.py size --run-config <run>/config.json --model <run>/learner/<n>.pth \\
        --data-root /home/revys/workspace/RL_18xx --seconds 3600 --out logs/search_data_bench/size
    uv run python scripts/bench_search_data.py ab --rounds 3 --seconds 120
"""

import argparse
import dataclasses
import json
import lzma
import os
import pickle
import random
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "agent" / "alphazero"))

PARTS = {
    "searches": ("search_decision", "search_action", "search_flags", "search_version", "search_value"),
    "rows": ("row_rotation", "row_legal_count", "row_flags", "row_weight", "row_price_slot", "row_price_lo",
             "row_price_hi", "legal", "target", "price_cells"),
}


def run_config(path, data_root):
    from rl18xx.agent.alphazero import search_learning as sl

    raw = json.loads(Path(path).read_text())
    fields = {f.name for f in dataclasses.fields(sl.SearchLearningConfig)}
    cfg = sl.SearchLearningConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items() if k in fields})
    for key in ("start_positions", "midgame_positions"):
        value = getattr(cfg, key)
        if value and not Path(value).is_absolute():
            setattr(cfg, key, str(Path(data_root) / value))
    if cfg.midgame_fraction <= 0:
        cfg.midgame_positions = None
    return cfg


def part_bytes(games: list, preset: int) -> dict:
    """xz bytes per game of each part on its own (a shard compresses them together)."""
    from rl18xx.agent.alphazero.search_data import column_bytes

    out = {"actions": len(lzma.compress(b"".join(g["actions"] for g in games), preset=preset))}
    for part, names in PARTS.items():
        out[part] = len(lzma.compress(b"".join(column_bytes(name, games)[1] for name in names), preset=preset))
    return {k: v / len(games) for k, v in out.items()}


def cmd_size(args):
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    from rl18xx.agent.alphazero import search_data as sd
    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.lockstep import ModelEvaluator
    from rl18xx.agent.alphazero.policy_gradient import load_server_model

    torch.set_num_threads(args.torch_threads)
    cfg = run_config(args.run_config, args.data_root)
    cfg.games_per_worker, cfg.save_data = args.games_per_worker, args.out
    if cfg.learner_temperature is None:
        cfg.learner_temperature = sl.run_temperature(args.model.split("+")[0])
    if cfg.prior_temperature is None:
        cfg.prior_temperature = cfg.learner_temperature
    if args.preset is None:
        args.preset = sd.PRESET
    net = load_server_model(args.model)
    net.eval()
    pack_seconds = []
    pack = sd.pack_game

    def timed_pack(*a, **k):
        t = time.perf_counter()
        out = pack(*a, **k)
        pack_seconds.append(time.perf_counter() - t)
        return out

    sd.pack_game = timed_pack  # search_learning._finish imports it at each call
    gen = sl.Generator(sl.worker_settings(cfg, 0), {"learner": ModelEvaluator(net)})
    writer = sd.SearchDataWriter(args.out, dataclasses.asdict(cfg), args.shard_games, args.preset)
    games, add_seconds, row_bytes, rows, decisions, started = [], [], [], 0, 0, time.time()
    while time.time() - started < args.seconds:
        out = gen.run(min(args.task_seconds, max(1.0, args.seconds - (time.time() - started))))
        stats = out["stats"]
        decisions += stats.get("searched_decisions", 0) + stats.get("fast_search_decisions", 0)
        decisions += stats.get("fast_decisions", 0)
        for record in out["games"]:
            t = time.perf_counter()
            writer.add(record["saved"])
            add_seconds.append(time.perf_counter() - t)
            games.append(record["saved"])
            rows += len(record["rows"])
            row_bytes.append(len(pickle.dumps(record["rows"], protocol=pickle.HIGHEST_PROTOCOL)))
        print(f"{time.time() - started:.0f}s: {len(games)} games, {decisions / (time.time() - started):.1f} "
              f"decisions/s", flush=True)
    t = time.perf_counter()
    writer.close()
    close_seconds = time.perf_counter() - t
    if not games:
        raise SystemExit("no game finished: raise --seconds")
    shard_bytes = sum(p.stat().st_size for p in sd.shard_paths(args.out))
    t = time.perf_counter()
    raw = sd.encode_shard(games, writer.config)
    compressed = lzma.compress(raw, preset=args.preset)
    compress_seconds = time.perf_counter() - t
    t = time.perf_counter()
    saved = list(sd.iter_games(args.out))
    rebuilt = sum(len(sd.rebuild_rows(g)) for g in saved)
    rebuild_seconds = time.perf_counter() - t
    meta = [g["meta"] for g in games]
    summary = {
        "games": len(games),
        "shards": len(sd.shard_paths(args.out)),
        "bytes_per_game": shard_bytes / len(games),
        "bytes_per_game_one_shard": len(compressed) / len(games),
        "part_bytes_per_game": part_bytes(games, args.preset),
        "uncompressed_bytes_per_game": len(raw) / len(games),
        "rows_per_game": rows / len(games),
        "searches_per_game": sum(len(g["columns"]["search_decision"]) for g in games) / len(games),
        "legal_per_row": sum(len(g["columns"]["legal"]) for g in games) / max(rows, 1),
        "actions_per_game": float(np.mean([m["actions"] for m in meta])),
        "decisions_per_game": float(np.mean([m["decisions"] for m in meta])),
        "starts": {k: sum(m["start_kind"] == k for m in meta) for k in sorted({m["start_kind"] for m in meta})},
        "pickled_rows_bytes_per_game": float(np.mean(row_bytes)),
        "pack_ms_per_game": 1000 * float(np.mean(pack_seconds)),
        "add_ms_per_game": 1000 * float(np.mean(add_seconds)),
        "close_seconds": close_seconds,
        "compress_ms_per_game": 1000 * compress_seconds / len(games),
        "rebuild_ms_per_game": 1000 * rebuild_seconds / len(saved),
        "rows_rebuilt": rebuilt,
        "decisions_per_second": decisions / (time.time() - started),
        "model": args.model,
        "run_config": args.run_config,
    }
    print(json.dumps(summary, indent=1))
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))


def cmd_ab(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np

    from rl18xx.agent.alphazero import search_data as sd
    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.lockstep import ModelEvaluator
    from search_test_utils import HashNet

    if args.run_config:
        cfg = run_config(args.run_config, args.data_root)
    else:
        # azl's search settings at a tenth of the readouts (the stand-in network is cheap).
        cfg = sl.SearchLearningConfig(
            policy_checkpoint="", value_checkpoint="", learner_temperature=1.0, prior_temperature=1.0,
            target="cq", n_full=20, n_fast=5, p_full=0.25, c_puct_init=0.05, reuse_tree="total",
            full_move_temperature=1.0, explore_decisions=120, value_lambda=0.97, critic_positions_per_game=0,
            start_positions=str(Path(args.data_root) / "human_games/start_positions_1830_4p_v2.jsonl"),
        )
    cfg.games_per_worker = args.games_per_worker
    directory = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="search_data_ab_"))
    writer = sd.SearchDataWriter(directory, dataclasses.asdict(cfg), args.shard_games)
    gens = {}
    for save in (False, True):
        cfg.save_data = str(directory) if save else None
        gens[save] = sl.Generator(sl.worker_settings(cfg, 0), {"learner": ModelEvaluator(HashNet())})
        # The same seeds: both play nearly the same games (the network is a hash of the position).
        gens[save].rng, gens[save].py_rng = np.random.default_rng(args.seed), random.Random(args.seed)
        gens[save].run(args.warmup)  # games under way, as in a long run
    rates, games = {False: [], True: []}, {False: 0, True: 0}
    for r in range(args.rounds):
        for save in (False, True) if r % 2 == 0 else (True, False):
            out = gens[save].run(args.seconds)
            stats = out["stats"]
            decisions = sum(stats.get(k, 0) for k in ("searched_decisions", "fast_search_decisions", "fast_decisions"))
            for record in out["games"]:
                if save:
                    writer.add(record["saved"])  # the trainer's part too (another process in a run)
            games[save] += len(out["games"])
            rates[save].append(decisions / stats["seconds"])
            print(f"round {r} save={save}: {rates[save][-1]:.1f} decisions/s, {len(out['games'])} games", flush=True)
    writer.close()
    summary = {
        "decisions_per_second_off": rates[False],
        "decisions_per_second_on": rates[True],
        "mean_off": float(np.mean(rates[False])),
        "mean_on": float(np.mean(rates[True])),
        "ratio_on_off": float(np.mean(rates[True]) / np.mean(rates[False])),
        "games_off": games[False],
        "games_on": games[True],
        "saved_bytes_per_game": sd.stats(directory)["bytes_per_game"],
    }
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("size", "ab"):
        p = sub.add_parser(name)
        p.add_argument("--run-config", default=None, help="A search-learning run's config.json (its search settings)")
        p.add_argument("--data-root", default=str(REPO_ROOT), help="Where the run's relative human_games/ paths live")
        p.add_argument("--games-per-worker", type=int, default=24)
        p.add_argument("--seconds", type=float, default=600)
        p.add_argument("--shard-games", type=int, default=256)
    p = sub.choices["size"]
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task-seconds", type=float, default=60)
    p.add_argument("--torch-threads", type=int, default=12)
    p.add_argument("--preset", type=int, default=None, help="xz preset (default search_data.PRESET)")
    p.add_argument("--gpu", action="store_true")
    p = sub.choices["ab"]
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=float, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Where the saved games go (default: a new temporary directory)")
    args = parser.parse_args()
    if args.command == "size" and not args.run_config:
        raise SystemExit("size needs --run-config")
    {"size": cmd_size, "ab": cmd_ab}[args.command](args)


if __name__ == "__main__":
    main()
