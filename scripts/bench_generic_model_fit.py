"""Model fit of the title-generic network (``model_generic``) per title, on the CPU: time and memory of a
forward + backward (training) and of a no-grad forward (inference) of the default-size model on random-walk
positions of each registered title, so the 1822 family (1822CA: 279 hexes, 138 entities) can be compared with
1830 / 1867 (``docs/rl_model.md`` "Model fit across titles").

Modes:

* ``value``: the value path alone (``value_only=True``): map encoder, entities, fusion, trunk, value heads;
* ``real``: the full forward with the position's real legal actions (titles with an RL layout: 1830, 1867);
* ``lays``: the full forward with a stub legal list of ``--lays`` tile lays per position -- every legal
  rotation-free (hex, tile, rotation) triple of random hexes, as a lay-heavy position's list would hold --
  plus a pass (no RL layout is needed: the rows are well-formed descriptors, not legal moves). It measures
  the map factors and the rotation factor's edge term at a given lay count.

Each (title, mode) runs in a child process, so its peak RSS is its own. Reported per position: training
step (forward + backward) and inference (no grad) time, the bytes autograd saves for the backward (summed
over distinct storages: the activation memory a training step holds), and the child's peak-RSS growth
during the measured steps.

    CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python scripts/bench_generic_model_fit.py \\
        --titles 1830 1867 1822 1822PNW 1822MX 1822CA --modes value lays --batch 32 --threads 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def walk_positions(title: str, players: int, count: int, seed: int) -> list:
    """``count`` positions of random walks of ``title`` (every 3rd position from move 60 on, several games)."""
    import engine_rs

    from rl18xx.agent.alphazero.generic_tokens import raw_game
    from rl18xx.agent.alphazero.titles import new_game

    rng = random.Random(seed)
    out: list = []
    game_seed = seed
    while len(out) < count:
        game_seed += 1
        if title == "1830":
            g = raw_game(new_game("1830", players, auction_unlock=True))
        else:
            ids = {1000 + i + 1: f"Player {i + 1}" for i in range(players)}
            g = engine_rs.BaseGame.new_titled(title, ids, None, game_seed)
        for step in range(900):
            if g.finished or len(out) >= count:
                break
            if title == "1830":
                legal = [int(i) for i in g.factored_legal_indices()]
                if not legal:
                    break
                if step >= 60 and step % 3 == 0:
                    out.append(g.pickle_clone())
                index = rng.choice(legal)
                pr = g.price_range_for_index(index)
                g.apply_action_index(index, rng.randint(*pr) if pr else None)
                continue
            if title == "1867":
                from scripts.g1867_fuzz import pick

                moves = json.loads(g.g1867_legal_moves_json(None))
            else:
                from scripts.g1822_fuzz import pick

                moves = json.loads(g.g1822_legal_moves_json(None))
            if not moves:
                break
            if step >= 60 and step % 3 == 0:
                out.append(g.pickle_clone())
            try:
                g.process_action(dict(pick(moves, rng)))
            except Exception:  # noqa: BLE001 -- a rare "(Ruby raises)" dead end: next game
                break
    return out[:count]


def stub_lays(game, n_lays: int, rng: random.Random):
    """A stub legal list: ``n_lays`` LayTile rows (random hexes, every tile x rotation of each until the
    count is reached) + a pass, as descriptor rows of the position's title."""
    import numpy as np

    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_tokens import spec_info

    info = spec_info(game.title)
    fam = mg.vocab()["families"]
    lay, pas = fam.index("LayTile"), fam.index("Pass")
    rows = []
    hexes = list(range(info["num_hexes"]))
    rng.shuffle(hexes)
    tiles = list(range(info["num_tiles"]))
    for h in hexes:
        if len(rows) >= n_lays:
            break
        for t in rng.sample(tiles, min(len(tiles), 8)):
            for r in range(6):
                if len(rows) >= n_lays:
                    break
                rows.append([lay, mg.EK_CORPORATION, 0, -1, -1, -1, h, -1, t, r, -1, -1, -1, 0])
    rows.append([pas] + [-1] * 13)
    rows = np.asarray(rows, dtype=np.int32)
    rows[-1, mg.C_FLAGS] = 0
    return mg.LegalActions(rows=rows, price_range=np.full((len(rows), 3), -1, dtype=np.int64))


def child(args) -> dict:
    import numpy as np
    import torch

    from rl18xx.agent.alphazero import model_generic as mg

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    title, mode = args.child
    players = args.players or (5 if title == "1822CA" else 4)
    games = walk_positions(title, players, args.batch, seed=args.seed)
    arrays = mg.build_inputs(games, dedupe_maps=args.dedupe)
    if mode == "real":
        arrays = mg.add_legal_inputs(arrays, [mg.legal_actions(g) for g in games])
    elif mode == "lays":
        rng = random.Random(args.seed)
        arrays = mg.add_legal_inputs(arrays, [stub_lays(g, args.lays, rng) for g in games])
    config = mg.GenericModelConfig(**json.loads(args.config)) if args.config else mg.GenericModelConfig()
    if args.edge_term == "reference":  # B2's original edge term (projection after the gather), for comparison
        mg.GenericPolicyValueModel._edge_term = mg.GenericPolicyValueModel._edge_term_reference
    model = mg.GenericPolicyValueModel(config)
    model.checkpoint_layers = args.checkpoint
    inputs = mg.to_device(arrays)
    value_only = mode == "value"
    b = len(games)
    n_rows = int(len(arrays.get("legal_rows", []))) if not value_only else 0
    n_maps = int(arrays["hex_feat"].shape[0])

    def loss_of(out):
        loss = out["win_logits"].float().clamp(min=-50).sum() * 1e-3 + out["score"].sum() * 1e-3
        if out["logp"] is not None:
            loss = loss + out["logp"].sum() * 1e-3
        return loss

    # warm-up (allocator, lazy init)
    model.zero_grad(set_to_none=True)
    loss_of(model(inputs, value_only=value_only)).backward()

    # bytes saved for the backward (distinct storages; the parameters themselves excluded)
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    params |= {b_.untyped_storage().data_ptr() for b_ in model.buffers()}
    seen: dict = {}

    def pack(t):
        try:
            key = t.untyped_storage().data_ptr()
            if key not in params:
                seen[key] = max(seen.get(key, 0), t.untyped_storage().nbytes())
        except Exception:  # noqa: BLE001
            pass
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(inputs, value_only=value_only)
    saved = sum(seen.values())
    del out

    rss0 = _rss_bytes()
    peak0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    t = time.perf_counter()
    for _ in range(args.reps):
        model.zero_grad(set_to_none=True)
        loss_of(model(inputs, value_only=value_only)).backward()
    train = (time.perf_counter() - t) / args.reps
    peak1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model(inputs, value_only=value_only)
        t = time.perf_counter()
        for _ in range(args.reps):
            model(inputs, value_only=value_only)
        infer = (time.perf_counter() - t) / args.reps
    return {
        "title": title,
        "mode": mode,
        "players": players,
        "batch": b,
        "maps": n_maps,
        "hexes": int(arrays["hex_feat"].shape[1]),
        "entities": int(arrays["ent_type"].shape[1]),
        "relations": int(arrays["rel_val"].shape[1]),
        "legal_rows": n_rows,
        "train_ms_per_pos": 1e3 * train / b,
        "infer_ms_per_pos": 1e3 * infer / b,
        "saved_mb_per_pos": saved / b / 2**20,
        "peak_rss_growth_mb_per_pos": max(0, peak1 - max(peak0, rss0)) / b / 2**20,
        "params": model.num_parameters(),
        "checkpoint": bool(args.checkpoint),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--titles", nargs="+", default=["1830", "1867", "1822", "1822PNW", "1822MX", "1822CA"])
    p.add_argument("--modes", nargs="+", default=["value", "lays"], choices=["value", "real", "lays"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--players", type=int, default=0, help="0: 4 (1822CA: 5)")
    p.add_argument("--lays", type=int, default=300, help="stub lay rows per position (mode lays)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dedupe", action="store_true", help="de-duplicate maps (positions of one map share its encoder)")
    p.add_argument("--config", default="", help="GenericModelConfig overrides as JSON")
    p.add_argument(
        "--edge-term", default="current", choices=["current", "reference"],
        help="reference: B2's original edge term (the rotation factor's neighbor projection after the gather)",
    )
    p.add_argument("--checkpoint", action="store_true", help="gradient checkpointing of the attention blocks")
    p.add_argument("--out", default="", help="append result lines (JSON) here")
    p.add_argument("--child", nargs=2, metavar=("TITLE", "MODE"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        print(json.dumps(child(args)), flush=True)
        return
    for title in args.titles:
        for mode in args.modes:
            if mode == "real" and title not in ("1830", "1867"):
                continue
            cmd = [sys.executable, __file__, "--child", title, mode] + [
                a for k, v in vars(args).items() if k not in ("child", "titles", "modes", "out", "dedupe", "checkpoint")
                for a in (f"--{k.replace('_', '-')}", str(v))
            ] + (["--dedupe"] if args.dedupe else []) + (["--checkpoint"] if args.checkpoint else [])
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode:
                print(f"{title} {mode}: failed\n{res.stderr[-2000:]}", flush=True)
                continue
            line = res.stdout.strip().splitlines()[-1]
            r = json.loads(line)
            print(
                f"{title:8s} {mode:5s} B={r['batch']} H={r['hexes']} E={r['entities']} R={r['relations']} "
                f"rows={r['legal_rows']}: train {r['train_ms_per_pos']:.2f} ms/pos, infer {r['infer_ms_per_pos']:.2f} "
                f"ms/pos, saved {r['saved_mb_per_pos']:.1f} MB/pos, "
                f"peak RSS +{r['peak_rss_growth_mb_per_pos']:.1f} MB/pos",
                flush=True,
            )
            if args.out:
                with open(args.out, "a") as f:
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
