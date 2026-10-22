"""Value gate: does the value keep improving with more saved games, and do dense targets help?

    build  replay saved eval games (search systems) into positions: the canonical encoding, the
           outcome (win share, placement points) and the dense targets (rl18xx.agent.alphazero.
           dense_targets), sharded; games split into train / val (a hash of the game id) and test
           (whole collections, the newest games)
    train  fine-tune a value network (a whole network: entity and map encoders, trunk, heads) on
           the first N training games in a fixed order -- outcome losses alone, or + the dense
           targets -- early-stopped on the val games
    eval   held-out test games: win CE and winner hit by stage, score MSE; sibling ranking
           (scripts/diag_sibling_values.py score) against az1's critic 280

    uv run python scripts/value_gate.py build --out logs/value_gate/data --workers 8
    uv run python scripts/value_gate.py train --data logs/value_gate/data --games 5000 --dense-weight 0 \\
        --out logs/value_gate/runs/base_5000
    uv run python scripts/value_gate.py eval --data logs/value_gate/data --model base=<ckpt> ...
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("value_gate")
NUM_PLAYERS = 4
H = 100  # dense_targets.SHORT_HORIZON
EVAL_ROOT = REPO_ROOT / "logs" / "eval"
# siblings_v2 (the sibling-ranking set) was built from these collections' games: never train or test on them.
SIBLING_SOURCES = ("az1_u250_sys200", "az1_u300_sys200", "az2_u0_sys200")
DEFAULT_TEST = ("azl_u2100_*", "azlc_u2070_*", "ctrl_*")
DEFAULT_INIT = "model_checkpoints_pg/rab_place_20261014/critic/1800.pth"
POSITION_KEYS = (
    "game_states", "nodes", "rotation", "stage", "game", "win", "placement",
    "p_final", "pc_final", "p_short", "c_final", "c_pres", "c_next", "c_next_mask", "h_color", "h_token", "h_route",
)  # fmt: skip
STORE_DTYPES = {"rotation": np.int8, "stage": np.int8, "c_pres": np.int8, "h_color": np.int8, "h_token": np.uint8,
                "h_route": np.uint8}
DENSE_KEYS = ("p_final", "pc_final", "p_short", "c_final", "c_pres", "c_next", "c_next_mask", "h_color", "h_token",
              "h_route")  # fmt: skip


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def search_collections(root: Path, since: float, exclude) -> list:
    """``[(directory, auction_unlock)]``: eval_head_to_head collections of search players saved after
    ``since`` (an mtime), minus ``exclude`` patterns."""
    out = []
    for d in sorted(root.iterdir()):
        if any(fnmatch.fnmatch(d.name, p) for p in exclude):
            continue
        gj, settings = d / "games.jsonl", d / "settings.json"
        if not (gj.exists() and settings.exists() and (d / "games").is_dir()) or gj.stat().st_mtime < since:
            continue
        try:
            s = json.loads(settings.read_text())
        except ValueError:
            continue
        players = s.get("players")
        if not players or all("policy_temperature" in p.get("overrides", {}) for p in players.values()):
            continue
        out.append((d, not s.get("start_positions")))
    return out


def replay_game(game_id: str, actions: list, record: dict, unlock: bool, per_game: int) -> dict:
    """One game's positions (``per_game`` decisions past the auction, uniform over the game) with
    their targets, as arrays keyed like ``POSITION_KEYS`` (``game`` left to the caller)."""
    from rl18xx.agent.alphazero import dense_targets as dt
    from rl18xx.agent.alphazero.policy_gradient import placement_points
    from rl18xx.agent.alphazero.start_positions import new_game

    rng = np.random.default_rng(zlib.crc32(game_id.encode()))
    g = new_game(NUM_PLAYERS, auction_unlock=unlock)
    rs = g._game
    n = len(actions)
    drawn = set(rng.choice(n, size=min(n, 3 * per_game), replace=False).tolist()) if n else set()
    need_short = {min(i + H, n) for i in drawn}
    shares_at, rows = {}, []
    for i, action in enumerate(actions):
        if i in need_short:
            shares_at[i] = dt.net_worth_shares(rs)
        if i in drawn and rs.round.round_type != "Auction" and len(rs.factored_legal_indices()) > 1:
            state, nodes, _, rotation, _ = rs.encode_canonical_py()
            rows.append((i, np.asarray(state, np.float32), np.asarray(nodes, np.float32), int(rotation)))
        rs.process_action(action)
    if not rs.finished:
        raise ValueError("the game did not finish on replay")
    final = dt.final_targets(rs, actions)
    shares_at[n] = final["p_final"][:, 0].copy()
    if len(rows) > per_game:
        keep = np.sort(rng.choice(len(rows), size=per_game, replace=False))
        rows = [rows[int(k)] for k in keep]
    revenues = dt.run_revenues(actions)
    win = np.asarray(record["win_share"], np.float32)
    placement = placement_points(np.asarray(record["net_worth"], np.float64)).astype(np.float32)
    out = {k: [] for k in POSITION_KEYS if k != "game"}
    for i, state, nodes, rotation in rows:
        per_game_targets = dict(final)
        per_game_targets["p_short"] = shares_at[min(i + H, n)]
        canon = dt.canonical(per_game_targets, rotation)
        nxt, mask = dt.next_run_targets(revenues, i)
        out["game_states"].append(state)
        out["nodes"].append(nodes.astype(np.float16))
        out["rotation"].append(rotation)
        out["stage"].append(min(2, 3 * i // max(n, 1)))
        out["win"].append(np.roll(win, -rotation))
        out["placement"].append(np.roll(placement, -rotation))
        for key in ("p_final", "pc_final", "p_short", "c_final", "c_pres", "h_color", "h_token", "h_route"):
            out[key].append(canon[key])
        out["c_next"].append(nxt)
        out["c_next_mask"].append(mask)
    return {k: np.stack(v).astype(STORE_DTYPES.get(k, np.stack(v).dtype)) if v else None for k, v in out.items()}


def collection_slice(directory: Path, lo: int, hi: int) -> list:
    """``value_refit.collection_games`` for the collection's games ``lo`` to ``hi`` only."""
    lines = [line for line in (directory / "games.jsonl").read_text().splitlines() if line.strip()][lo:hi]
    out = []
    for line in lines:
        record = json.loads(line)
        path = directory / "games" / f"{record['game_idx']}.json"
        if not path.exists():
            continue
        actions = json.loads(path.read_text())
        if isinstance(actions, dict):
            actions = actions.get("actions", [])
        out.append((f"{directory.name}/{record['game_idx']}", actions, record))
    return out


def _build_task(task) -> dict:
    """Pool task: a slice of one collection's games -> stacked arrays + per-game metadata."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    directory, unlock, lo, hi, per_game = task
    games = collection_slice(Path(directory), lo, hi)
    parts, meta, skipped = [], [], 0
    for game_id, actions, record in games:
        try:
            arrays = replay_game(game_id, actions, record, unlock, per_game)
        except Exception as e:  # a game recorded before an engine rule change
            LOGGER.debug(f"{game_id}: skipped ({e!r})")
            skipped += 1
            continue
        if arrays["game_states"] is None:
            continue
        parts.append(arrays)
        meta.append({"id": game_id, "positions": int(len(arrays["game_states"]))})
    if not parts:
        return {"arrays": None, "meta": meta, "skipped": skipped}
    stacked = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return {"arrays": stacked, "meta": meta, "skipped": skipped}


def build(args) -> None:
    from multiprocessing import get_context

    out = Path(args.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    since = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M"))
    collections = search_collections(Path(args.root), since, (*SIBLING_SOURCES, *args.exclude))
    tasks = []
    for directory, unlock in collections:
        n = sum(1 for line in (directory / "games.jsonl").read_text().splitlines() if line.strip())
        if args.max_games_per_collection:
            n = min(n, args.max_games_per_collection)
        for lo in range(0, n, args.chunk):
            tasks.append((directory, unlock, lo, min(n, lo + args.chunk), args.per_game))
    LOGGER.info(f"{len(collections)} collections, {len(tasks)} tasks")
    games, shards, skipped, n_pos = [], [], 0, 0
    started = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for k, result in enumerate(pool.imap_unordered(_build_task, tasks)):
            skipped += result["skipped"]
            if result["arrays"] is None:
                continue
            base = len(games)
            first = result["meta"][0]["id"].split("/")[0]
            test = any(fnmatch.fnmatch(first, p) for p in args.test)
            for m in result["meta"]:
                split = "test" if test else ("val" if (zlib.crc32(m["id"].encode()) % 1000) < 50 else "train")
                games.append({**m, "split": split})
            arrays = result["arrays"]
            counts = [m["positions"] for m in result["meta"]]
            arrays["game"] = np.repeat(np.arange(base, base + len(counts), dtype=np.int32), counts)
            name = f"shard_{len(shards):05d}.npz"
            np.savez(out / "shards" / name, **arrays)
            shards.append(name)
            n_pos += len(arrays["game"])
            if k % 20 == 0:
                LOGGER.info(f"{k + 1}/{len(tasks)} tasks, {len(games)} games, {n_pos} positions, {skipped} skipped, "
                            f"{time.time() - started:.0f}s")
    manifest = {"shards": shards, "games": games, "skipped": skipped, "per_game": args.per_game, "horizon": H,
                "collections": [str(d) for d, _ in collections], "test_patterns": list(args.test)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    by_split = {s: sum(1 for g in games if g["split"] == s) for s in ("train", "val", "test")}
    LOGGER.info(f"done: {len(games)} games {by_split}, {n_pos} positions, {skipped} skipped")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_data(data_dir: Path) -> tuple:
    """``(manifest, arrays)``: every shard concatenated (nodes stay float16)."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    parts = [dict(np.load(Path(data_dir) / "shards" / s)) for s in manifest["shards"]]
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return manifest, arrays


def split_positions(manifest: dict, arrays: dict, split: str) -> np.ndarray:
    splits = np.array([g["split"] for g in manifest["games"]])
    return np.flatnonzero(splits[arrays["game"]] == split)


def training_games(manifest: dict, n_games: int, seed: int = 0) -> np.ndarray:
    """The first ``n_games`` training games in a fixed random order (nested subsets across N)."""
    train = np.array([i for i, g in enumerate(manifest["games"]) if g["split"] == "train"])
    order = np.random.default_rng(seed).permutation(train)
    return order[:n_games]


def batch_tensors(arrays: dict, idx: np.ndarray, device, dense: bool) -> dict:
    import torch

    def t(key, dtype=torch.float32):
        return torch.from_numpy(np.asarray(arrays[key][idx])).to(device=device, dtype=dtype)

    out = {"gs": t("game_states"), "nodes": t("nodes"), "win": t("win"), "placement": t("placement")}
    if dense:
        for key in DENSE_KEYS:
            out[key] = t(key, torch.long if key in ("c_pres", "h_color") else torch.float32)
    return out


def predict(model, arrays: dict, idx: np.ndarray, device, batch_size: int = 512) -> dict:
    """Canonical win probabilities and score output (4 seats) of ``model`` at positions ``idx``."""
    import torch
    import torch.nn.functional as F

    win, score = [], []
    with torch.no_grad():
        for b in range(0, len(idx), batch_size):
            chunk = idx[b : b + batch_size]
            gs = torch.from_numpy(arrays["game_states"][chunk]).float().to(device)
            nodes = torch.from_numpy(arrays["nodes"][chunk]).float().to(device)
            _, w, s, _ = model(gs, nodes, value_only=True)
            win.append(F.softmax(w.float(), dim=-1)[:, :NUM_PLAYERS].cpu().numpy())
            score.append(s.float()[:, :NUM_PLAYERS].cpu().numpy())
    return {"win": np.concatenate(win), "score": np.concatenate(score)}


def report(arrays: dict, idx: np.ndarray, pred: dict) -> dict:
    """Win CE / winner hit overall and by stage, score MSE (``value_refit.stage_report``'s)."""
    win, stage, placement = arrays["win"][idx], arrays["stage"][idx], arrays["placement"][idx]
    p = np.clip(pred["win"] / pred["win"].sum(axis=1, keepdims=True), 1e-9, 1.0)
    ce = -(win * np.log(p)).sum(axis=1)
    hit = (np.take_along_axis(win, p.argmax(axis=1)[:, None], axis=1)[:, 0] > 0).astype(np.float64)
    out = {"n": int(len(idx)), "win_ce": float(ce.mean()), "win_ce_se": float(ce.std() / np.sqrt(len(ce))),
           "winner_hit": float(hit.mean()), "score_mse": float(((pred["score"] - placement) ** 2).mean())}
    for s, name in enumerate(("early", "mid", "late")):
        m = stage == s
        out[f"win_ce_{name}"] = float(ce[m].mean()) if m.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(args) -> None:
    import torch
    import torch.nn.functional as F

    from rl18xx.agent.alphazero import dense_targets as dt
    from rl18xx.agent.alphazero.policy_gradient import _load
    from rl18xx.agent.alphazero.value_refit import write_model_checkpoint

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, arrays = load_data(Path(args.data))
    games = set(training_games(manifest, args.games, args.seed).tolist())
    train_idx = np.flatnonzero(np.isin(arrays["game"], np.array(sorted(games))))
    val_idx = split_positions(manifest, arrays, "val")
    val_idx = np.random.default_rng(1).permutation(val_idx)[: args.val_positions]
    LOGGER.info(f"{len(games)} training games, {len(train_idx)} positions; {len(val_idx)} val positions")

    model = _load(REPO_ROOT / args.init if not Path(args.init).is_absolute() else args.init).to(device)
    model.eval()  # no dropout: the network trained is the one searched with
    dense = args.dense_weight > 0
    heads = dt.build_dense_heads(model).to(device) if dense else None
    tap = dt.ActivationTap(model) if dense else None
    groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if dense:
        groups.append({"params": list(heads.parameters()), "lr": args.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    def val_ce() -> float:
        return report(arrays, val_idx, predict(model, arrays, val_idx, device))["win_ce"]

    history = [{"step": 0, "val_ce": val_ce()}]
    best = (history[0]["val_ce"], {k: v.detach().clone() for k, v in model.state_dict().items()}, 0)
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    max_steps = min(args.max_steps, int(args.max_epochs * steps_per_epoch))
    eval_every = max(50, min(args.eval_every, steps_per_epoch))
    bad, step, started = 0, 0, time.time()
    while step < max_steps and bad < args.patience:
        order = rng.permutation(train_idx)
        for b in range(0, len(order) - args.batch_size + 1, args.batch_size):
            batch = batch_tensors(arrays, order[b : b + args.batch_size], device, dense)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, win_logits, score, _ = model(batch["gs"], batch["nodes"], value_only=True)
            logp = torch.log_softmax(win_logits.float()[:, :NUM_PLAYERS], dim=1)
            ce = -(batch["win"] * logp).sum(1).mean()
            mse = F.mse_loss(score.float()[:, :NUM_PLAYERS], batch["placement"])
            loss = ce + args.score_weight * mse
            parts = {}
            if dense:
                dense_total, parts = dt.dense_loss(heads, tap, batch)
                loss = loss + args.dense_weight * dense_total
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for group in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            opt.step()
            step += 1
            if step % eval_every == 0 or step >= max_steps:
                v = val_ce()
                history.append(
                    {"step": step, "val_ce": v, "train_ce": float(ce.detach()), "epoch": step / steps_per_epoch,
                     **{f"dense_{k}": x for k, x in parts.items()}}
                )
                LOGGER.info(f"step {step}/{max_steps} epoch {step / steps_per_epoch:.2f} val CE {v:.4f} "
                            f"(best {best[0]:.4f}) train CE {float(ce.detach()):.4f} {time.time() - started:.0f}s")
                if v < best[0] - 1e-4:
                    best, bad = (v, {k: t.detach().clone() for k, t in model.state_dict().items()}, step), 0
                else:
                    bad += 1
            if step >= max_steps or bad >= args.patience:
                break
    model.load_state_dict(best[1])
    provenance = {"value_gate": vars(args) | {"best_step": best[2], "train_positions": int(len(train_idx))}}
    ckpt = write_model_checkpoint(model, args.init, out / "learner" / "0.pth", provenance)
    (out / "history.json").write_text(json.dumps({"best_val_ce": best[0], "best_step": best[2], "history": history},
                                                 indent=1))
    LOGGER.info(f"saved {ckpt} (best val CE {best[0]:.4f} at step {best[2]})")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def sibling_spearman(name: str, checkpoint: str, data: str, critic: str) -> dict:
    """``diag_sibling_values.py score`` on siblings_v2 against az1's critic 280: Spearman by step type."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "diag_sibling_values.py"), "score", "--data", str(data),
           "--source", f"critic280={critic}", "--source", f"{name}={checkpoint}", "--reference", "critic280"]
    text = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
    out, section = {}, None
    for line in text.splitlines():
        if line.startswith("== "):
            section = line[3:].split()[0].replace("step:", "")
        elif section and line.split() and line.split()[0] == name:
            try:
                out[section] = float(line.split()[1])
            except (IndexError, ValueError):
                pass
    return out


def evaluate(args) -> None:
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, arrays = load_data(Path(args.data))
    test_idx = split_positions(manifest, arrays, "test")
    results = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        model = _load(REPO_ROOT / path if not Path(path).is_absolute() else path).to(device).eval()
        r = {"name": name, "path": path, **report(arrays, test_idx, predict(model, arrays, test_idx, device))}
        del model
        torch.cuda.empty_cache()
        if not args.no_siblings:
            checkpoint = str(REPO_ROOT / path if not Path(path).is_absolute() else path)
            r["sibling"] = sibling_spearman(name, checkpoint, args.siblings, args.sibling_critic)
        results.append(r)
        LOGGER.info(json.dumps(r))
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--root", default=str(EVAL_ROOT))
    p.add_argument("--out", required=True)
    p.add_argument("--since", default="2026-10-16 18:00", help="collections saved after (local time)")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--test", nargs="*", default=list(DEFAULT_TEST), help="test collections (patterns)")
    p.add_argument("--per-game", type=int, default=16)
    p.add_argument("--chunk", type=int, default=40, help="games per pool task / shard")
    p.add_argument("--max-games-per-collection", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--init", default=DEFAULT_INIT)
    p.add_argument("--dense-weight", type=float, default=0.0)
    p.add_argument("--score-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=float, default=8.0)
    p.add_argument("--max-steps", type=int, default=40_000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val-positions", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("--data", required=True)
    p.add_argument("--model", nargs="+", required=True, help="name=checkpoint")
    p.add_argument("--no-siblings", action="store_true")
    p.add_argument("--siblings", default=str(REPO_ROOT / "logs" / "shared_diag" / "siblings_v2"))
    p.add_argument("--sibling-critic", default=str(REPO_ROOT / "model_checkpoints_search/az1_20261017/critic/280.pth"))
    p.add_argument("--out", default=None, help="append results (JSONL)")
    args = parser.parse_args()
    {"build": build, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
