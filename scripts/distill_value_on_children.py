"""Distill a critic into a shared policy+value network on the search's distribution.

The shared network's value heads were fit on positions along played lines; a search
compares a position's children, bad moves included, and there the refit ranks no
better than the policy prior (``diag_sibling_values.py``). Here the whole network --
trunk included -- is trained toward a teacher critic's win probabilities (and its
score head's output) on positions *and every child of each*
(``diag_sibling_values.py build`` sets), with KL(original || tuned) at the run's
temperature on the policy over each state's legal moves and on the price head
keeping the policy where it was. Optionally ``--outcome-weight`` mixes the game's
outcome into the positions' (not the children's) win targets.

Checkpoints are chosen on held-out games of the training sets (distillation CE); the
sibling ranking on ``--test`` (a held-out ``build`` set) and the policy drift are
logged at each evaluation.

Usage::

    uv run python scripts/distill_value_on_children.py --data logs/shared_diag/train_children \\
        --test logs/shared_diag/siblings_v2 --init model_checkpoints_pg/az1u280_shared_20261018/learner/280.pth \\
        --teacher critic280=model_checkpoints_search/az1_20261017/critic/280.pth \\
        --out model_checkpoints_pg/shared_diag_distill_20261018/learner/280.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import diag_sibling_values as dsv  # noqa: E402

LOGGER = logging.getLogger("distill_value_on_children")
NUM_PLAYERS = 4


def teacher_outputs(path: str, name: str, data_dir: Path, data: dict) -> tuple:
    """``(win (rows, 4), score (rows, 4))`` of the teacher on every stored state, canonical (cached)."""
    import torch
    import torch.nn.functional as F

    from rl18xx.agent.alphazero.composite_model import load_policy_value

    win_cache, score_cache = data_dir / f"values.{name}.npy", data_dir / f"scores.{name}.npy"
    if win_cache.exists() and score_cache.exists():
        return np.load(win_cache), np.load(score_cache)
    model = load_policy_value(path)
    device = model.device
    win, score = [], []
    with torch.no_grad():
        for b in range(0, len(data["game_states"]), 256):
            gs = torch.from_numpy(data["game_states"][b : b + 256]).float().to(device)
            nodes = torch.from_numpy(data["maps"][data["map_index"][b : b + 256]]).float().to(device)
            rt = torch.from_numpy(data["round_types"][b : b + 256]).long().to(device)
            npl = torch.full((len(gs),), NUM_PLAYERS, dtype=torch.long, device=device)
            _, w, s, _ = model(gs, nodes, rt, None, npl, value_only=True)
            win.append(F.softmax(w.float(), -1)[:, :NUM_PLAYERS].cpu().numpy())
            score.append(s.float()[:, :NUM_PLAYERS].cpu().numpy())
    win, score = np.concatenate(win), np.concatenate(score)
    np.save(win_cache, win)
    np.save(score_cache, score)
    del model
    torch.cuda.empty_cache()
    return win, score


class Rows:
    """Every non-terminal state of one or more ``build`` sets, with its targets."""

    def __init__(self, dirs: list, teacher: tuple, outcome_weight: float, val_fraction: float):
        self.sets = []
        for d in dirs:
            data = dsv.load_data(Path(d))
            meta = json.loads((Path(d) / "meta.json").read_text())
            win, score = teacher_outputs(teacher[1], teacher[0], Path(d), data)
            game_of_row = np.empty(len(data["game_states"]), dtype=object)
            is_parent = np.zeros(len(data["game_states"]), dtype=bool)
            outcome = np.full((len(data["game_states"]), NUM_PLAYERS), np.nan, dtype=np.float32)
            for p in meta["positions"]:
                game_of_row[p["enc"]] = p["game"]
                is_parent[p["enc"]] = True
                rot = int(data["rotations"][p["enc"]])
                outcome[p["enc"]] = np.roll(np.asarray(p["win"], dtype=np.float32), -rot)
                lo, hi = p["children"]
                for c in range(lo, hi):
                    ch = meta["children"][c]
                    if not ch["terminal"]:
                        game_of_row[ch["enc"]] = p["game"]
            target = win.copy()
            if outcome_weight > 0:
                target[is_parent] = (1 - outcome_weight) * win[is_parent] + outcome_weight * outcome[is_parent]
            offsets = np.concatenate([[0], np.cumsum(data["legal_lengths"])])
            val = np.array([(zlib.crc32(str(g).encode()) % 10_000) < val_fraction * 10_000 for g in game_of_row])
            self.sets.append({"data": data, "target": target, "score": score, "offsets": offsets, "val": val,
                              "parent": is_parent, "dir": str(d)})
        self.index = [(k, r) for k, s in enumerate(self.sets) for r in range(len(s["target"]))]
        self.train = np.array([i for i, (k, r) in enumerate(self.index) if not self.sets[k]["val"][r]])
        self.validation = np.array([i for i, (k, r) in enumerate(self.index) if self.sets[k]["val"][r]])

    def batch(self, ids, device) -> dict:
        import torch

        from rl18xx.agent.alphazero.mcts import POLICY_SIZE
        from rl18xx.agent.alphazero.policy_gradient import _lay_tile_range

        gs, nodes, rt, target, score, legal_rows = [], [], [], [], [], []
        for i in ids:
            k, r = self.index[int(i)]
            s = self.sets[k]
            d = s["data"]
            gs.append(d["game_states"][r])
            nodes.append(d["maps"][d["map_index"][r]])
            rt.append(d["round_types"][r])
            target.append(s["target"][r])
            score.append(s["score"][r])
            legal_rows.append(d["legal_flat"][s["offsets"][r] : s["offsets"][r + 1]])
        legal = torch.zeros(len(ids), POLICY_SIZE, dtype=torch.bool)
        for b, ix in enumerate(legal_rows):
            legal[b, torch.from_numpy(ix.astype(np.int64))] = True
        lo, hi = _lay_tile_range()
        return {
            "gs": torch.from_numpy(np.stack(gs)).float().to(device),
            "nodes": torch.from_numpy(np.stack(nodes)).float().to(device),
            "rt": torch.tensor(rt, dtype=torch.long, device=device),
            "npl": torch.full((len(ids),), NUM_PLAYERS, dtype=torch.long, device=device),
            "target": torch.from_numpy(np.stack(target)).float().to(device),
            "score": torch.from_numpy(np.stack(score)).float().to(device),
            "legal": legal.to(device),
            "tiles": legal[:, lo:hi].any(dim=1),
        }


def losses(original, tuned, batch, temperature: float):
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _masked_log_softmax

    with torch.no_grad():
        logits0, _, _, _ = original(batch["gs"], batch["nodes"], batch["rt"], None, batch["npl"], lay_tile_rows=batch["tiles"])
        price0 = original.last_price_components["price_logits"]
    logits, win_logits, score, _ = tuned(batch["gs"], batch["nodes"], batch["rt"], None, batch["npl"], lay_tile_rows=batch["tiles"])
    price = tuned.last_price_components["price_logits"]
    legal = batch["legal"]
    log_v = torch.log_softmax(win_logits.float()[:, :NUM_PLAYERS], dim=1)
    ce = -(batch["target"] * log_v).sum(1).mean()
    floor = -(batch["target"] * torch.log(batch["target"].clamp_min(1e-9))).sum(1).mean()
    mse = ((score.float()[:, :NUM_PLAYERS] - batch["score"]) ** 2).mean()
    lp0 = _masked_log_softmax(logits0 / temperature, legal)
    lp1 = _masked_log_softmax(logits / temperature, legal)
    kl = ((lp0.exp() * legal) * (lp0 - lp1).masked_fill(~legal, 0.0)).sum(1).mean()
    c0, c1 = torch.log_softmax(price0.float(), -1), torch.log_softmax(price.float(), -1)
    price_kl = (c0.exp() * (c0 - c1)).sum(-1).mean()
    agree = (lp0.masked_fill(~legal, -1e30).argmax(1) == lp1.masked_fill(~legal, -1e30).argmax(1)).float().mean()
    return {"ce": ce, "kl_target": ce - floor, "mse": mse, "kl": kl, "price_kl": price_kl, "top1": agree}


def evaluate(original, tuned, rows: Rows, test, temperature: float, device, max_rows: int = 8192) -> dict:
    import torch

    tuned.eval()
    out = {}
    ids = rows.validation[:max_rows]
    sums, n = {}, 0
    with torch.no_grad():
        for b in range(0, len(ids), 256):
            batch = rows.batch(ids[b : b + 256], device)
            stats = losses(original, tuned, batch, temperature)
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(batch["gs"])
            n += len(batch["gs"])
    out.update({f"val_{k}": v / max(n, 1) for k, v in sums.items()})
    if test is not None:
        data, meta, ref_name, ref_values = test
        canon = dsv.child_values(tuned, data, device)
        values = {ref_name: ref_values, "tuned": dsv.mover_values(canon, data, meta)}
        summary = dsv.summarize(dsv.position_metrics(values, meta, ref_name), list(values))
        for g in ("all", "step:stock", "step:track", "step:train"):
            if g in summary:
                for key in ("spearman", "top1", "regret"):
                    out[f"test_{g}_{key}"] = summary[g]["tuned"][key]
        # The test positions' win CE against their games' outcomes.
        parents = np.array([p["enc"] for p in meta["positions"]])
        rot = data["rotations"][parents]
        win = np.stack([np.roll(np.asarray(p["win"]), -r) for p, r in zip(meta["positions"], rot)])
        pr = np.clip(canon[parents], 1e-9, 1)
        out["test_outcome_ce"] = float(-(win * np.log(pr / pr.sum(1, keepdims=True))).sum(1).mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="diag_sibling_values build sets to train on")
    parser.add_argument("--test", type=Path, default=None, help="A held-out build set for the sibling ranking")
    parser.add_argument("--init", required=True, help="The network to start from (also the policy anchor by default)")
    parser.add_argument(
        "--policy-teacher", default=None,
        help="The policy (and price head) to match: KL(teacher || tuned) at T (default: --init, an anchor)",
    )
    parser.add_argument(
        "--reset-policy-head", action="store_true",
        help="Start the policy and price heads from --policy-teacher's (with a critic as --init)",
    )
    parser.add_argument("--teacher", required=True, help="name=<critic checkpoint>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trunk-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=1.0, help="The policy (and price) KL's weight")
    parser.add_argument("--value-weight", type=float, default=1.0, help="The value targets' (win CE + score MSE) weight")
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--outcome-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    from rl18xx.agent.alphazero import value_refit as vr
    from rl18xx.agent.alphazero.policy_gradient import _load, use_tf32

    use_tf32()
    torch.manual_seed(args.seed)
    teacher = tuple(args.teacher.split("=", 1))
    rows = Rows(args.data, teacher, args.outcome_weight, args.val_fraction)
    LOGGER.info(f"{len(rows.train)} training states, {len(rows.validation)} validation states")
    test = None
    if args.test is not None:
        tdata = dsv.load_data(args.test)
        tmeta = json.loads((args.test / "meta.json").read_text())
        twin, _ = teacher_outputs(teacher[1], teacher[0], args.test, tdata)
        test = (tdata, tmeta, teacher[0], dsv.mover_values(twin, tdata, tmeta))

    # The policy target: the starting network itself (an anchor) unless --policy-teacher names
    # another network (e.g. starting from the critic and distilling the policy into it).
    original = _load(args.policy_teacher or args.init).eval()
    for p in original.parameters():
        p.requires_grad_(False)
    tuned = _load(args.init).eval()  # eval: no dropout, the policy trained is the one played
    if args.reset_policy_head:
        start = _load(args.policy_teacher)
        tuned.policy_head.load_state_dict(start.policy_head.state_dict())
        tuned.price_head.load_state_dict(start.price_head.state_dict())
        del start
    for p in tuned.parameters():
        p.requires_grad_(True)
    device = next(tuned.parameters()).device
    heads = {id(p) for h in (tuned.win_loss_head, tuned.score_head) for p in h.parameters()}
    groups = [
        {"params": [p for p in tuned.parameters() if id(p) not in heads], "lr": args.trunk_lr},
        {"params": [p for p in tuned.parameters() if id(p) in heads], "lr": args.head_lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps = int(args.epochs * len(rows.train) / args.batch_size)

    history = [{"step": 0, **evaluate(original, tuned, rows, test, args.temperature, device)}]
    LOGGER.info(f"step 0: {history[-1]}")

    def criterion(h):  # the validation loss (the value's distance from its target + the policy's KL)
        return args.value_weight * h["val_kl_target"] + args.kl_weight * (h["val_kl"] + h["val_price_kl"])

    best = (criterion(history[0]), copy.deepcopy(tuned.state_dict()), 0)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(rows.train)
    pos, t0 = 0, time.time()
    for step in range(1, steps + 1):
        if pos + args.batch_size > len(order):
            order, pos = rng.permutation(rows.train), 0
        ids = order[pos : pos + args.batch_size]
        pos += args.batch_size
        # Linear warmup, then cosine decay to 10%.
        frac = min(1.0, step / max(args.warmup, 1)) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * frac
        tuned.eval()
        batch = rows.batch(ids, device)
        stats = losses(original, tuned, batch, args.temperature)
        loss = (
            args.value_weight * (stats["ce"] + args.score_weight * stats["mse"])
            + args.kl_weight * (stats["kl"] + stats["price_kl"])
        )
        if not torch.isfinite(loss):
            raise RuntimeError("distillation diverged (non-finite loss)")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for group in opt.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
        opt.step()
        if step % 50 == 0:
            LOGGER.info(
                f"step {step}/{steps}: kl_target {float(stats['kl_target']):.4f} mse {float(stats['mse']):.4f} "
                f"policy kl {float(stats['kl']):.5f} price kl {float(stats['price_kl']):.5f} "
                f"top1 {float(stats['top1']):.3f} ({time.time() - t0:.0f}s)"
            )
        if step % args.eval_every == 0 or step == steps:
            history.append({"step": step, **evaluate(original, tuned, rows, test, args.temperature, device)})
            LOGGER.info(f"step {step}: {history[-1]}")
            if criterion(history[-1]) < best[0]:
                best = (criterion(history[-1]), copy.deepcopy(tuned.state_dict()), step)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    tuned.load_state_dict(best[1])
    provenance = {"distill": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                              "best_step": best[2], "history": history}}
    vr.write_model_checkpoint(tuned, args.policy_teacher or args.init, args.out, provenance)
    LOGGER.info(f"saved {args.out} (step {best[2]})")


if __name__ == "__main__":
    main()
