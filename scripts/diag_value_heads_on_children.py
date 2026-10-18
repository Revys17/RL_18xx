"""Can a frozen trunk rank siblings once its value heads see children? (shared-network diagnosis, step 3)

``diag_sibling_values.py build`` sets hold positions and every child of each. Here a
win head is fit on a trunk's frozen value-head input (``capture_value_input``) toward
a reference critic's win probabilities -- on the positions only (the refit's
distribution: played lines) or on the positions and their children (the search's) --
and scored on a held-out set's sibling ranking against the same critic. A trunk
whose features hold what the critic sees ranks well once its head is fit on
children; one that lacks them does not, whatever its head.

Usage::

    uv run python scripts/diag_value_heads_on_children.py --train logs/shared_diag/train_children \\
        --test logs/shared_diag/siblings_v2 --reference critic280=model_checkpoints_search/az1_20261017/critic/280.pth \\
        --trunk shared=model_checkpoints_pg/az1u280_shared_20261018/learner/280.pth \\
        --trunk critic=model_checkpoints_search/az1_20261017/critic/280.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import diag_sibling_values as dsv  # noqa: E402

LOGGER = logging.getLogger("diag_value_heads_on_children")
NUM_PLAYERS = 4


def load(data_dir: Path):
    return dsv.load_data(data_dir), json.loads((data_dir / "meta.json").read_text())


def reference_values(path: str, name: str, data_dir: Path, data: dict) -> np.ndarray:
    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value

    cache = data_dir / f"values.{name}.npy"
    if cache.exists():
        return np.load(cache)
    model = load_policy_value(path)
    canon = dsv.child_values(model, data, model.device)
    np.save(cache, canon)
    del model
    torch.cuda.empty_cache()
    return canon


def seat_inputs(model, data: dict, batch_size: int = 256) -> np.ndarray:
    """(rows, 4, F) float16: the value heads' input for every stored state."""
    import torch

    out = []
    device = model.device
    with torch.no_grad():
        for b in range(0, len(data["game_states"]), batch_size):
            gs = torch.from_numpy(data["game_states"][b : b + batch_size]).float().to(device)
            nodes = torch.from_numpy(data["maps"][data["map_index"][b : b + batch_size]]).float().to(device)
            rt = torch.from_numpy(data["round_types"][b : b + batch_size]).long().to(device)
            npl = torch.full((len(gs),), NUM_PLAYERS, dtype=torch.long, device=device)
            store, handle = model.capture_value_input()
            try:
                model(gs, nodes, rt, None, npl, value_only=True)
            finally:
                handle.remove()
            out.append(store["seat_input"][:, :NUM_PLAYERS].to(torch.float16).cpu().numpy())
    return np.concatenate(out)


def row_sets(meta: dict) -> dict:
    """Row indices of the positions and of the non-terminal children, with each row's game."""
    parents = [p["enc"] for p in meta["positions"]]
    parent_game = [p["game"] for p in meta["positions"]]
    children, child_game = [], []
    for p in meta["positions"]:
        lo, hi = p["children"]
        for c in range(lo, hi):
            ch = meta["children"][c]
            if not ch["terminal"]:
                children.append(ch["enc"])
                child_game.append(p["game"])
    return {"parents": (np.asarray(parents), parent_game), "children": (np.asarray(children), child_game)}


def fit_win_head(head, x: np.ndarray, y: np.ndarray, games: list, device, lr=3e-4, epochs=40, patience=4, seed=0):
    """Fit ``head`` ((B, 4, F) -> (B, 4, 1) logits) to soft targets ``y`` by CE; 10% of games held out."""
    import torch

    torch.manual_seed(seed)
    head = copy.deepcopy(head).float().to(device)
    val = np.array([(zlib.crc32(g.encode()) % 10) == 0 for g in games])
    xt, yt = torch.from_numpy(x[~val]).to(device), torch.from_numpy(y[~val]).float().to(device)
    xv, yv = torch.from_numpy(x[val]).to(device), torch.from_numpy(y[val]).float().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def ce(xb, yb):
        return -(yb * torch.log_softmax(head(xb.float()).squeeze(-1), dim=1)).sum(1).mean()

    def evaluate():
        head.eval()
        with torch.no_grad():
            return float(sum(ce(xv[b : b + 4096], yv[b : b + 4096]) * len(xv[b : b + 4096]) for b in range(0, len(xv), 4096)) / max(len(xv), 1))

    best, best_state, bad, history = evaluate(), copy.deepcopy(head.state_dict()), 0, []
    history.append(best)
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(xt))
        for b in range(0, len(order), 1024):
            idx = torch.from_numpy(order[b : b + 1024]).to(device)
            loss = ce(xt[idx], yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = evaluate()
        history.append(v)
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head.eval(), history


def head_canon(head, x: np.ndarray, device) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for b in range(0, len(x), 4096):
            xb = torch.from_numpy(x[b : b + 4096]).float().to(device)
            out.append(torch.softmax(head(xb).squeeze(-1), dim=1).cpu().numpy())
    return np.concatenate(out)


def reset(head):
    import torch

    head = copy.deepcopy(head)
    for m in head.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            torch.nn.init.constant_(m.bias, 0)
    torch.nn.init.zeros_(head[-1].weight)
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="name=<critic checkpoint>")
    parser.add_argument("--trunk", action="append", required=True, help="name=<checkpoint>")
    parser.add_argument("--out", type=Path, default=Path("logs/shared_diag/heads_on_children.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value

    ref_name, _, ref_path = args.reference.partition("=")
    train, train_meta = load(args.train)
    test, test_meta = load(args.test)
    y_train = reference_values(ref_path, ref_name, args.train, train)
    y_test_canon = reference_values(ref_path, ref_name, args.test, test)
    ref_values = dsv.mover_values(y_test_canon, test, test_meta)
    sets = row_sets(train_meta)
    report = {}
    for item in args.trunk:
        name, _, path = item.partition("=")
        model = load_policy_value(path)
        device = model.device
        x_train = seat_inputs(model, train)
        x_test = seat_inputs(model, test)
        own = dsv.mover_values(head_canon(model.win_loss_head, x_test, device), test, test_meta)
        variants = {f"{name}:own": own}
        for rows_name in ("parents", "parents+children"):
            rows = sets["parents"][0] if rows_name == "parents" else np.concatenate([sets["parents"][0], sets["children"][0]])
            games = sets["parents"][1] + ([] if rows_name == "parents" else sets["children"][1])
            for init in ("own", "reset"):
                start = model.win_loss_head if init == "own" else reset(model.win_loss_head)
                head, history = fit_win_head(start, x_train[rows], y_train[rows], games, device)
                LOGGER.info(f"{name} {rows_name} init={init}: val CE {history[0]:.4f} -> {min(history):.4f} ({len(history) - 1} epochs, {len(rows)} rows)")
                variants[f"{name}:{rows_name}:{init}"] = dsv.mover_values(head_canon(head, x_test, device), test, test_meta)
        values = {ref_name: ref_values, **variants}
        rows_m = dsv.position_metrics(values, test_meta, ref_name)
        summary = dsv.summarize(rows_m, list(values))
        print(dsv.format_table({k: summary[k] for k in ("all", "step:stock", "step:track", "step:train") if k in summary}, list(values)))
        report[name] = summary
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
