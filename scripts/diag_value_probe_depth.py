"""Where in a trunk is the information a critic ranks siblings by? (shared-network diagnosis)

Value heads read ``[player token, trunk output]``. This probes, for frozen networks,
win heads fit (toward a reference critic, on positions + children of
``diag_sibling_values.py build`` sets) on other inputs: the trunk after the fusion
and after each residual block, the player tokens alone, and a wider head on the
usual input -- each scored by sibling ranking on a held-out set.

Usage::

    uv run python scripts/diag_value_probe_depth.py --train logs/shared_diag/train_children \\
        --test logs/shared_diag/siblings_v2 --reference critic280=<critic.pth> --network shared=<shared.pth>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import diag_sibling_values as dsv  # noqa: E402
import diag_value_heads_on_children as dvh  # noqa: E402

LOGGER = logging.getLogger("diag_value_probe_depth")
NUM_PLAYERS = 4


def depth_features(model, data: dict, batch_size: int = 256) -> dict:
    """``{"tokens": (rows, 4, d_entity), "fusion": (rows, d_trunk), "block<i>": ...}`` float16."""
    import torch

    device = model.device
    store = {}
    hooks = [model.fusion.register_forward_hook(lambda m, a, o: store.__setitem__("fusion", o))]
    for i, block in enumerate(model.res_blocks):
        hooks.append(block.register_forward_hook(lambda m, a, o, i=i: store.__setitem__(f"block{i + 1}", o)))
    cap, handle = model.capture_value_input()
    hooks.append(handle)
    out = {}
    try:
        with torch.no_grad():
            for b in range(0, len(data["game_states"]), batch_size):
                gs = torch.from_numpy(data["game_states"][b : b + batch_size]).float().to(device)
                nodes = torch.from_numpy(data["maps"][data["map_index"][b : b + batch_size]]).float().to(device)
                rt = torch.from_numpy(data["round_types"][b : b + batch_size]).long().to(device)
                npl = torch.full((len(gs),), NUM_PLAYERS, dtype=torch.long, device=device)
                model(gs, nodes, rt, None, npl, value_only=True)
                d_entity = model.config.d_entity
                feats = {"tokens": cap["seat_input"][:, :NUM_PLAYERS, :d_entity], **{k: v for k, v in store.items()}}
                for k, v in feats.items():
                    out.setdefault(k, []).append(v.to(torch.float16).cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    return {k: np.concatenate(v) for k, v in out.items()}


def seat_input(feats: dict, key: str) -> np.ndarray:
    tokens = feats["tokens"]
    if key == "tokens":
        return tokens
    x = feats[key]
    return np.concatenate([tokens, np.repeat(x[:, None, :], NUM_PLAYERS, axis=1)], axis=2)


def make_head(in_dim: int, hidden: int, layers: int):
    import torch.nn as nn

    mods, d = [], in_dim
    for _ in range(layers - 1):
        mods.extend([nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()])
        d = hidden
    mods.append(nn.Linear(d, 1))
    head = nn.Sequential(*mods)
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--network", action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("logs/shared_diag/probe_depth.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value

    ref_name, _, ref_path = args.reference.partition("=")
    train, train_meta = dvh.load(args.train)
    test, test_meta = dvh.load(args.test)
    y_train = dvh.reference_values(ref_path, ref_name, args.train, train)
    y_test = dvh.reference_values(ref_path, ref_name, args.test, test)
    ref_values = dsv.mover_values(y_test, test, test_meta)
    sets = dvh.row_sets(train_meta)
    rows = np.concatenate([sets["parents"][0], sets["children"][0]])
    games = sets["parents"][1] + sets["children"][1]
    report = {}
    for item in args.network:
        name, _, path = item.partition("=")
        model = load_policy_value(path)
        device = model.device
        ftrain, ftest = depth_features(model, train), depth_features(model, test)
        values = {ref_name: ref_values}
        d_entity, d_trunk = model.config.d_entity, model.config.d_trunk
        probes = [("tokens", "tokens", d_trunk // 2, 3), ("fusion", "fusion", d_trunk // 2, 3)]
        probes += [(f"block{i}", f"block{i}", d_trunk // 2, 3) for i in range(1, len(model.res_blocks) + 1)]
        last = f"block{len(model.res_blocks)}"
        probes += [(f"{last}:wide", last, 1024, 4)]
        for label, key, hidden, layers in probes:
            x_train, x_test = seat_input(ftrain, key), seat_input(ftest, key)
            head = make_head(x_train.shape[-1], hidden, layers)
            head, history = dvh.fit_win_head(head, x_train[rows], y_train[rows], games, device)
            LOGGER.info(f"{name} {label}: val CE {min(history):.4f} ({len(history) - 1} epochs)")
            values[f"{name}:{label}"] = dsv.mover_values(dvh.head_canon(head, x_test, device), test, test_meta)
            report.setdefault(name, {})[label] = {"val_ce": float(min(history))}
        summary = dsv.summarize(dsv.position_metrics(values, test_meta, ref_name), list(values))
        print(dsv.format_table({k: summary[k] for k in ("all", "step:stock", "step:track") if k in summary}, list(values)))
        for label in report[name]:
            report[name][label].update({k: summary["all"][f"{name}:{label}"][k] for k in ("spearman", "top1", "regret")})
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
