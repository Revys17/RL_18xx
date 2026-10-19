"""How well does each value source rank a position's children -- what a search compares?

A search does not need calibrated values along the played line; it needs a
position's children ranked right, bad moves included. ``build`` samples decision
positions of saved games (``eval_head_to_head`` collections), expands each one's
root in a Rust search tree configured as the head-to-head eval's (the priors
from ``--expander``) and builds every child the way the search does
(``maybe_add_child``: forced follow-ups collapsed, an open price from the price
head's proposal), and stores the children's encodings. It also finds the move
the game actually played (the child whose encoding equals the recorded action's
result, with the recorded price). ``score`` evaluates each child with every
value source -- the win probability of the parent's mover, the way the search
reads it -- and reports, against a reference source, by game stage and step type:
within-position Spearman correlation, top-1 agreement, top-3 overlap, the
reference's regret of each source's best child, the value spread, and how often
each source's best child is the move played.

Usage::

    uv run python scripts/diag_sibling_values.py build --collection EV/az1_u250_sys200 \\
        --collection EV/az1_u300_sys200 --games 40 --per-game 9 --out logs/shared_diag/siblings
    uv run python scripts/diag_sibling_values.py score --data logs/shared_diag/siblings \\
        --source critic280=model_checkpoints_search/az1_20261017/critic/280.pth --source shared=... \\
        --reference critic280
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("diag_sibling_values")
NUM_PLAYERS = 4
STAGES = ("early", "mid", "late")
STEP_TYPES = ("stock", "track", "token", "route", "dividend", "train", "other")

EVAL_OVERRIDES = dict(
    num_readouts=200, min_readouts=200, c_puct_init=0.05, c_puct_by_round={}, mcts_mean_q=True, fpu_reduction=0.0,
    dedupe_leaves=True, parallel_readouts=8, prior_temperature=0.5,
)


# ---------------------------------------------------------------------------
# Positions and children
# ---------------------------------------------------------------------------


def step_type(rs, legal) -> str:
    """``search_learning.decision_step``: stock, track, token, route, dividend, train or other."""
    from rl18xx.agent.alphazero.search_learning import decision_step

    return decision_step(rs, legal)


def collection_games(directory: Path) -> list:
    """``value_refit.collection_games``, also for a policy-gradient run's saved games
    (``games.jsonl`` rows named by ``name``, ``make_game_record`` files)."""
    from rl18xx.agent.alphazero import value_refit as vr

    first = next((line for line in (directory / "games.jsonl").read_text().splitlines() if line.strip()), "{}")
    if "game_idx" in json.loads(first):
        return vr.collection_games(directory)
    out = []
    for line in (directory / "games.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "game_file" in record:
            path = directory / record["game_file"]
        elif "name" in record:
            path = directory / "games" / f"{record['name']}.json"
        else:
            continue
        if not path.exists() or "win_share" not in record or "net_worth" not in record:
            continue
        actions = json.loads(path.read_text())
        actions = actions.get("actions", []) if isinstance(actions, dict) else actions
        out.append((f"{directory.name}/{path.stem}", actions, record))
    return out


def sample_positions(
    collection: Path, games: int, per_game: int, rng: np.random.Generator, offset: int = 0
) -> list:
    """Decision positions (several legal moves, past the auction) drawn uniformly
    from each of the collection's games ``offset`` to ``offset + games``: ``(game id,
    action index, n actions, engine clone, recorded action, record)``."""
    from rl18xx.agent.alphazero.start_positions import new_game

    out = []
    for game_id, actions, record in collection_games(collection)[offset : offset + games]:
        g = new_game(NUM_PLAYERS)
        rs = g._game
        candidates = []
        try:
            for i, action in enumerate(actions):
                if g.finished:
                    break
                if rs.round.round_type != "Auction" and len(rs.factored_legal_indices()) > 1:
                    candidates.append(i)
                rs.process_action(action)
        except Exception as e:  # a game recorded before an engine rule change
            LOGGER.warning(f"{game_id}: skipped ({e})")
            continue
        if not candidates:
            continue
        picked = set(rng.choice(candidates, size=min(per_game, len(candidates)), replace=False).tolist())
        g = new_game(NUM_PLAYERS)
        rs = g._game
        for i, action in enumerate(actions):
            if i in picked:
                out.append((game_id, i, len(actions), rs.pickle_clone(), action, record))
            rs.process_action(action)
    return out


def played_index(rs, action: dict, legal) -> int:
    """The legal index whose result encodes like the recorded ``action``'s (-1: none)."""
    after = rs.pickle_clone()
    after.process_action(dict(action))
    target_state, target_nodes, *_ = after.encode_canonical_py()
    price = action.get("price")
    for ix in legal:
        c = rs.pickle_clone()
        pr = rs.price_range_for_index(int(ix))
        p = None
        if pr is not None:
            p = int(price) if price is not None and pr[0] <= int(price) <= pr[1] else int(pr[0])
        try:
            c.apply_action_index(int(ix), p)
        except Exception:
            continue
        state, nodes, *_ = c.encode_canonical_py()
        if np.array_equal(state, target_state) and np.array_equal(nodes, target_nodes):
            return int(ix)
    return -1


def build(args) -> None:
    import engine_rs
    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value
    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
    from rl18xx.rust_adapter import RustGameAdapter

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expander = load_policy_value(args.expander)
    searcher = LockstepSearcher({"m": ModelEvaluator(expander)}, dedupe_maps=False)

    positions = []
    for collection in args.collection:
        positions.extend(sample_positions(Path(collection), args.games, args.per_game, rng))
        LOGGER.info(f"{collection}: {len(positions)} positions so far")

    meta, child_meta = [], []
    store = Store()
    terminal_values = []
    t0 = time.time()
    for b in range(0, len(positions), 32):
        chunk = positions[b : b + 32]
        players, searches = [], []
        for game_id, i, n, rs, action, record in chunk:
            config = SelfPlayConfig(
                network=expander, use_inference_server=False, **EVAL_OVERRIDES, softpick_move_cutoff=0,
                dirichlet_noise_weight=0.0, enable_resign=False, noresign_holdout_rate=0.0, max_game_length=1000,
                player_count_distribution={NUM_PLAYERS: 1.0}, auction_unlock=True,
            )
            player = RustMCTSPlayer(config)
            player.initialize_game(RustGameAdapter(rs.pickle_clone()))
            s = player.start_search(key="m")
            players.append(player)
            if s is not None:
                searches.append(s)
        with torch.no_grad():
            searcher.round(searches)  # each root's expansion (priors, price head)
        for (game_id, i, n, rs, action, record), player in zip(chunk, players):
            tree = player._rust_player
            root = tree.root_idx
            stats = tree.node_stats(root)
            legal = [int(x) for x in stats["legal"]]
            prior = np.asarray(stats["child_prior"], dtype=np.float64)[: len(legal)]
            mover = int(rs.encode_canonical_py()[3])
            played = played_index(rs, action, legal) if not args.no_played else -1
            keep = list(range(len(legal)))
            if len(legal) > args.max_children:
                top = list(np.argsort(-prior)[: args.max_children // 2])
                rest = [k for k in range(len(legal)) if k not in set(top)]
                extra = list(rng.choice(rest, size=args.max_children - len(top), replace=False))
                keep = sorted(set(top) | set(extra) | ({legal.index(played)} if played in legal else set()))
            first_child = len(child_meta)
            games = []
            for k in keep:
                idx = tree.maybe_add_child(root, legal[k], None)
                game = tree.get_game_for_idx(idx)
                finished = bool(game.finished)
                if finished:
                    terminal_values.append(np.asarray(engine_rs.mcts_terminal_value(game, True), dtype=np.float32))
                else:
                    terminal_values.append(np.full(6, np.nan, dtype=np.float32))
                    games.append((len(child_meta), game))
                child_meta.append({"move": legal[k], "prior": float(prior[k]), "terminal": finished,
                                   "played": legal[k] == played})
            # Encode the non-terminal children (rotation = the child's mover) and the position itself.
            if games:
                for (ci, _), (row, rot) in zip(games, store.add([g for _, g in games])):
                    child_meta[ci]["rotation"] = rot
                    child_meta[ci]["enc"] = row
            (parent_row, _), = store.add([rs])
            seat = record["seats"][mover] if "seats" in record else "?"
            meta.append({
                "game": game_id, "action": i, "n_actions": n, "stage": STAGES[min(2, 3 * i // max(n, 1))],
                "step": step_type(rs, legal), "mover": mover, "side": seat, "legal": len(legal),
                "children": [first_child, len(child_meta)], "played": played, "enc": parent_row,
                "win": [float(x) for x in record["win_share"]],
            })
        LOGGER.info(
            f"{b + len(chunk)} / {len(positions)} positions, {len(child_meta)} children, {len(store.maps)} maps, "
            f"{time.time() - t0:.0f}s"
        )
    for key, array in {"terminal_values": np.stack(terminal_values), **store.arrays()}.items():
        np.save(out / f"{key}.npy", array)
    (out / "meta.json").write_text(json.dumps({"positions": meta, "children": child_meta, "args": vars(args)}))
    LOGGER.info(f"wrote {out}: {len(meta)} positions, {len(child_meta)} children")


class Store:
    """Encoded states, each distinct hex map once (``map_index``), with their legal moves."""

    def __init__(self):
        self.states, self.map_index, self.round_types, self.rotations, self.legal = [], [], [], [], []
        self.maps, self.map_ids = [], {}

    def add(self, games: list) -> list:
        """Encode engine ``games``; ``[(row, rotation)]``."""
        import hashlib

        import engine_rs

        enc = engine_rs.encode_games(games, False)
        out = []
        for j, game in enumerate(games):
            nf = np.ascontiguousarray(enc["node_features"][j], dtype=np.float32)
            key = hashlib.blake2b(nf.tobytes(), digest_size=16).digest()
            mi = self.map_ids.get(key)
            if mi is None:
                mi = self.map_ids[key] = len(self.maps)
                self.maps.append(nf)
            row = len(self.states)
            self.states.append(np.asarray(enc["game_states"][j], dtype=np.float32))
            self.map_index.append(mi)
            self.round_types.append(int(enc["round_types"][j]))
            self.rotations.append(int(enc["rotations"][j]))
            self.legal.append(np.asarray(game.factored_legal_indices(), dtype=np.int32))
            out.append((row, int(enc["rotations"][j])))
        return out

    def arrays(self) -> dict:
        return {
            "game_states": np.stack(self.states),
            "maps": np.stack(self.maps),
            "map_index": np.asarray(self.map_index, dtype=np.int64),
            "round_types": np.asarray(self.round_types, dtype=np.int64),
            "rotations": np.asarray(self.rotations, dtype=np.int64),
            "legal_flat": np.concatenate(self.legal),
            "legal_lengths": np.asarray([len(x) for x in self.legal], dtype=np.int64),
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


DATA_KEYS = ("terminal_values", "game_states", "maps", "map_index", "round_types", "rotations", "legal_flat",
             "legal_lengths")


def load_data(data_dir: Path, mmap: bool = True) -> dict:
    """A build set's arrays (``<key>.npy`` files, memory-mapped; or an older ``children.npz``)."""
    data_dir = Path(data_dir)
    if (data_dir / "children.npz").exists():
        return dict(np.load(data_dir / "children.npz"))
    return {k: np.load(data_dir / f"{k}.npy", mmap_mode="r" if mmap else None) for k in DATA_KEYS}


def child_values(model, data, device, batch_size: int = 256) -> np.ndarray:
    """(n_encoded, 4) win probabilities, canonical frame (the child's mover first)."""
    import torch
    import torch.nn.functional as F

    out = []
    gs_all, rt_all = data["game_states"], data["round_types"]
    with torch.no_grad():
        for b in range(0, len(gs_all), batch_size):
            gs = torch.from_numpy(gs_all[b : b + batch_size]).float().to(device)
            nodes = torch.from_numpy(data["maps"][data["map_index"][b : b + batch_size]]).float().to(device)
            rt = torch.from_numpy(rt_all[b : b + batch_size]).long().to(device)
            npl = torch.full((len(gs),), NUM_PLAYERS, dtype=torch.long, device=device)
            _, w, _, _ = model(gs, nodes, rt, None, npl, value_only=True)
            out.append(F.softmax(w.float(), -1)[:, :NUM_PLAYERS].cpu().numpy())
    return np.concatenate(out)


def mover_values(canon: np.ndarray, data, meta: dict) -> np.ndarray:
    """Per child, the parent mover's win probability (terminal children: their result)."""
    children = meta["children"]
    out = np.zeros(len(children), dtype=np.float64)
    for pos in meta["positions"]:
        m = pos["mover"]
        lo, hi = pos["children"]
        for c in range(lo, hi):
            ch = children[c]
            if ch["terminal"]:
                tv = data["terminal_values"][c][:NUM_PLAYERS]
                out[c] = float(tv[m] / max(tv.sum(), 1e-9))
            else:
                out[c] = float(canon[ch["enc"]][(m - ch["rotation"]) % NUM_PLAYERS])
    return out


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def position_metrics(values: dict, meta: dict, reference: str) -> list:
    children = meta["children"]
    rows = []
    for pos in meta["positions"]:
        lo, hi = pos["children"]
        if hi - lo < 2:
            continue
        ref = values[reference][lo:hi]
        played = [c - lo for c in range(lo, hi) if children[c]["played"]]
        prior = np.array([children[c]["prior"] for c in range(lo, hi)])
        row = {"stage": pos["stage"], "step": pos["step"], "side": pos["side"], "n": hi - lo,
               "game": pos["game"], "same_mover": [], "sources": {}}
        rotations = [children[c].get("rotation", -1) for c in range(lo, hi)]
        same = np.array([r == pos["mover"] for r in rotations])
        ref_top3 = set(np.argsort(-ref)[:3].tolist())
        for name, v in list(values.items()) + [("prior", None)]:
            x = prior if v is None else v[lo:hi]
            best = int(np.argmax(x))
            top3 = set(np.argsort(-x)[:3].tolist())
            row["sources"][name] = {
                "spearman": spearman(x, ref),
                "top1": float(best == int(np.argmax(ref))),
                "top3": len(top3 & ref_top3) / min(3, hi - lo),
                "regret": float(ref.max() - ref[best]),
                "spread": float(np.std(x)),
                "played_top1": float(best in played) if played else float("nan"),
                "diff_same_mover": float(np.mean((x - ref)[same])) if v is not None and same.any() else float("nan"),
                "diff_other_mover": float(np.mean((x - ref)[~same])) if v is not None and (~same).any() else float("nan"),
                "mixed_movers": bool(same.any() and (~same).any()),
                # Mean value of the children where the parent's mover moves again minus
                # where another player moves next (positions with both kinds).
                "mover_gap": (
                    float(np.mean(x[same]) - np.mean(x[~same])) if v is not None and same.any() and (~same).any()
                    else float("nan")
                ),
            }
        rows.append(row)
    return rows


METRICS = ("spearman", "top1", "top3", "regret", "spread", "played_top1", "diff_same_mover", "diff_other_mover",
           "mover_gap")


def summarize(rows: list, sources: list) -> dict:
    groups = {
        "all": rows,
        "mixed_movers": [r for r in rows if r["sources"]["prior"]["mixed_movers"]],
        # The moves az1's own search (policy + critic @200, side A of its eval collections) played.
        "az1_moves": [r for r in rows if r["side"] == "A" and r["game"].startswith("az1_")],
    }
    for s in STAGES:
        groups[f"stage:{s}"] = [r for r in rows if r["stage"] == s]
    for t in STEP_TYPES:
        groups[f"step:{t}"] = [r for r in rows if r["step"] == t]
    out = {}
    for g, rs in groups.items():
        if not rs:
            continue
        out[g] = {"positions": len(rs)}
        for name in sources + ["prior"]:
            stats = {}
            for key in METRICS:
                vals = np.array([r["sources"][name][key] for r in rs], dtype=np.float64)
                stats[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            out[g][name] = stats
    return out


def format_table(summary: dict, sources: list) -> str:
    lines = []
    cols = METRICS
    for g, d in summary.items():
        lines.append(f"\n== {g} ({d['positions']} positions)")
        lines.append(f"{'source':16s} " + " ".join(f"{c[-11:]:>11s}" for c in cols))
        for name in sources + ["prior"]:
            s = d[name]
            lines.append(f"{name:16s} " + " ".join(f"{s[c]:11.3f}" for c in cols))
    return "\n".join(lines)


def score(args) -> None:
    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value

    data_dir = Path(args.data)
    data = load_data(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    values, canon_all = {}, {}
    for item in args.source:
        name, _, path = item.partition("=")
        cache = data_dir / f"values.{name}.npy"
        if cache.exists() and not args.no_cache:
            canon = np.load(cache)
        else:
            model = load_policy_value(path)
            canon = child_values(model, data, model.device)
            np.save(cache, canon)
            del model
            torch.cuda.empty_cache()
        canon_all[name] = canon
        values[name] = mover_values(canon, data, meta)
        LOGGER.info(f"{name}: scored")
    names = [item.partition("=")[0] for item in args.source]
    rows = position_metrics(values, meta, args.reference)
    summary = summarize(rows, names)
    text = format_table(summary, names)
    print(text)
    report = {"reference": args.reference, "sources": dict(item.partition("=")[::2] for item in args.source),
              "summary": summary}
    (data_dir / f"report.{args.reference}.json").write_text(json.dumps(report, indent=2))
    (data_dir / f"report.{args.reference}.txt").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--collection", action="append", required=True)
    b.add_argument("--games", type=int, default=40, help="Games per collection")
    b.add_argument("--per-game", type=int, default=9)
    b.add_argument("--max-children", type=int, default=96)
    b.add_argument("--expander", default="model_checkpoints_search/az1_20261017/learner/280.pth"
                   "+model_checkpoints_search/az1_20261017/critic/280.pth")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-played", action="store_true", help="Skip finding the played move (training sets)")
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--source", action="append", required=True, help="name=<checkpoint>")
    s.add_argument("--reference", required=True)
    s.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("rl18xx.agent.alphazero.rust_mcts_player").setLevel(logging.WARNING)
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
