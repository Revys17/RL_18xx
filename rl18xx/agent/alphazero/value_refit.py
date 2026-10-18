"""Fit a policy network's own value heads to a critic, for one shared network (az2).

gen4's learner (``model_checkpoints_pg/rab_place_20261014/learner/1800.pth``)
never trained its value heads during policy gradient (a separate critic did),
so they are stale. Before the search reads values from the learner itself
(``search-learning --shared``) they are refit here with the trunk frozen -- the
policy is bit-for-bit unchanged -- on positions of saved search games
(``eval_head_to_head`` collections: ``games/<idx>.json`` action lists,
``games.jsonl`` outcomes), toward a mix of the critic's predictions and the
games' outcomes:

- win head: cross-entropy to ``(1 - a) * critic win probabilities + a * win share``;
- score head: squared error to ``(1 - a) * critic score + a * placement points``
  (gen4's critic regresses placement points 1 / 0.5 / 0.2 / 0 there),

``a`` = ``outcome_weight``. The trunk's per-seat value input ``[player token,
trunk]`` is computed once per position (:func:`extract`), so fitting the heads
is cheap. :func:`stage_report` scores any value source on held-out games the
way the az1 critic checks did: every 25th action after the 20th with more than
one legal move, win CE against the outcome and winner hit, by thirds of the game.
"""

from __future__ import annotations

import copy
import json
import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

NUM_PLAYERS = 4
STAGES = ("early", "mid", "late")


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@dataclass
class Position:
    game_state: np.ndarray  # canonical encoding (the mover first)
    node_features: np.ndarray
    rotation: int  # the mover's seat
    win: np.ndarray  # the game's win share, absolute seat order
    placement: np.ndarray  # placement points, absolute seat order
    stage: int  # 0 / 1 / 2: the action's third of the game
    game: str  # "<collection>/<game idx>"


def collection_games(directory) -> list:
    """``[(game id, action list, games.jsonl record)]`` of an eval collection."""
    directory = Path(directory)
    out = []
    for line in (directory / "games.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        path = directory / "games" / f"{record['game_idx']}.json"
        if not path.exists():
            continue
        actions = json.loads(path.read_text())
        if isinstance(actions, dict):  # a make_game_record export
            actions = actions.get("actions", [])
        out.append((f"{directory.name}/{record['game_idx']}", actions, record))
    return out


def game_positions(game_id: str, actions: list, record: dict, select, points=(1.0, 0.5, 0.2, 0.0)) -> list:
    """Replay a saved game from the empty board and keep the positions ``select(i,
    n, legal_count)`` picks (``i`` the action's index of ``n``) as :class:`Position` s."""
    from rl18xx.agent.alphazero.policy_gradient import placement_points
    from rl18xx.agent.alphazero.start_positions import new_game

    win = np.asarray(record["win_share"], dtype=np.float32)
    placement = placement_points(np.asarray(record["net_worth"], dtype=np.float64), points).astype(np.float32)
    g = new_game(NUM_PLAYERS)
    rs = g._game
    n = len(actions)
    out = []
    for i, action in enumerate(actions):
        if not g.finished and select(i, n, rs):
            state, nodes, _, rotation, _ = rs.encode_canonical_py()
            out.append(Position(state, nodes, int(rotation), win, placement, min(2, 3 * i // max(n, 1)), game_id))
        rs.process_action(action)
    return out


def heldout_selector(i: int, n: int, rs) -> bool:
    """The az1 critic checks' positions: every 25th action after the 20th, with a choice."""
    return i > 20 and i % 25 == 0 and len(rs.factored_legal_indices()) > 1


def sampled_selector(k: int, n_actions: int, rng: np.random.Generator):
    """About ``k`` decisions with a choice, uniformly over the game's actions."""
    chosen = set(rng.choice(n_actions, size=min(n_actions, 2 * k), replace=False).tolist()) if n_actions else set()
    kept = []

    def select(i, n, rs):
        if i in chosen and len(kept) < k and len(rs.factored_legal_indices()) > 1:
            kept.append(i)
            return True
        return False

    return select


def in_validation(game_id: str, fraction: float) -> bool:
    return (zlib.crc32(game_id.encode()) % 10_000) < fraction * 10_000


# ---------------------------------------------------------------------------
# Features and predictions
# ---------------------------------------------------------------------------


@dataclass
class Extracted:
    """Per position: the learner's value-head input (N, 4, d_entity + d_trunk) float16 and
    each value source's canonical-frame predictions, plus outcome targets (canonical frame)."""

    seat_input: Optional[np.ndarray]
    win: np.ndarray  # (N, 4) outcome win share, canonical
    placement: np.ndarray  # (N, 4) placement points, canonical
    stage: np.ndarray  # (N,)
    games: list
    predictions: dict = field(default_factory=dict)  # name -> {"win": (N, 4) probs, "score": (N, 4)}


def _canonical(values: np.ndarray, rotation: int) -> np.ndarray:
    return np.roll(values, -int(rotation))


def extract(positions: list, learner=None, sources: Optional[dict] = None, device=None, batch_size: int = 256) -> Extracted:
    """Run ``learner`` (keeping its value heads' input) and every model in ``sources``
    (name -> model; their win probabilities and score output) over ``positions``."""
    import torch
    import torch.nn.functional as F

    sources = dict(sources or {})
    if learner is not None:
        sources.setdefault("learner", learner)
    feats, preds = [], {name: {"win": [], "score": []} for name in sources}
    for b in range(0, len(positions), batch_size):
        chunk = positions[b : b + batch_size]
        gs = torch.from_numpy(np.stack([p.game_state for p in chunk])).float().to(device)
        nodes = torch.from_numpy(np.stack([p.node_features for p in chunk])).float().to(device)
        with torch.no_grad():
            for name, model in sources.items():
                store, handle = model.capture_value_input() if model is learner else ({}, None)
                try:
                    _, win_logits, score, _ = model(gs, nodes, value_only=True)
                finally:
                    if handle is not None:
                        handle.remove()
                preds[name]["win"].append(F.softmax(win_logits.float(), dim=-1)[:, :NUM_PLAYERS].cpu().numpy())
                preds[name]["score"].append(score.float()[:, :NUM_PLAYERS].cpu().numpy())
                if model is learner:
                    feats.append(store["seat_input"][:, :NUM_PLAYERS].to(torch.float16).cpu().numpy())
    return Extracted(
        seat_input=np.concatenate(feats) if feats else None,
        win=np.stack([_canonical(p.win, p.rotation) for p in positions]).astype(np.float32),
        placement=np.stack([_canonical(p.placement, p.rotation) for p in positions]).astype(np.float32),
        stage=np.array([p.stage for p in positions], dtype=np.int64),
        games=[p.game for p in positions],
        predictions={name: {k: np.concatenate(v) for k, v in d.items()} for name, d in preds.items()},
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


@dataclass
class RefitConfig:
    outcome_weight: float = 0.25
    score_weight: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 30
    patience: int = 4
    init: str = "stale"  # "stale": start from the learner's heads; "reset": fresh heads
    seed: int = 0


def head_losses(win_head, score_head, x, target_win, target_score):
    """Win CE (soft targets) and score MSE of per-seat heads on ``x`` (B, 4, F)."""
    import torch

    win_logits = win_head(x).squeeze(-1)
    score = score_head(x).squeeze(-1)
    ce = -(target_win * torch.log_softmax(win_logits, dim=1)).sum(dim=1).mean()
    mse = ((score - target_score) ** 2).mean()
    return ce, mse, win_logits, score


def fit_heads(learner, train: Extracted, val: Extracted, critic_name: str, cfg: RefitConfig, device=None) -> dict:
    """Fit copies of ``learner``'s win / score heads on the extracted features (trunk
    frozen), early-stopped on the validation loss; returns ``{"win_head", "score_head",
    "history"}`` (the best epoch's heads, on the CPU)."""
    import torch

    torch.manual_seed(cfg.seed)
    win_head = copy.deepcopy(learner.win_loss_head).float().to(device)
    score_head = copy.deepcopy(learner.score_head).float().to(device)
    if cfg.init == "reset":
        for head in (win_head, score_head):
            for m in head.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.LayerNorm):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.zeros_(head[-1].weight)
    params = list(win_head.parameters()) + list(score_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def targets(data: Extracted):
        a = cfg.outcome_weight
        crit = data.predictions[critic_name]
        return (
            torch.from_numpy((1 - a) * crit["win"] + a * data.win).float(),
            torch.from_numpy((1 - a) * crit["score"] + a * data.placement).float(),
        )

    tw, ts = targets(train)
    vw, vs = targets(val)
    xt = torch.from_numpy(train.seat_input)
    xv = torch.from_numpy(val.seat_input)

    def evaluate():
        win_head.eval(), score_head.eval()
        total, ce_sum, mse_sum, out_ce = 0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for b in range(0, len(xv), 4096):
                x = xv[b : b + 4096].float().to(device)
                ce, mse, logits, _ = head_losses(win_head, score_head, x, vw[b : b + 4096].to(device), vs[b : b + 4096].to(device))
                k = len(x)
                ce_sum += float(ce) * k
                mse_sum += float(mse) * k
                ow = torch.from_numpy(val.win[b : b + 4096]).to(device)
                out_ce += float(-(ow * torch.log_softmax(logits, dim=1)).sum(dim=1).sum())
                total += k
        return {"loss": (ce_sum + cfg.score_weight * mse_sum) / total, "ce": ce_sum / total,
                "mse": mse_sum / total, "outcome_ce": out_ce / total}

    best, best_state, history, bad = None, None, [], 0
    rng = np.random.default_rng(cfg.seed)
    history.append({"epoch": 0, **evaluate()})
    best = history[0]["loss"]
    best_state = (copy.deepcopy(win_head.state_dict()), copy.deepcopy(score_head.state_dict()))
    for epoch in range(1, cfg.max_epochs + 1):
        win_head.train(), score_head.train()
        order = rng.permutation(len(xt))
        for b in range(0, len(order), cfg.batch_size):
            idx = torch.from_numpy(order[b : b + cfg.batch_size])
            ce, mse, _, _ = head_losses(
                win_head, score_head, xt[idx].float().to(device), tw[idx].to(device), ts[idx].to(device)
            )
            opt.zero_grad(set_to_none=True)
            (ce + cfg.score_weight * mse).backward()
            opt.step()
        stats = {"epoch": epoch, **evaluate()}
        history.append(stats)
        LOGGER.info(f"refit epoch {epoch}: {stats}")
        if stats["loss"] < best - 1e-5:
            best, bad = stats["loss"], 0
            best_state = (copy.deepcopy(win_head.state_dict()), copy.deepcopy(score_head.state_dict()))
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    win_head.load_state_dict(best_state[0])
    score_head.load_state_dict(best_state[1])
    return {"win_head": win_head.cpu().eval(), "score_head": score_head.cpu().eval(), "history": history}


def head_predictions(win_head, score_head, data: Extracted, device=None) -> dict:
    import torch

    win, score = [], []
    win_head, score_head = win_head.to(device), score_head.to(device)
    with torch.no_grad():
        for b in range(0, len(data.seat_input), 4096):
            x = torch.from_numpy(data.seat_input[b : b + 4096]).float().to(device)
            win.append(torch.softmax(win_head(x).squeeze(-1), dim=1).cpu().numpy())
            score.append(score_head(x).squeeze(-1).cpu().numpy())
    return {"win": np.concatenate(win), "score": np.concatenate(score)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def stage_report(data: Extracted, prediction: dict) -> dict:
    """Win CE against the outcome and winner hit (the argmax seat won), overall and per
    third of the game; score MSE against placement points."""
    p = np.clip(prediction["win"] / prediction["win"].sum(axis=1, keepdims=True), 1e-9, 1.0)
    ce = -(data.win * np.log(p)).sum(axis=1)
    hit = (np.take_along_axis(data.win, p.argmax(axis=1)[:, None], axis=1)[:, 0] > 0).astype(np.float64)
    mse = ((prediction["score"] - data.placement) ** 2).mean(axis=1)
    out = {"n": int(len(ce)), "win_ce": float(ce.mean()), "winner_hit": float(hit.mean()), "score_mse": float(mse.mean())}
    for s, name in enumerate(STAGES):
        m = data.stage == s
        out[f"win_ce_{name}"] = float(ce[m].mean()) if m.any() else float("nan")
        out[f"winner_hit_{name}"] = float(hit[m].mean()) if m.any() else float("nan")
        out[f"n_{name}"] = int(m.sum())
    return out


def format_report(name: str, r: dict) -> str:
    return (
        f"{name:16s} CE {r['win_ce']:.3f}  winner {r['winner_hit']:.3f}  score MSE {r['score_mse']:.4f} | "
        f"by stage (early/mid/late) CE {r['win_ce_early']:.3f} / {r['win_ce_mid']:.3f} / {r['win_ce_late']:.3f}"
        f"  winner {r['winner_hit_early']:.3f} / {r['winner_hit_mid']:.3f} / {r['winner_hit_late']:.3f}"
    )


def write_refit_checkpoint(learner_path, win_head, score_head, out_path, provenance: dict) -> Path:
    """The learner at ``learner_path`` with these value heads, saved at ``out_path``
    (``<run>/learner/<n>.pth``) with ``<run>/config.json`` carrying the source run's
    ``learner_temperature`` (so search learning reads T from it) and ``provenance``."""
    from rl18xx.agent.alphazero.policy_gradient import _load, _save
    from rl18xx.agent.alphazero.search_learning import run_temperature

    out_path = Path(out_path)
    model = _load(learner_path)
    model.win_loss_head.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in win_head.state_dict().items()})
    model.score_head.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in score_head.state_dict().items()})
    _save(model, out_path)
    config = {"learner_temperature": run_temperature(learner_path), "source": str(learner_path), **provenance}
    if out_path.parent.name == "learner":
        (out_path.parent.parent / "config.json").write_text(json.dumps(config, indent=2))
    return out_path


def collect(directories: Iterable, select_factory, exclude: Iterable = ()) -> list:
    """Positions of every game in ``directories`` (``select_factory(game_id, n_actions)``
    gives each game's selector), skipping ``exclude`` collection names."""
    exclude = {Path(e).name for e in exclude}
    out = []
    for directory in directories:
        if Path(directory).name in exclude:
            continue
        games = collection_games(directory)
        for game_id, actions, record in games:
            try:
                out.extend(game_positions(game_id, actions, record, select_factory(game_id, len(actions))))
            except Exception as e:  # a game recorded before an engine change
                LOGGER.warning(f"{game_id}: skipped ({e!r})")
        LOGGER.info(f"{directory}: {len(games)} games, {len(out)} positions so far")
    return out
