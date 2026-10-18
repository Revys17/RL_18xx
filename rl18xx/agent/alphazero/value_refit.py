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
    legal: Optional[np.ndarray] = None  # int32 policy indices (the trunk fine-tune's policy anchor)


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
            legal = np.asarray(rs.factored_legal_indices(), dtype=np.int32)
            out.append(Position(state, nodes, int(rotation), win, placement, min(2, 3 * i // max(n, 1)), game_id, legal))
        rs.process_action(action)
    return out


def heldout_selector(i: int, n: int, rs) -> bool:
    """The az1 critic checks' positions: every 25th action after the 20th, with a choice."""
    return i > 20 and i % 25 == 0 and len(rs.factored_legal_indices()) > 1


def sampled_selector(k: int, n_actions: int, rng: np.random.Generator):
    """``k`` decisions with a choice, uniformly over the game's actions: 2k candidate
    actions are drawn, those with a choice kept, and :func:`collect` thins them to k
    at random (keeping the first k in game order would skew toward the opening)."""
    chosen = set(rng.choice(n_actions, size=min(n_actions, 2 * k), replace=False).tolist()) if n_actions else set()

    def select(i, n, rs):
        return i in chosen and len(rs.factored_legal_indices()) > 1

    select.keep, select.rng = k, rng
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
    # On the GPU the features fit (~0.9 GB for 165k positions in fp16): gathering
    # batches there instead of on the host is ~20x faster.
    xt = torch.from_numpy(train.seat_input).to(device)
    xv = torch.from_numpy(val.seat_input).to(device)
    tw, ts, vw, vs = (t.to(device) for t in (tw, ts, vw, vs))

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
            idx = torch.from_numpy(order[b : b + cfg.batch_size]).to(device)
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


# ---------------------------------------------------------------------------
# Trunk fine-tune (policy anchored)
# ---------------------------------------------------------------------------


@dataclass
class FinetuneConfig:
    """The whole network on the value targets, with KL(original || tuned) over each
    position's legal moves at ``temperature`` (and over every price slot's cells)
    keeping the policy where it was."""

    outcome_weight: float = 0.25
    score_weight: float = 1.0
    policy_kl_weight: float = 1.0
    trunk_lr: float = 1e-5
    head_lr: float = 1e-4
    batch_size: int = 256
    epochs: int = 2
    eval_every: int = 200
    temperature: float = 0.5
    seed: int = 0


def _batch_tensors(positions: list, idx, device) -> tuple:
    import torch

    from rl18xx.agent.alphazero.mcts import POLICY_SIZE
    from rl18xx.agent.alphazero.policy_gradient import _lay_tile_range

    chunk = [positions[int(i)] for i in idx]
    gs = torch.from_numpy(np.stack([p.game_state for p in chunk])).float().to(device)
    nodes = torch.from_numpy(np.stack([p.node_features for p in chunk])).float().to(device)
    legal = torch.zeros(len(chunk), POLICY_SIZE, dtype=torch.bool)
    for b, p in enumerate(chunk):
        legal[b, torch.from_numpy(p.legal.astype(np.int64))] = True
    lo, hi = _lay_tile_range()
    tiles = legal[:, lo:hi].any(dim=1)
    return gs, nodes, legal.to(device), tiles


def policy_drift(original, tuned, positions: list, temperature: float, device=None, batch_size: int = 256) -> dict:
    """KL(original || tuned) over the legal moves at ``temperature`` (mean per
    position) and how often the two policies' top moves agree."""
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _masked_log_softmax

    kl, agree = [], []
    with torch.no_grad():
        for b in range(0, len(positions), batch_size):
            idx = range(b, min(b + batch_size, len(positions)))
            gs, nodes, legal, tiles = _batch_tensors(positions, idx, device)
            lp0 = _masked_log_softmax(original(gs, nodes, lay_tile_rows=tiles)[0] / temperature, legal)
            lp1 = _masked_log_softmax(tuned(gs, nodes, lay_tile_rows=tiles)[0] / temperature, legal)
            p0 = lp0.exp() * legal
            kl.append((p0 * (lp0 - lp1).masked_fill(~legal, 0.0)).sum(dim=1).cpu().numpy())
            agree.append(
                (lp0.masked_fill(~legal, -1e30).argmax(1) == lp1.masked_fill(~legal, -1e30).argmax(1)).float().cpu().numpy()
            )
    kl, agree = np.concatenate(kl), np.concatenate(agree)
    return {"policy_kl": float(kl.mean()), "policy_kl_p99": float(np.quantile(kl, 0.99)), "top1_agreement": float(agree.mean())}


def model_predictions(model, positions: list, device=None, batch_size: int = 256) -> dict:
    """A whole network's win probabilities and score output (canonical, 4 seats)."""
    import torch
    import torch.nn.functional as F

    win, score = [], []
    with torch.no_grad():
        for b in range(0, len(positions), batch_size):
            chunk = positions[b : b + batch_size]
            gs = torch.from_numpy(np.stack([p.game_state for p in chunk])).float().to(device)
            nodes = torch.from_numpy(np.stack([p.node_features for p in chunk])).float().to(device)
            _, w, s, _ = model(gs, nodes, value_only=True)
            win.append(F.softmax(w.float(), dim=-1)[:, :NUM_PLAYERS].cpu().numpy())
            score.append(s.float()[:, :NUM_PLAYERS].cpu().numpy())
    return {"win": np.concatenate(win), "score": np.concatenate(score)}


def finetune(learner, positions: list, data: Extracted, val_positions: list, val: Extracted, critic_name: str,
             cfg: FinetuneConfig, device=None, win_head=None, score_head=None) -> dict:
    """Fine-tune a copy of ``learner`` -- trunk and value heads, starting from
    ``win_head`` / ``score_head`` if given (a heads-only fit) -- toward ``(1 - a) *
    critic + a * outcome`` on ``positions`` (``data`` holds their targets in the same
    order), with ``policy_kl_weight`` * KL(learner || tuned) on the policy at T and on
    the price head; the best validation outcome CE wins. Returns ``{"model",
    "history"}``."""
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _masked_log_softmax

    torch.manual_seed(cfg.seed)
    original = learner.eval()
    tuned = copy.deepcopy(learner)
    if win_head is not None:
        tuned.win_loss_head.load_state_dict(win_head.state_dict())
    if score_head is not None:
        tuned.score_head.load_state_dict(score_head.state_dict())
    tuned.to(device).eval()  # eval: no dropout, the policy trained is the one played
    heads = {id(p) for h in (tuned.win_loss_head, tuned.score_head) for p in h.parameters()}
    groups = [
        {"params": [p for p in tuned.parameters() if id(p) not in heads], "lr": cfg.trunk_lr},
        {"params": [p for p in tuned.parameters() if id(p) in heads], "lr": cfg.head_lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    a = cfg.outcome_weight
    crit = data.predictions[critic_name]
    target_win = torch.from_numpy((1 - a) * crit["win"] + a * data.win).float()
    target_score = torch.from_numpy((1 - a) * crit["score"] + a * data.placement).float()
    T = cfg.temperature

    def evaluate(step):
        pred = model_predictions(tuned, val_positions, device)
        report = stage_report(val, pred)
        drift = policy_drift(original, tuned, val_positions[:4096], T, device)
        return {"step": step, "outcome_ce": report["win_ce"], "winner_hit": report["winner_hit"], **drift}

    history = [evaluate(0)]
    best = (history[0]["outcome_ce"], copy.deepcopy(tuned.state_dict()), 0)
    rng = np.random.default_rng(cfg.seed)
    step = 0
    for _ in range(cfg.epochs):
        order = rng.permutation(len(positions))
        for b in range(0, len(order), cfg.batch_size):
            idx = order[b : b + cfg.batch_size]
            gs, nodes, legal, tiles = _batch_tensors(positions, idx, device)
            with torch.no_grad():
                logits0, _, _, _ = original(gs, nodes, lay_tile_rows=tiles)
                price0 = original.last_price_components["price_logits"]
            logits, win_logits, score, _ = tuned(gs, nodes, lay_tile_rows=tiles)
            price = tuned.last_price_components["price_logits"]
            tw, ts = target_win[idx].to(device), target_score[idx].to(device)
            ce = -(tw * torch.log_softmax(win_logits.float()[:, :NUM_PLAYERS], dim=1)).sum(1).mean()
            mse = ((score.float()[:, :NUM_PLAYERS] - ts) ** 2).mean()
            lp0 = _masked_log_softmax(logits0 / T, legal)
            lp1 = _masked_log_softmax(logits / T, legal)
            kl = ((lp0.exp() * legal) * (lp0 - lp1).masked_fill(~legal, 0.0)).sum(1).mean()
            c0, c1 = torch.log_softmax(price0.float(), -1), torch.log_softmax(price.float(), -1)
            price_kl = (c0.exp() * (c0 - c1)).sum(-1).mean()
            loss = ce + cfg.score_weight * mse + cfg.policy_kl_weight * (kl + price_kl)
            if not torch.isfinite(loss):
                raise RuntimeError("value fine-tune diverged (non-finite loss)")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for group in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            opt.step()
            step += 1
            if step % cfg.eval_every == 0:
                history.append(evaluate(step))
                LOGGER.info(
                    f"fine-tune step {step}: {history[-1]} (train ce {float(ce.detach()):.4f}, kl {float(kl.detach()):.5f})"
                )
                if history[-1]["outcome_ce"] < best[0]:
                    best = (history[-1]["outcome_ce"], copy.deepcopy(tuned.state_dict()), step)
    if step % cfg.eval_every:
        history.append(evaluate(step))
        if history[-1]["outcome_ce"] < best[0]:
            best = (history[-1]["outcome_ce"], copy.deepcopy(tuned.state_dict()), step)
    tuned.load_state_dict(best[1])
    return {"model": tuned.eval(), "history": history, "best_step": best[2]}


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
    from rl18xx.agent.alphazero.policy_gradient import _load

    model = _load(learner_path)
    model.win_loss_head.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in win_head.state_dict().items()})
    model.score_head.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in score_head.state_dict().items()})
    return write_model_checkpoint(model, learner_path, out_path, provenance)


def write_model_checkpoint(model, learner_path, out_path, provenance: dict) -> Path:
    """``model`` saved at ``out_path`` with ``<run>/config.json`` carrying
    ``learner_path``'s run's ``learner_temperature`` and ``provenance``."""
    from rl18xx.agent.alphazero.policy_gradient import _save
    from rl18xx.agent.alphazero.search_learning import run_temperature

    out_path = Path(out_path)
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
            select = select_factory(game_id, len(actions))
            try:
                positions = game_positions(game_id, actions, record, select)
            except Exception as e:  # a game recorded before an engine change
                LOGGER.warning(f"{game_id}: skipped ({e!r})")
                continue
            keep = getattr(select, "keep", None)
            if keep is not None and len(positions) > keep:
                picked = np.sort(select.rng.choice(len(positions), size=keep, replace=False))
                positions = [positions[int(i)] for i in picked]
            out.extend(positions)
        LOGGER.info(f"{directory}: {len(games)} games, {len(out)} positions so far")
    return out
