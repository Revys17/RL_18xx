"""Supervised pretraining of the title-generic model (RL bridge milestone C1; ``docs/rl_model.md``
"Supervised data and pretraining").

Mirrors the legacy 1830 recipe (``pretraining.pretrain_model`` + ``_refit_value_heads``, the defaults
of ``main.py pretrain``):

* **Stage 1, joint run**: CE of the policy to the human move over the position's legal moves with
  label smoothing 0.03 over the legal set, an entropy bonus 0.01, the price NLL (weight 0.1), the win
  CE (weight 0.1) and the score MSE on placement points (weight 0.1); AdamW (lr 1e-3, weight decay
  1e-4 on matrices only, as ``train.build_optimizer``), cosine LR to 0 over the planned steps, gradient
  clip 1.0, batch 256; bf16 autocast on CUDA, fp32 evaluation; a validation pass per epoch (or every
  ``eval_every`` steps), the best validation loss kept (``best_joint.pth``) and early stopping after
  ``patience`` evaluations without improvement. With the value refit on, "best" is the joint loss without
  its value terms (``select_metric``): the value heads memorize the games long before the policy stops
  improving, and on the smaller titles the total picked a checkpoint 1-4 epochs early (1822CA top-1 .619
  selected vs .673 at the last epoch, docs/training_runs_2026_11.md "The new titles' SL").
* **Stage 2, value refit**: the best joint checkpoint's value heads re-initialized and trained alone on
  the frozen network (``value_stop_grad``), validated every ``refit_eval_steps`` steps on the win CE,
  the best heads kept (``refit.pth``), because with a few thousand games the value head memorizes
  games within about an epoch while the policy keeps improving.

Logged (TensorBoard under ``<out>/tb`` and ``<out>/history.jsonl``): losses, top-1 / top-5 over the
legal moves, top-1 per descriptor family (Par / BuyTrain.* / LayTile / ... separately) and per player
count, price NLL, value CE against the equal-odds baseline ln(n) and the winner accuracy (the argmax
seat is a winner). Checkpoints go through ``model_generic.save_checkpoint`` (titles, layout
fingerprints, encoder version; the training config and metrics in ``extra``).

:func:`compare_with_legacy` scores a generic checkpoint and a legacy 1830 checkpoint (e.g. SL ckpt 10)
on the same positions of a dataset split: per-family top-1, policy CE and value CE / winner accuracy.

Several datasets (milestone C3, ``docs/rl_model.md`` "Mixed-title training and the sharing probe"): ``data`` as
a list of ``path`` / ``label=path``, mixed per batch at ``mix`` shares by the stream loader
(``generic_data.MixedPositionBatches``), split overrides (``splits``), per-dataset evaluations, selection and
early stopping on one dataset (``select_title`` / ``select_split``), a final per-dataset evaluation
(``final_eval``); ``init`` starts from another run's checkpoint through :func:`load_init`'s checks.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from rl18xx.agent.alphazero import generic_data as gd
from rl18xx.agent.alphazero import model_generic as mg

LOGGER = logging.getLogger(__name__)

#: Small model config for CPU smoke runs and tests (~150k parameters).
SMALL_CONFIG = dict(
    d_model=32,
    num_heads=4,
    hex_layers=1,
    entity_layers=1,
    fusion_layers=1,
    d_trunk=64,
    trunk_blocks=1,
    d_head=32,
    d_edge=8,
    d_cond=16,
    d_id=8,
)
MODEL_PRESETS = {"default": {}, "small": SMALL_CONFIG}


@dataclass
class GenericPretrainConfig:
    #: One dataset directory, or several (a mixed-title run): each ``path`` or ``label=path`` (the label
    #: names the dataset in ``mix``, ``splits``, the per-dataset options and the metrics; default its title).
    data: Union[str, List[str]]
    out: str
    #: ``GenericModelConfig`` overrides (a preset name's dict, plus any fields).
    model: dict = field(default_factory=dict)
    #: Warm start from a generic checkpoint (else a fresh model).
    init: Optional[str] = None
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_lr_multiplier: float = 1.0
    epochs: int = 10
    #: Evaluations without improvement before stopping (legacy ``pretrain_early_stop_patience``).
    patience: int = 4
    label_smoothing: float = 0.03
    entropy_weight: float = 0.01
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.1
    score_loss_weight: float = 0.1
    price_loss_weight: float = 0.1
    grad_clip: float = 1.0
    #: Epochs during which value gradients reach the trunk (None: joint throughout, legacy default).
    value_joint_epochs: Optional[int] = None
    value_refit: bool = True
    refit_eval_steps: int = 250
    refit_patience: int = 4
    refit_max_steps: int = 20000
    device: str = "auto"
    #: "auto" (bf16 autocast on CUDA only), "on", "off".
    bf16: str = "auto"
    workers: int = 4
    buffer_positions: int = 32768
    seed: int = 0
    #: Stage 1 step cap (smoke runs); None = epochs x steps per epoch.
    max_steps: Optional[int] = None
    #: Validate every N steps (None: after each epoch; with the stream loader, every epoch-equivalent).
    eval_every: Optional[int] = None
    #: The first N games of the training / evaluated split: an int for every dataset or ``{label: N}``.
    train_max_games: Optional[Union[int, Dict[str, int]]] = None
    val_max_games: Optional[Union[int, Dict[str, int]]] = None
    log_every: int = 50
    threads: int = 4
    # -- several datasets (mixed titles), split overrides, selection (docs/rl_model.md "Mixed-title training") --
    #: ``{label: share}`` of the training positions (normalized; 0 = evaluated only). None: each dataset's
    #: share is its number of training positions (the datasets concatenated).
    mix: Optional[Dict[str, float]] = None
    #: ``{label: split override}`` (``generic_data.load_splits``: a ``{split: [game ids]}`` dict or JSON file).
    splits: Optional[Dict[str, object]] = None
    #: The label whose ``select_split`` total picks the best checkpoint and drives early stopping (and whose
    #: win CE drives the value refit). None: the one dataset, or the mix-weighted mean over ``eval_titles``.
    select_title: Optional[str] = None
    select_split: str = "validation"
    #: What picks the best joint checkpoint and drives early stopping: "policy" = the joint loss without its
    #: value terms (smoothed policy CE, entropy bonus, price NLL: what stays after the value refit), "total" =
    #: the whole joint loss, "auto" = policy with the value refit on, else total.
    select_metric: str = "auto"
    #: Labels evaluated during training (None: every dataset).
    eval_titles: Optional[List[str]] = None
    #: After training, evaluate ``final.pth`` on every dataset's ``report_split`` (``summary["final_eval"]``).
    final_eval: bool = False
    report_split: str = "validation"
    report_max_games: Optional[Union[int, Dict[str, int]]] = None
    #: "epochs" (one dataset, a pass per epoch: C1's loop), "stream" (``MixedPositionBatches``: any number of
    #: datasets, endless, step-based) or "auto" (stream when there are several datasets or a ``mix``).
    loader: str = "auto"


# ---------------------------------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------------------------------


def segment_rank(logp: Tensor, pos: Tensor, offsets: Tensor, target: Tensor) -> Tensor:
    """(B,) rank of each position's target among its legal rows: rows scoring higher, plus equal rows
    earlier in the list (argmax's first-index tie-break). Top-1 = rank 0."""
    b = offsets.shape[0] - 1
    flat = offsets[:-1] + target
    t = logp[flat][pos]
    rows = torch.arange(logp.shape[0], device=logp.device)
    ahead = (logp > t) | ((logp == t) & (rows < flat[pos]))
    return logp.new_zeros(b).index_add(0, pos, ahead.float())


def compute_losses(out: dict, b: dict, cfg: GenericPretrainConfig) -> Dict[str, Tensor]:
    """The training objective's parts (per-batch means) for a collated batch ``b`` on the device."""
    inputs = b["inputs"]
    pos, offsets = inputs["legal_pos"], inputs["legal_offsets"]
    nb = offsets.shape[0] - 1
    logp = out["logp"]
    counts = (offsets[1:] - offsets[:-1]).clamp(min=1).float()
    nll = -logp[offsets[:-1] + b["target_index"]]
    mean_neg = logp.new_zeros(nb).index_add(0, pos, -logp) / counts
    eps = cfg.label_smoothing
    policy = ((1.0 - eps) * nll + eps * mean_neg).mean()
    entropy = mg.policy_entropy(out, inputs).mean()
    price = mg.price_loss(out, b["price_cell"], b["price_log_count"], b["price_valid"])
    v = mg.value_losses(out, b["win"], b["score"])
    total = (
        cfg.policy_loss_weight * policy
        + cfg.value_loss_weight * v["ce"]
        + cfg.score_loss_weight * v["mse"]
        + cfg.price_loss_weight * price
        - cfg.entropy_weight * entropy
    )
    return {
        "total": total,
        "policy": policy,
        "policy_ce": nll.mean(),
        "entropy": entropy,
        "price": price,
        "value": v["ce"],
        "score": v["mse"],
    }


class MetricSums:
    """Sums over batches: losses, ranks, per-family and per-player-count top-1, value CE / baseline /
    winner hits, price NLL."""

    def __init__(self, families):
        self.families = families
        self.n = 0
        self.sums = defaultdict(float)
        self.fam_hits = np.zeros(len(families))
        self.fam_n = np.zeros(len(families))
        self.players = defaultdict(lambda: np.zeros(5))  # hits, n, value ce, baseline, winner hits
        self.price_nll = 0.0
        self.price_n = 0

    @torch.no_grad()
    def add(self, out: dict, b: dict, losses: Optional[dict] = None, cfg: Optional[GenericPretrainConfig] = None):
        inputs = b["inputs"]
        nb = int(b["target_index"].shape[0])
        self.n += nb
        if losses is not None:
            for k, v in losses.items():
                self.sums[k] += float(v) * nb
        ok_value = out.get("win_logits") is not None
        if out.get("logp") is not None:
            logp = out["logp"].float()
            self.sums["policy_nll"] += float(-logp[inputs["legal_offsets"][:-1] + b["target_index"]].sum())
            rank = segment_rank(logp, inputs["legal_pos"], inputs["legal_offsets"], b["target_index"])
            top1 = (rank == 0).cpu().numpy()
            self.sums["top1"] += float(top1.sum())
            self.sums["top5"] += float((rank < 5).sum())
            # a uniform policy over the legal moves: its expected top-1 (forced positions count 1)
            counts = (inputs["legal_offsets"][1:] - inputs["legal_offsets"][:-1]).float()
            self.sums["chance_top1"] += float((1.0 / counts.clamp(min=1)).sum())
            fam = b["family"].cpu().numpy()
            np.add.at(self.fam_hits, fam, top1)
            np.add.at(self.fam_n, fam, 1)
            vp = out["price_logp"]
            if vp is not None and vp.shape[0]:
                valid = b["price_valid"]
                if bool(valid.any()):
                    nll = -(vp.gather(1, b["price_cell"][:, None]).squeeze(1) - b["price_log_count"])
                    self.price_nll += float(nll[valid].sum())
                    self.price_n += int(valid.sum())
        else:
            top1 = None
        if ok_value:
            present = out["present"]
            log_v = torch.log_softmax(out["win_logits"].float(), dim=1)
            ce = -(b["win"] * log_v * present.float()).sum(1)
            n_players = b["num_players"]
            base = torch.log(n_players.float())
            hit = (b["win"].gather(1, log_v.argmax(1, keepdim=True)).squeeze(1) > 0).float()
            mse = (((out["score"].float() - b["score"]) ** 2) * present.float()).sum(1) / present.float().sum(1)
            self.sums["value_ce"] += float(ce.sum())
            self.sums["value_baseline"] += float(base.sum())
            self.sums["winner_hits"] += float(hit.sum())
            self.sums["score_mse"] += float(mse.sum())
            npl = n_players.cpu().numpy()
            for k, p in enumerate(npl):
                row = self.players[int(p)]
                row[1] += 1
                row[2] += float(ce[k])
                row[3] += float(base[k])
                row[4] += float(hit[k])
                if top1 is not None:
                    row[0] += float(top1[k])

    def result(self) -> dict:
        n = max(self.n, 1)
        out = {"positions": self.n}
        for k, v in self.sums.items():
            if k in ("top1", "top5"):
                out[k] = v / n
            elif k == "winner_hits":
                out["winner_acc"] = v / n
            else:
                out[k] = v / n
        if self.price_n:
            out["price_nll"] = self.price_nll / self.price_n
            out["price_targets"] = self.price_n
        out["top1_by_family"] = {
            self.families[f]: {"top1": float(self.fam_hits[f] / self.fam_n[f]), "n": int(self.fam_n[f])}
            for f in np.flatnonzero(self.fam_n)
        }
        out["by_players"] = {
            str(p): {
                "n": int(r[1]),
                "top1": float(r[0] / r[1]),
                "value_ce": float(r[2] / r[1]),
                "value_baseline": float(r[3] / r[1]),
                "winner_acc": float(r[4] / r[1]),
            }
            for p, r in sorted(self.players.items())
            if r[1]
        }
        return out


# ---------------------------------------------------------------------------------------------------
# Loaders, optimizer, evaluation
# ---------------------------------------------------------------------------------------------------


def make_loader(
    data: str,
    split: str,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int = 0,
    epoch: int = 0,
    max_games: Optional[int] = None,
    buffer_positions: int = 32768,
    pin_memory: bool = False,
    config=None,
    splits=None,
):
    stream = gd.PositionBatches(
        data,
        split,
        batch_size,
        shuffle=shuffle,
        buffer_positions=buffer_positions,
        seed=seed,
        epoch=epoch,
        max_games=max_games,
        config=config,
        splits=splits,
    )
    return _data_loader(stream, workers, pin_memory)


def make_mixed_loader(
    sources, batch_size: int, workers: int, seed: int = 0, buffer_positions: int = 32768, pin_memory: bool = False,
    config=None,
):
    """An endless ``DataLoader`` over :class:`generic_data.MixedPositionBatches` (several datasets mixed)."""
    stream = gd.MixedPositionBatches(
        sources, batch_size, buffer_positions=buffer_positions, seed=seed, config=config
    )
    return _data_loader(stream, workers, pin_memory)


def _data_loader(stream, workers: int, pin_memory: bool):
    kwargs = dict(batch_size=None, num_workers=workers, pin_memory=pin_memory)
    if workers > 0:
        kwargs.update(worker_init_fn=gd.loader_worker_init, prefetch_factor=4, multiprocessing_context="spawn")
    return torch.utils.data.DataLoader(stream, **kwargs)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, value_lr_multiplier: float = 1.0):
    """AdamW as ``train.build_optimizer``: decay on matrices only (no LayerNorm / embedding / bias
    decay), the value heads at ``lr * value_lr_multiplier``."""
    no_decay = set()
    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.Embedding)):
            no_decay.update(id(p) for p in module.parameters(recurse=False))
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_value = name.startswith("win_head") or name.startswith("score_head")
        decay = p.ndim >= 2 and id(p) not in no_decay
        groups[(is_value, decay)].append(p)
    param_groups = [
        {
            "params": params,
            "lr": lr * (value_lr_multiplier if is_value else 1.0),
            "weight_decay": weight_decay if decay else 0.0,
        }
        for (is_value, decay), params in sorted(groups.items())
    ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def reset_value_heads(model: mg.GenericPolicyValueModel) -> None:
    """Fresh win / score heads (zero-initialized outputs: equal odds), as the model builds them."""
    device = next(model.parameters()).device
    model.win_head = model._value_mlp().to(device)
    model.score_head = model._value_mlp().to(device)
    for head in (model.win_head, model.score_head):
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def evaluate(
    model: mg.GenericPolicyValueModel,
    data: str,
    split: str = "validation",
    batch_size: int = 256,
    device: torch.device | str = "cpu",
    workers: int = 0,
    max_games: Optional[int] = None,
    cfg: Optional[GenericPretrainConfig] = None,
    value_only: bool = False,
    splits=None,
) -> dict:
    """Metrics of ``model`` on a split, fp32, positions in a fixed order (:class:`MetricSums`)."""
    device = torch.device(device)
    was_training = model.training
    model.eval()
    sums = MetricSums(mg.vocab()["families"])
    loader = make_loader(
        data, split, batch_size, False, workers, max_games=max_games, config=model.config, splits=splits
    )
    for batch in loader:
        b = gd.batch_to_device(batch, device)
        out = model(b["inputs"], value_only=value_only)
        losses = compute_losses(out, b, cfg) if (cfg is not None and not value_only) else None
        sums.add(out, b, losses)
    model.train(was_training)
    return sums.result()


class _Log:
    def __init__(self, out: Path):
        from torch.utils.tensorboard import SummaryWriter

        self.out = out
        self.writer = SummaryWriter(str(out / "tb"))
        self.history = open(out / "history.jsonl", "a")

    def write(self, entry: dict, prefix: str, step: int) -> None:
        self.history.write(json.dumps(entry) + "\n")
        self.history.flush()
        for k, v in entry.items():
            if isinstance(v, (int, float)) and k not in ("step", "epoch"):
                self.writer.add_scalar(f"{prefix}/{k}", v, step)
        for fam, r in (entry.get("top1_by_family") or {}).items():
            self.writer.add_scalar(f"{prefix}_family_top1/{fam}", r["top1"], step)
        for p, r in (entry.get("by_players") or {}).items():
            for k in ("top1", "value_ce", "winner_acc"):
                self.writer.add_scalar(f"{prefix}_{p}p/{k}", r[k], step)

    def close(self):
        self.writer.close()
        self.history.close()


def _summary_line(m: dict) -> str:
    fams = m.get("top1_by_family", {})
    pick = " ".join(
        f"{k}={fams[k]['top1']:.3f}"
        for k in ("Par", "BuyShares", "LayTile", "BuyTrain.depot", "Bid.company")
        if k in fams
    )
    return (
        f"positions={m.get('positions')} total={m.get('total', float('nan')):.4f} "
        + (f"(policy part {m['policy_total']:.4f}) " if "policy_total" in m else "")
        + f"policy_ce={m.get('policy_ce', float('nan')):.4f} top1={m.get('top1', float('nan')):.3f} "
        f"top5={m.get('top5', float('nan')):.3f} (chance top1 {m.get('chance_top1', float('nan')):.3f}) "
        f"price={m.get('price_nll', float('nan')):.3f} "
        f"value={m.get('value_ce', float('nan')):.4f} (equal odds {m.get('value_baseline', float('nan')):.4f}, "
        f"winner {m.get('winner_acc', float('nan')):.3f}) | {pick}"
    )


# ---------------------------------------------------------------------------------------------------
# Datasets of a run, and initialization from another run's checkpoint
# ---------------------------------------------------------------------------------------------------


def _per_label(value, label: str, labels: List[str], name: str):
    """An option given for every dataset (a scalar) or per dataset (``{label: value}``)."""
    if isinstance(value, dict):
        unknown = sorted(set(value) - set(labels))
        if unknown:
            raise ValueError(f"{name}: unknown dataset labels {unknown} (have {labels})")
        return value.get(label)
    return value


def _split_entry(entry) -> tuple:
    """``label=path`` -> (label, path); a bare path -> (None, path)."""
    text = str(entry)
    head, sep, tail = text.partition("=")
    if sep and head and "/" not in head and os.sep not in head:
        return head, tail
    return None, text


def resolve_sources(cfg: GenericPretrainConfig) -> List[gd.DataSource]:
    """The run's datasets as :class:`generic_data.DataSource` s: labels (``label=path``, else the dataset's
    title), split overrides, game caps and mix weights (``cfg.mix``, else each dataset's training positions)."""
    entries = [cfg.data] if isinstance(cfg.data, (str, Path)) else list(cfg.data)
    if not entries:
        raise ValueError("no dataset")
    parsed = []
    for entry in entries:
        label, path = _split_entry(entry)
        if label is None:
            titles = gd.GenericDataset(path, check_layouts=False).titles
            label = titles[0] if len(titles) == 1 else "+".join(titles)
        parsed.append((label, path))
    labels = [label for label, _ in parsed]
    if len(set(labels)) != len(labels):
        raise ValueError(f"dataset labels {labels} are not unique: name them (label=path)")
    for name in ("mix", "splits"):
        unknown = sorted(set(getattr(cfg, name) or {}) - set(labels))
        if unknown:
            raise ValueError(f"{name}: unknown dataset labels {unknown} (have {labels})")
    for name in ("select_title",):
        value = getattr(cfg, name)
        if value is not None and value not in labels:
            raise ValueError(f"{name} {value!r} is not a dataset label (have {labels})")
    unknown = sorted(set(cfg.eval_titles or ()) - set(labels))
    if unknown:
        raise ValueError(f"eval_titles: unknown dataset labels {unknown} (have {labels})")
    sources = []
    for label, path in parsed:
        sources.append(
            gd.DataSource(
                path=str(path),
                label=label,
                splits=(cfg.splits or {}).get(label),
                max_games=_per_label(cfg.train_max_games, label, labels, "train_max_games"),
                val_max_games=_per_label(cfg.val_max_games, label, labels, "val_max_games"),
                report_max_games=_per_label(cfg.report_max_games, label, labels, "report_max_games"),
            )
        )
    if cfg.mix is not None:
        missing = [s.label for s in sources if s.label not in cfg.mix]
        if missing:
            raise ValueError(f"mix: no share for {missing}")
        for s in sources:
            s.weight = float(cfg.mix[s.label])
    else:
        for s in sources:
            ds = s.dataset(check_layouts=False)
            dec = ds.decisions()
            s.weight = float(sum(dec[g] for g in s.train_ids(ds)))
    if sum(s.weight for s in sources) <= 0 or any(s.weight < 0 for s in sources):
        raise ValueError(f"mix shares must be >= 0 with a positive sum: {[(s.label, s.weight) for s in sources]}")
    return sources


def _use_stream(cfg: GenericPretrainConfig, sources: List[gd.DataSource]) -> bool:
    if cfg.loader not in ("auto", "epochs", "stream"):
        raise ValueError(f"loader must be auto / epochs / stream, got {cfg.loader!r}")
    if cfg.loader == "epochs":
        if len(sources) > 1:
            raise ValueError("the epochs loader reads one dataset; use the stream loader for several")
        return False
    return cfg.loader == "stream" or len(sources) > 1 or cfg.mix is not None


def _merge_titles(*lists) -> List[str]:
    out: List[str] = []
    for titles in lists:
        for t in titles:
            if t not in out:
                out.append(t)
    return out


def load_init(path, model_overrides: Optional[dict], datasets: Dict[str, gd.GenericDataset], device) -> tuple:
    """``(model, generic_meta)`` of a checkpoint a run starts from (fine-tuning). Refuses
    (``model_generic.CheckpointMismatch``) what ``load_checkpoint`` refuses -- another architecture,
    encoder version, seat count, price cells, a vocabulary that is not an append-only extension of the
    checkpoint's, a layout fingerprint that differs from the engine's -- and, against this run's datasets:
    a requested model config that differs from the checkpoint's (the run keeps the checkpoint's
    architecture), a dataset title the checkpoint has no embedding row for (registered after it was
    trained), and a dataset whose layout for a title the checkpoint was trained on differs from the
    checkpoint's."""
    model, meta = mg.load_checkpoint(path, device)
    saved = model.config.to_json()
    differ = {k: (v, saved.get(k)) for k, v in (model_overrides or {}).items() if saved.get(k) != v}
    if differ:
        raise mg.CheckpointMismatch(
            f"--init {path}: the requested model config differs from the checkpoint's {differ} "
            "(requested, checkpoint); a run from a checkpoint keeps its architecture"
        )
    titles = mg.vocab()["titles"]
    for label, ds in datasets.items():
        for title in ds.titles:
            if title not in titles:
                raise mg.CheckpointMismatch(f"dataset {label}: title {title} unknown to the running engine")
            tid = titles.index(title) + 1
            if tid >= model.config.num_titles:
                raise mg.CheckpointMismatch(
                    f"--init {path}: dataset {label}'s title {title} (id {tid}) is beyond the checkpoint's "
                    f"{model.config.num_titles - 1} titles"
                )
            ours, theirs = ds.meta["layouts"].get(title), (meta.get("layouts") or {}).get(title)
            if ours and theirs and (ours["version"], ours["fingerprint"]) != (theirs["version"], theirs["fingerprint"]):
                raise mg.CheckpointMismatch(
                    f"--init {path}: {title} layout {theirs['version']} {theirs['fingerprint']} != dataset "
                    f"{label}'s {ours['version']} {ours['fingerprint']}"
                )
    return model, meta


# ---------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------


SELECT_METRICS = ("auto", "policy", "total")


def select_key(cfg: GenericPretrainConfig) -> str:
    """The evaluation key that picks the best joint checkpoint (``cfg.select_metric``)."""
    if cfg.select_metric not in SELECT_METRICS:
        raise ValueError(f"select_metric {cfg.select_metric!r}: one of {SELECT_METRICS}")
    metric = cfg.select_metric
    if metric == "auto":
        metric = "policy" if cfg.value_refit else "total"
    return "policy_total" if metric == "policy" else "total"


def add_policy_total(m: dict, cfg: GenericPretrainConfig) -> None:
    """``m["policy_total"]``: an evaluation's joint loss without the value terms (win CE, score MSE)."""
    if "total" in m:
        m["policy_total"] = m["total"] - cfg.value_loss_weight * m.get("value", 0.0) - cfg.score_loss_weight * m.get(
            "score", 0.0
        )


class _Selection:
    """Which number picks the best checkpoint: the selected dataset's, else the mix-weighted mean over
    the evaluated datasets (plain mean if they all have weight 0)."""

    def __init__(self, cfg: GenericPretrainConfig, sources: List[gd.DataSource]):
        self.labels = list(cfg.eval_titles or [s.label for s in sources])
        self.select = cfg.select_title or (sources[0].label if len(sources) == 1 else None)
        if self.select and self.select not in self.labels:
            self.labels.append(self.select)
        weights = {s.label: s.weight for s in sources}
        total = sum(weights[label] for label in self.labels)
        self.weights = {label: (weights[label] / total if total > 0 else 1.0 / len(self.labels)) for label in self.labels}

    def value(self, metrics: Dict[str, dict], key: str) -> float:
        if self.select:
            return float(metrics[self.select][key])
        return float(sum(self.weights[label] * metrics[label][key] for label in self.labels))

    def describe(self) -> str:
        return self.select or "mix-weighted mean of " + ", ".join(f"{k} {w:.3f}" for k, w in self.weights.items())


def _check_split(source: gd.DataSource, split: str, what: str) -> None:
    ds = source.dataset(check_layouts=False)
    if not ds.game_ids(split):
        raise ValueError(f"dataset {source.label}: no {split!r} games for {what} ({ds.split_names()})")


def pretrain(cfg: GenericPretrainConfig) -> dict:
    """Run both stages; returns a summary (also ``<out>/summary.json``)."""
    torch.set_num_threads(max(1, cfg.threads))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    use_amp = cfg.bf16 == "on" or (cfg.bf16 == "auto" and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sources = resolve_sources(cfg)
    stream = _use_stream(cfg, sources)
    datasets = {s.label: s.dataset() for s in sources}  # checks each dataset's layouts against the engine
    multi = len(sources) > 1
    selection = _Selection(cfg, sources)
    for label in selection.labels:
        _check_split(next(s for s in sources if s.label == label), cfg.select_split, "evaluation")
    titles = _merge_titles(*(ds.titles for ds in datasets.values()))
    init_meta = None
    if cfg.init:
        model, init_meta = load_init(cfg.init, cfg.model, datasets, device)
        titles = _merge_titles(init_meta.get("titles") or [], titles)
    else:
        model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**cfg.model)).to(device)
    init_info = None
    if init_meta is not None:
        extra = init_meta.get("extra") or {}
        init_info = {"path": str(cfg.init), "titles": init_meta.get("titles"), "stage": extra.get("stage"),
                     "step": extra.get("step"), "data": extra.get("data")}
    (out / "config.json").write_text(
        json.dumps(
            {
                "pretrain": asdict(cfg),
                "model": model.config.to_json(),
                "data_meta": _data_meta(datasets),
                "sources": [asdict(s) for s in sources],
                "loader": "stream" if stream else "epochs",
                "selection": selection.describe(),
                "select_key": select_key(cfg),
                "init": init_info,
            },
            indent=2,
        )
    )
    log = _Log(out)
    train_positions = {}
    for s in sources:
        dec = datasets[s.label].decisions()
        train_positions[s.label] = sum(dec[g] for g in s.train_ids(datasets[s.label]))
    LOGGER.info(
        "Generic pretraining: %s parameters, device %s, bf16 %s, %s loader, data %s, selection on %s %s %s%s",
        f"{model.num_parameters():,}", device, use_amp, "stream" if stream else "epochs",
        ", ".join(
            f"{s.label}={s.path} ({train_positions[s.label]} train positions, share {s.weight / sum(x.weight for x in sources):.3f})"
            for s in sources
        ),
        selection.describe(), cfg.select_split, select_key(cfg), f", init {cfg.init}" if cfg.init else "",
    )
    if stream:
        positions = sum(train_positions[s.label] for s in sources if s.weight > 0)
    else:
        positions = train_positions[sources[0].label]
    steps_per_epoch = max(1, positions // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps) if not stream else cfg.max_steps
    eval_every = cfg.eval_every or (steps_per_epoch if stream else None)
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.value_lr_multiplier)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    best_loss, best_step, evals_since = float("inf"), None, 0
    key = select_key(cfg)
    best_path = out / "best_joint.pth"
    step, stop, started = 0, False, time.time()
    window = defaultdict(float)
    window_n = 0
    history_evals: List[dict] = []
    labels = [s.label for s in sources]

    def run_eval(epoch: int) -> None:
        nonlocal best_loss, best_step, evals_since, stop
        t0 = time.time()
        per = {}
        for s in sources:
            if s.label not in selection.labels:
                continue
            m = evaluate(
                model, s.path, cfg.select_split, batch_size=cfg.batch_size, device=device, workers=cfg.workers,
                max_games=s.val_max_games, cfg=cfg, splits=s.splits,
            )
            add_policy_total(m, cfg)
            per[s.label] = m
        seconds = round(time.time() - t0, 1)
        value = selection.value(per, key)
        improved = value < best_loss
        for label, m in per.items():
            m.update(stage="joint", step=step, epoch=epoch, seconds=seconds)
            entry = {"kind": "eval", **m}
            if multi or stream:
                entry.update(source=label, split=cfg.select_split)
            log.write(entry, f"val_{label}" if multi else "val", step)
            LOGGER.info(
                "eval step %d epoch %d%s: %s%s", step, epoch, f" [{label} {cfg.select_split}]" if multi or stream else "",
                _summary_line(m), " (best)" if improved and label == selection.select else "",
            )
        history_evals.append({"step": step, "value": value, "by_source": per})
        if multi:
            log.write({"kind": "select", "step": step, "epoch": epoch, "value": value, "best": improved}, "val", step)
            LOGGER.info("selection (%s): %.4f%s", selection.describe(), value, " (best)" if improved else "")
        if improved:
            best_loss, best_step, evals_since = value, step, 0
            metrics = per[next(iter(per))] if len(per) == 1 else per
            mg.save_checkpoint(model, best_path, titles, extra=_extra(cfg, "joint", step, epoch, metrics, datasets, init_info))
        else:
            evals_since += 1
            if cfg.patience and evals_since >= cfg.patience:
                LOGGER.warning("early stop at step %d: no improvement in %d evaluations", step, evals_since)
                stop = True

    def train_step(batch, epoch: int) -> None:
        nonlocal step, window_n
        b = gd.batch_to_device(batch, device)
        with _autocast(device, use_amp):
            out_ = model(b["inputs"])
        losses = compute_losses(out_, b, cfg)
        if not torch.isfinite(losses["total"]):
            parts = {k: float(v) for k, v in losses.items()}
            raise RuntimeError(f"generic pretraining diverged at step {step}: {parts}")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        if not torch.isfinite(norm):
            raise RuntimeError(f"generic pretraining diverged at step {step}: non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        step += 1
        with torch.no_grad():
            inputs = b["inputs"]
            rank = segment_rank(out_["logp"].float(), inputs["legal_pos"], inputs["legal_offsets"], b["target_index"])
            top1 = (rank == 0).float()
            window["top1"] += float(top1.mean())
            if "source" in b:  # per-dataset training metrics
                nll = -out_["logp"].float()[inputs["legal_offsets"][:-1] + b["target_index"]]
                for k, label in enumerate(labels):
                    mask = b["source"] == k
                    window[f"n:{label}"] += float(mask.sum())
                    window[f"nll:{label}"] += float(nll[mask].sum())
                    window[f"top1:{label}"] += float(top1[mask].sum())
        for k, v in losses.items():
            window[k] += float(v.detach())
        window_n += 1
        if step % cfg.log_every == 0 or step == total_steps:
            entry = {k: v / window_n for k, v in window.items() if ":" not in k}
            by_source = {}
            n_all = sum(window.get(f"n:{label}", 0.0) for label in labels)
            for label in labels:
                n = window.get(f"n:{label}", 0.0)
                if n:
                    by_source[label] = {
                        "share": n / n_all,
                        "policy_ce": window[f"nll:{label}"] / n,
                        "top1": window[f"top1:{label}"] / n,
                    }
            if by_source:
                entry["by_source"] = by_source
            entry.update(
                kind="train", step=step, epoch=epoch, lr=optimizer.param_groups[0]["lr"],
                grad_norm=float(norm), elapsed=round(time.time() - started, 1),
            )
            log.write(entry, "train", step)
            for label, r in by_source.items():
                for key, val in r.items():
                    log.writer.add_scalar(f"train_{label}/{key}", val, step)
            LOGGER.info(
                "step %d/%d epoch %d: loss %.4f policy_ce %.4f top1 %.3f price %.3f value %.4f lr %.2e%s",
                step, total_steps, epoch, entry["total"], entry["policy_ce"], entry["top1"], entry["price"],
                entry["value"], entry["lr"],
                "".join(f" | {k} ce {r['policy_ce']:.3f} ({r['share']:.2f})" for k, r in by_source.items()),
            )
            window.clear()
            window_n = 0

    epoch = 0
    if stream:
        model.train()
        loader = make_mixed_loader(
            sources, cfg.batch_size, cfg.workers, seed=cfg.seed, buffer_positions=cfg.buffer_positions,
            pin_memory=device.type == "cuda", config=model.config,
        )
        for batch in loader:
            epoch = step // steps_per_epoch
            model.value_stop_grad = cfg.value_joint_epochs is not None and epoch >= cfg.value_joint_epochs
            train_step(batch, epoch)
            if eval_every and step % eval_every == 0:
                run_eval(epoch)
            if stop or step >= total_steps:
                break
        del loader
    else:
        for epoch in range(cfg.epochs):
            model.train()
            model.value_stop_grad = cfg.value_joint_epochs is not None and epoch >= cfg.value_joint_epochs
            src = sources[0]
            loader = make_loader(
                src.path, "train", cfg.batch_size, True, cfg.workers, seed=cfg.seed, epoch=epoch,
                max_games=src.max_games, buffer_positions=cfg.buffer_positions,
                pin_memory=device.type == "cuda", config=model.config, splits=src.splits,
            )
            for batch in loader:
                train_step(batch, epoch)
                if eval_every and step % eval_every == 0:
                    run_eval(epoch)
                if stop or step >= total_steps:
                    break
            if not eval_every and not stop:
                run_eval(epoch)
            if stop or step >= total_steps:
                break
    if eval_every and (not history_evals or history_evals[-1]["step"] != step) and not stop:
        run_eval(epoch)
    model.value_stop_grad = False
    last_extra = _extra(cfg, "joint_last", step, epoch, None, datasets, init_info)
    mg.save_checkpoint(model, out / "last_joint.pth", titles, extra=last_extra)
    summary = {
        "joint": {"best_step": best_step, "best_val": best_loss, "select_key": key, "steps": step,
                  "evals": len(history_evals), "planned_steps": total_steps, "early_stopped": stop},
        "best_joint": str(best_path),
        "selection": {"on": selection.describe(), "split": cfg.select_split},
        "loader": "stream" if stream else "epochs",
    }
    if multi or stream:
        summary["joint"]["curve"] = [
            {"step": h["step"], "value": h["value"],
             **{label: {k: m.get(k) for k in ("total", "policy_total", "policy_ce", "top1", "value_ce")}
                for label, m in h["by_source"].items()}}
            for h in history_evals
        ]
    final = best_path
    if cfg.value_refit and best_path.exists():
        refit = refit_value_heads(
            best_path, cfg, device, use_amp, log, datasets, sources=sources, stream=stream, selection=selection,
            titles=titles, init_info=init_info,
        )
        summary["refit"] = refit
        final = Path(refit["path"])
    summary["final"] = str(final)
    if final.exists():
        import shutil

        shutil.copyfile(final, out / "final.pth")
        if cfg.final_eval:
            summary["final_eval"] = final_evaluation(out / "final.pth", cfg, sources, device, log, step)
    summary["seconds"] = round(time.time() - started, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.close()
    LOGGER.info("Generic pretraining done: %s", json.dumps({k: v for k, v in summary.items() if k != "final_eval"}))
    return summary


def final_evaluation(path, cfg: GenericPretrainConfig, sources: List[gd.DataSource], device, log=None, step=0) -> dict:
    """Metrics of a checkpoint on every dataset's ``report_split`` (datasets without that split are skipped),
    ``report_max_games`` games each."""
    model, _ = mg.load_checkpoint(path, device)
    out = {}
    for s in sources:
        if not s.dataset(check_layouts=False).game_ids(cfg.report_split):
            LOGGER.info("final eval: %s has no %s games, skipped", s.label, cfg.report_split)
            continue
        t0 = time.time()
        m = evaluate(
            model, s.path, cfg.report_split, batch_size=cfg.batch_size, device=device, workers=cfg.workers,
            max_games=s.report_max_games, cfg=cfg, splits=s.splits,
        )
        m.update(seconds=round(time.time() - t0, 1), split=cfg.report_split, checkpoint=str(path))
        out[s.label] = m
        if log is not None:
            log.write({"kind": "final_eval", "source": s.label, "step": step, **m}, f"final_{s.label}", step)
        LOGGER.info("final eval [%s %s]: %s", s.label, cfg.report_split, _summary_line(m))
    return out


def refit_value_heads(
    best_path: Path,
    cfg: GenericPretrainConfig,
    device: torch.device,
    use_amp: bool,
    log: _Log,
    dataset,
    sources: Optional[List[gd.DataSource]] = None,
    stream: bool = False,
    selection: Optional[_Selection] = None,
    titles: Optional[List[str]] = None,
    init_info: Optional[dict] = None,
) -> dict:
    """Stage 2: fresh value heads on the frozen best joint checkpoint, early-stopped on validation win CE
    (the selected dataset's, or the mix-weighted mean; ``cfg.select_split``). ``dataset`` is the run's
    dataset, or ``{label: dataset}``; with ``stream`` the heads train on the run's mixed stream."""
    datasets = dataset if isinstance(dataset, dict) else {"_": dataset}
    if sources is None:
        sources = resolve_sources(cfg)
    if selection is None:
        selection = _Selection(cfg, sources)
    if titles is None:
        titles = _merge_titles(*(ds.titles for ds in datasets.values()))
    model, _ = mg.load_checkpoint(best_path, device)
    reset_value_heads(model)
    heads = (model.win_head, model.score_head)
    head_ids = {id(p) for h in heads for p in h.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.value_stop_grad = True
    optimizer = torch.optim.AdamW(
        [p for h in heads for p in h.parameters()], lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def run_eval() -> tuple:
        per = {}
        for s in sources:
            if s.label in selection.labels:
                per[s.label] = evaluate(
                    model, s.path, cfg.select_split, batch_size=cfg.batch_size, device=device, workers=cfg.workers,
                    max_games=s.val_max_games, value_only=True, splits=s.splits,
                )
        return per, {k: selection.value(per, k) for k in ("value_ce", "value_baseline", "winner_acc", "score_mse")}

    per, m = run_eval()
    best_ce, best_acc = m["value_ce"], m["winner_acc"]
    best_state = [copy.deepcopy(h.state_dict()) for h in heads]
    LOGGER.info("value refit: fresh heads val value CE %.4f (equal odds %.4f)", best_ce, m["value_baseline"])
    curve = []
    step = evals_since = 0
    window: List[float] = []
    epoch = 0
    model.train()

    def batches():
        nonlocal epoch
        if stream:
            yield from make_mixed_loader(
                sources, cfg.batch_size, cfg.workers, seed=cfg.seed + 7919, buffer_positions=cfg.buffer_positions,
                pin_memory=device.type == "cuda", config=model.config,
            )
            return
        src = sources[0]
        while True:
            loader = make_loader(
                src.path, "train", cfg.batch_size, True, cfg.workers, seed=cfg.seed + 7919, epoch=epoch,
                max_games=src.max_games, buffer_positions=cfg.buffer_positions,
                pin_memory=device.type == "cuda", config=model.config, splits=src.splits,
            )
            epoch += 1
            progressed = False
            for batch in loader:
                progressed = True
                yield batch
            if not progressed:
                return

    if cfg.refit_max_steps > 0 and cfg.refit_patience > 0:
        for batch in batches():
            b = gd.batch_to_device(batch, device)
            with _autocast(device, use_amp):
                out_ = model(b["inputs"], value_only=True)
            v = mg.value_losses(out_, b["win"], b["score"])
            loss = v["ce"] + cfg.score_loss_weight * v["mse"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"value refit diverged at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            window.append(float(v["ce"].detach()))
            step += 1
            if step % cfg.refit_eval_steps == 0 or step >= cfg.refit_max_steps:
                per, m = run_eval()
                entry = {
                    "kind": "refit_eval", "step": step, "train_value": float(np.mean(window)),
                    "value_ce": m["value_ce"], "value_baseline": m["value_baseline"], "winner_acc": m["winner_acc"],
                    "score_mse": m["score_mse"],
                }
                if len(per) == 1:
                    entry["by_players"] = next(iter(per.values()))["by_players"]
                else:
                    entry["by_source"] = {
                        label: {k: r[k] for k in ("value_ce", "value_baseline", "winner_acc", "score_mse")}
                        for label, r in per.items()
                    }
                window = []
                curve.append(entry)
                log.write(entry, "value_refit", step)
                improved = m["value_ce"] < best_ce
                if improved:
                    best_ce, best_acc, evals_since = m["value_ce"], m["winner_acc"], 0
                    best_state = [copy.deepcopy(h.state_dict()) for h in heads]
                else:
                    evals_since += 1
                LOGGER.info(
                    "value refit step %d: train %.4f val %.4f (equal odds %.4f) winner %.3f%s",
                    step, entry["train_value"], m["value_ce"], m["value_baseline"], m["winner_acc"],
                    " (best)" if improved else "",
                )
                if evals_since >= cfg.refit_patience or step >= cfg.refit_max_steps:
                    break
    for h, state in zip(heads, best_state):
        h.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = True
    model.value_stop_grad = False
    path = Path(cfg.out) / "refit.pth"
    mg.save_checkpoint(
        model, path, titles,
        extra=_extra(cfg, "refit", step, None, {"value_ce": best_ce, "winner_acc": best_acc}, datasets, init_info),
    )
    return {"path": str(path), "steps": step, "val_value_ce": best_ce, "val_winner_acc": best_acc, "curve": curve}


def _data_meta(dataset) -> dict:
    """A dataset's meta summary; for ``{label: dataset}`` one per label (a single one unwrapped)."""
    if isinstance(dataset, dict):
        if len(dataset) == 1:
            return _data_meta(next(iter(dataset.values())))
        return {label: _data_meta(ds) for label, ds in dataset.items()}
    m = dataset.meta
    return {k: m.get(k) for k in ("titles", "layouts", "encoder_version_at_conversion", "counts", "sources", "created")}


def _extra(cfg, stage, step, epoch, metrics, dataset, init_info=None) -> dict:
    keep = None
    if metrics:
        if all(isinstance(v, dict) and "positions" in v for v in metrics.values()):  # {label: metrics}
            keep = {label: {k: v for k, v in m.items() if k != "by_players"} for label, m in metrics.items()}
        else:
            keep = {k: v for k, v in metrics.items() if k not in ("by_players",)}
    return {
        "stage": stage,
        "step": step,
        "epoch": epoch,
        "data": cfg.data if isinstance(cfg.data, str) else list(cfg.data),
        "data_meta": _data_meta(dataset),
        "pretrain_config": asdict(cfg),
        "metrics": keep,
        "init": init_info,
    }


# ---------------------------------------------------------------------------------------------------
# Comparison with a legacy 1830 checkpoint on the same positions
# ---------------------------------------------------------------------------------------------------


@torch.no_grad()
def compare_with_legacy(
    generic_path: str,
    legacy_path: str,
    data: str,
    split: str = "validation",
    batch_size: int = 128,
    device: str = "cpu",
    max_games: Optional[int] = None,
    max_positions: Optional[int] = None,
) -> dict:
    """Score a generic checkpoint and a legacy 1830 checkpoint (``composite_model.load_policy_value``
    spec) on the same decisions of ``split``: top-1 / top-5 over the legal moves overall, per descriptor
    family and per player count, policy CE, and the value heads' win CE (vs equal odds) and winner
    accuracy -- both models read the same positions, so the families and positions match exactly."""
    from rl18xx.agent.alphazero.composite_model import load_policy_value
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.rust_adapter import RustGameAdapter

    device = torch.device(device)
    generic, _ = mg.load_checkpoint(generic_path, device)
    generic.eval()
    legacy = load_policy_value(legacy_path)  # on the best available device (its own ``device``)
    legacy.eval()
    dataset = gd.GenericDataset(data)
    families = mg.vocab()["families"]
    sums = {"generic": MetricSums(families), "legacy": MetricSums(families)}
    rotation_mismatch = 0
    pending: List[tuple] = []

    def flush():
        nonlocal rotation_mismatch
        if not pending:
            return
        items = [p[0] for p in pending]
        batch = gd.collate(items, generic.config)
        b = gd.batch_to_device(batch, device)
        sums["generic"].add(generic(b["inputs"]), b)
        # legacy: dense policy over the 1830 layout, read at each position's legal indices (the same list)
        encoded = [p[1] for p in pending]
        legal = [[int(i) for i in it["indices"]] for it in items]
        probs, _, values = legacy.run_many_encoded(encoded, legal)
        probs, values = probs.float().to(device), values.float().to(device)
        logp_rows, win_rows = [], []
        for k, it in enumerate(items):
            p = probs[k, torch.as_tensor(legal[k], dtype=torch.long, device=probs.device)]
            p = p / p.sum().clamp(min=1e-30)
            logp_rows.append(torch.log(p.clamp(min=1e-30)))
            n = it["num_players"]
            v = values[k, :n].float()
            v = v / v.sum().clamp(min=1e-30)
            w = torch.full((gd.MAX_SEATS,), mg.MASKED, device=device)
            w[:n] = torch.log(v.clamp(min=1e-30))
            win_rows.append(w)
            if int(encoded[k][6]) != int(it["enc"]["meta"][0, 5]):
                rotation_mismatch += 1
        legacy_out = {
            "logp": torch.cat(logp_rows).to(device),
            "price_logp": None,
            "win_logits": torch.stack(win_rows),
            "present": b["inputs"]["ent_mask"][:, 1:7],
            "score": torch.zeros_like(b["win"]),
        }
        sums["legacy"].add(legacy_out, b)
        pending.clear()

    seen = 0
    for gid in dataset.game_ids(split)[: max_games or None]:
        record = dataset.record(gid)
        d = None
        for decision, game in gd.replay(record):
            item = gd.position_item(record, decision, game, d)
            pending.append((item, _rust_encode(RustGameAdapter(game.pickle_clone()))))
            seen += 1
            if len(pending) >= batch_size:
                flush()
            if max_positions and seen >= max_positions:
                break
        if max_positions and seen >= max_positions:
            break
    flush()
    result = {name: s.result() for name, s in sums.items()}
    result["rotation_mismatch"] = rotation_mismatch
    result["generic_checkpoint"] = str(generic_path)
    result["legacy_checkpoint"] = str(legacy_path)
    result["data"] = str(data)
    result["split"] = split
    return result


__all__ = [
    "GenericPretrainConfig",
    "MODEL_PRESETS",
    "SMALL_CONFIG",
    "build_optimizer",
    "compare_with_legacy",
    "compute_losses",
    "evaluate",
    "final_evaluation",
    "load_init",
    "make_mixed_loader",
    "pretrain",
    "refit_value_heads",
    "reset_value_heads",
    "resolve_sources",
    "segment_rank",
]
