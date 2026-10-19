"""Policy targets from a PUCT search's root (docs/search_learning_design.md,
"Q-based policy targets").

A visit distribution moves probability where the prior sent visits; in 1830
siblings differ by ~0.01-0.03 win share, so at low exploration the visits
largely echo the prior. The targets here move probability because a move's
Q is better:

- :func:`completed_q_target` (``cq``): Gumbel MuZero's improved policy
  softmax(logits + sigma(completed q)) (Danihelka et al. 2022), computed for
  any expanded root -- a PUCT one too. logits = log of the tree's priors
  (after ``prior_temperature``), an unvisited move's q completed by the root's
  ``v_mix``, sigma(q) = (c_visit + max_b N(b)) * c_scale * q on the mover's 0-1
  win share; ``rescale`` first maps the completed q's to [0, 1] by their
  min-max, as mctx's ``qtransform_completed_by_mix_value(rescale_values=True)``
  (its default, with c_scale 0.1). With rescale off it is the Gumbel root's
  ``gumbel_policy_target`` (``engine-rs/src/gumbel.rs``; equal to float32
  rounding, tested), here on any root.
- :func:`rpo_target` (``rpo``): Grill et al. 2020, "Monte-Carlo Tree Search as
  Regularized Policy Optimization": the exact solution of
  max_y [q^T y - lambda_N KL(pi_theta, y)], pi_bar = lambda_N pi_theta / (alpha - q),
  lambda_N = c * sqrt(sum_b N_b) / (|A| + sum_b N_b), alpha by bisection so pi_bar
  sums to 1 (the paper's bounds: alpha in [max_b (q_b + lambda_N pi_b),
  max_b q_b + lambda_N]). c is the tree's own effective exploration constant
  by default (``root_target_stats()["c_puct"]``: what PUCT used at the root),
  which makes pi_bar the policy our PUCT approximates. q: visited moves' mean
  value; unvisited ones ``vmix`` (default, the cq target's completion),
  ``parent`` (the root's mean value) or ``min`` (the lowest visited q);
  ``normalize`` maps q to [0, 1] by the min-max over the root's value and its
  visited children (MuZero / mctx ``qtransform_by_parent_and_siblings``).

A price-bearing root move is one slot: its N and q aggregate its price
grandchildren (the tree mirrors their backups into the slot), the policy
head's view of the slot, as on a Gumbel root.

Every function takes ``stats`` = :func:`root_stats` of a searched tree and
returns a float64 distribution over ``stats["legal"]`` (the root's legal moves,
``legal_action_indices_at_root`` order).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

LOW_LOGIT = -1e9  # gumbel.rs's floor for log-priors
Q_NORM_EPS = 1e-6
TARGET_KINDS = ("visits", "cq", "rpo")
RPO_UNVISITED = ("vmix", "parent", "min")


def root_stats(tree) -> dict:
    """``tree.root_target_stats()`` (a ``engine_rs.RustMCTSPlayer``) as numpy."""
    s = dict(tree.root_target_stats())
    s["legal"] = np.asarray(s["legal"], dtype=np.int64)
    for key in ("child_n", "child_prior", "child_q", "completed_q"):
        s[key] = np.asarray(s[key], dtype=np.float64)
    s["open_price"] = np.asarray(s["open_price"], dtype=bool)
    for key in ("raw_value", "root_value", "v_mix", "root_n", "c_puct"):
        s[key] = float(s[key])
    return s


def visit_target(stats: dict) -> np.ndarray:
    n = stats["child_n"]
    return n / n.sum() if n.sum() > 0 else np.full(len(n), 1.0 / max(len(n), 1))


def _logits(prior: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(prior > 0, np.maximum(np.log(np.where(prior > 0, prior, 1.0)), LOW_LOGIT), LOW_LOGIT)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _minmax(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (q - lo) / max(hi - lo, Q_NORM_EPS)


def completed_q_target(stats: dict, c_visit: float = 50.0, c_scale: float = 1.0, rescale: bool = False) -> np.ndarray:
    """softmax(logits + (c_visit + max N) * c_scale * completed q) over the legal moves."""
    q = stats["completed_q"].copy()
    if rescale:
        q = _minmax(q, float(q.min()), float(q.max()))
    scale = (c_visit + float(stats["child_n"].max(initial=0.0))) * c_scale
    return _softmax(_logits(stats["child_prior"]) + scale * q)


def rpo_q(stats: dict, unvisited: str = "vmix", normalize: bool = False) -> np.ndarray:
    """The q-hat of :func:`rpo_target`: visited moves' mean values for the mover,
    unvisited ones completed per ``unvisited``, optionally min-max normalized."""
    if unvisited not in RPO_UNVISITED:
        raise ValueError(f"unvisited must be one of {RPO_UNVISITED}, got {unvisited!r}")
    visited = stats["child_n"] > 0
    q = stats["child_q"].copy()
    root_value = stats["root_value"] if np.isfinite(stats["root_value"]) else stats["raw_value"]
    if not visited.any():
        q[:] = root_value
    elif unvisited == "vmix":
        q[~visited] = stats["v_mix"]
    elif unvisited == "parent":
        q[~visited] = root_value
    else:
        q[~visited] = float(q[visited].min())
    if normalize:
        seen = q[visited] if visited.any() else q
        q = _minmax(q, min(root_value, float(seen.min())), max(root_value, float(seen.max())))
    return q


def rpo_lambda(stats: dict, c: Optional[float] = None) -> float:
    """lambda_N = c * sqrt(sum N) / (|A| + sum N); c None / <= 0: the tree's own c_puct."""
    n = float(stats["child_n"].sum())
    c = float(stats["c_puct"]) if c is None or c <= 0 else float(c)
    return c * np.sqrt(n) / (len(stats["child_n"]) + n)


def solve_rpo(prior: np.ndarray, q: np.ndarray, lam: float, iters: int = 80) -> np.ndarray:
    """argmax_y [q^T y - lam KL(prior, y)] over the simplex: lam prior / (alpha - q),
    alpha by bisection (Grill et al. 2020, Proposition 1 / Appendix B.3)."""
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / prior.sum()
    q = np.asarray(q, dtype=np.float64)
    if lam <= 0:
        out = np.zeros(len(q))
        out[int(np.argmax(np.where(prior > 0, q, -np.inf)))] = 1.0
        return out
    support = prior > 0
    lo = float(np.max(q[support] + lam * prior[support]))
    hi = float(np.max(q[support]) + lam)
    p, qs = prior[support], q[support]
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            gap = mid - qs
            # alpha at or below a move's q: that move alone would need more than all the mass.
            if (gap <= 0).any() or (lam * p / gap).sum() > 1.0:
                lo = mid
            else:
                hi = mid
    out = np.zeros(len(q))
    out[support] = lam * prior[support] / (hi - q[support])
    return out / out.sum()


def rpo_target(
    stats: dict, c: Optional[float] = None, unvisited: str = "vmix", normalize: bool = False
) -> np.ndarray:
    """Grill et al.'s pi_bar at the root (see the module docs); the prior when
    the root has no child visits."""
    prior = stats["child_prior"]
    if stats["child_n"].sum() <= 0 or prior.sum() <= 0:
        return prior / prior.sum() if prior.sum() > 0 else visit_target(stats)
    return solve_rpo(prior, rpo_q(stats, unvisited, normalize), rpo_lambda(stats, c))


def search_target(tree, legal: np.ndarray, settings: dict) -> np.ndarray:
    """A PUCT root's training target over ``legal`` (its root's legal moves) per
    ``settings["target"]``: ``visits`` (``visit_target`` "pruned": KataGo's pruned
    visit counts, else the raw counts -- unnormalized), ``cq`` or ``rpo`` (their
    ``cq_*`` / ``gumbel_c_*`` / ``rpo_*`` settings)."""
    kind = settings["target"]
    if kind == "visits":
        if settings.get("visit_target", "pruned") == "pruned":
            return np.asarray(tree.pruned_child_n_at_root(), dtype=np.float64)[legal]
        return np.asarray(tree.child_n_at_root(), dtype=np.float64)[legal]
    stats = root_stats(tree)
    if not np.array_equal(stats["legal"], legal):
        raise ValueError("search_target: legal moves are not the root's")
    if kind == "cq":
        target = completed_q_target(
            stats,
            float(settings.get("gumbel_c_visit", 50.0)),
            float(settings.get("gumbel_c_scale", 1.0)),
            bool(settings.get("cq_rescale", False)),
        )
    elif kind == "rpo":
        target = rpo_target(
            stats,
            settings.get("rpo_c"),
            str(settings.get("rpo_unvisited", "vmix")),
            bool(settings.get("rpo_qnorm", False)),
        )
    else:
        raise ValueError(f"unknown PUCT target {kind!r}")
    return np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
