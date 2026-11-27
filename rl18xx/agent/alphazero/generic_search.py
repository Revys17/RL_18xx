"""Searched play for titles whose legal moves come from their own generator (1867; the 1822 family's trees work the
same way): a title-generic policy's priors and price cells and a title-generic critic's values, searched by the Rust
tree (``docs/rl_model.md`` "Searching 1867").

**The tree** is the 1830 one (``engine_rs.RustMCTSPlayer``: mean Q with first-play urgency, price grandchildren by
progressive widening, ``dedupe_leaves``, ``prior_temperature``, forced-move collapsing, tree reuse, the Gumbel root),
built on a generator title's game (``mcts.rs`` ``TreeMoves::Generator``): its legal moves are the title generator's
under the title's own cross-player policy (what the generic policy-gradient runs trained on), indexed in the title's
RL layout, a priced move's prices on its own grid with the generic price head's cells (step units over the move's
range), and each node backs up the value of the encoder's mover (``rl_mover_id``: seat 0 of the generic encoding).
A single legal move whose price is still open is a decision node, not part of a forced chain.

**Evaluation** is always lockstep (``lockstep.LockstepSearcher`` with a ``lockstep.GenericEvaluator``): the leaves
go to the networks as games with their legal moves' price ranges (``engine_rs.mcts_collect_leaf_games``), and the
price outputs come back per legal move (``mcts_apply_results``'s ``price_rows``), the values moved from the generic
encoder's frame into the tree's. The networks are generic inference servers (an eval on the GPU) or in-process
``LocalGenericClient`` s (:func:`local_evaluator`: the bots, CPU tests).

**Specs** (:func:`parse_search_spec`) follow ``scripts/eval_head_to_head.py``'s player grammar:
``[generic:]<policy>[+[generic:]<value>][#win|#score|#blend=<w>][@<readouts>][/<c_puct_init>][:<option>,...]`` with
its options (``mean``, ``par=``, ``dedupe``, ``ptemp=``, ``qnorm``, ``nowiden``, ``min=``, ``budget=``,
``gumbel=``, ``gscale=``, ``cvisit=``, ``cscale=``; ``sample=<t>``: the policy alone at temperature t).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

GENERIC_PREFIX = "generic:"
#: The 1830 bots' search settings (``advisor.SEARCH_OVERRIDES``): mean Q with first-play urgency at the parent's
#: value, c_puct 0.05 in every round, 8 leaves per batch, each distinct leaf backed up once.
BOT_SEARCH = dict(
    mcts_mean_q=True, fpu_reduction=0.0, c_puct_init=0.05, c_puct_by_round={}, dedupe_leaves=True, parallel_readouts=8
)
#: The bots' budget per decision when a spec gives none.
BOT_READOUTS = 400


def _strip_generic(part: str) -> str:
    return part[len(GENERIC_PREFIX) :] if part.startswith(GENERIC_PREFIX) else part


def search_options(options: str) -> dict:
    """``SelfPlayConfig`` overrides of a spec's ``:<option>,...`` part (``eval_head_to_head.parse_player``'s
    options; ``sample=<t>`` gives ``{"policy_temperature": t}`` alone)."""
    overrides: dict = {}
    for option in filter(None, str(options).split(",")):
        key, _, value = option.partition("=")
        if key == "mean":
            overrides.update(mcts_mean_q=True, fpu_reduction=float(value) if value else 0.0)
        elif key == "par":
            overrides["parallel_readouts"] = int(value)
        elif key == "min":
            overrides["min_readouts"] = int(value)
        elif key == "ptemp":
            overrides["prior_temperature"] = float(value)
        elif key == "dedupe":
            overrides["dedupe_leaves"] = True
        elif key == "qnorm":
            overrides["mcts_q_normalize"] = True
        elif key == "nowiden":
            overrides["mcts_move_widening"] = False
        elif key == "gumbel":
            overrides["gumbel_k"] = int(value) if value else 16
        elif key == "gscale":
            overrides["gumbel_noise_scale"] = float(value)
        elif key == "cvisit":
            overrides["gumbel_c_visit"] = float(value)
        elif key == "cscale":
            overrides["gumbel_c_scale"] = float(value)
        elif key == "budget":
            if value not in ("additional", "total"):
                raise ValueError(f"budget={value!r}: additional or total")
            overrides["readout_budget"] = value
        elif key == "sample":
            return {"policy_temperature": float(value) if value else 1.0}
        else:
            raise ValueError(f"unknown search option {option!r}")
    if overrides.get("mcts_q_normalize") and not overrides.get("mcts_mean_q"):
        raise ValueError("qnorm needs mean")
    return overrides


@dataclass
class SearchSpec:
    """A searched generic system: a policy checkpoint (priors, price cells), a value checkpoint (None: the
    policy network's own heads) read from ``value_head``, and the search's ``SelfPlayConfig`` overrides."""

    policy: str
    value: Optional[str]
    value_head: str = "win"
    overrides: dict = field(default_factory=dict)

    @property
    def readouts(self) -> int:
        return int(self.overrides.get("num_readouts", 200))

    @property
    def policy_only(self) -> bool:
        return "policy_temperature" in self.overrides

    def label(self) -> str:
        parts = GENERIC_PREFIX + self.policy + (f"+{GENERIC_PREFIX}{self.value}" if self.value else "")
        return parts + (f"#{self.value_head}" if self.value_head != "win" else "")

    def terminal_utility(self, num_players: int) -> tuple:
        """``(weight, points)`` a finished game backs up at ``num_players`` (``composite_model.terminal_utility``:
        placement points when the value is a placement-reward run's score head)."""
        from rl18xx.agent.alphazero.composite_model import checkpoint_reward, terminal_utility

        reward = checkpoint_reward(self.value or self.policy)
        if reward is None:
            return 0.0, ()
        return terminal_utility(self.value_head, reward, num_players)

    def config(self, num_players: int, **extra):
        """The tree's ``SelfPlayConfig`` for a ``num_players`` game (``extra``: more overrides)."""
        weight, points = self.terminal_utility(num_players)
        overrides = {**self.overrides, **extra}
        if weight > 0:
            overrides.update(terminal_score_weight=float(weight), terminal_score_points=tuple(points))
        return search_config(**overrides)


def parse_search_spec(
    spec: str, default_readouts: int = 200, default_head: str = "win", policy: Optional[str] = None
) -> SearchSpec:
    """A spec (module docstring) as a :class:`SearchSpec`. With ``policy`` (the bots' ``--title-search``: the
    policy is ``--title-policy``'s) the spec names only the value: ``[generic:]<value>[#<head>][@...][/...][:...]``
    (``self`` for the policy's own heads). Checkpoint files must exist."""
    body = str(spec)
    body, _, options = _split_options(body)
    body, _, search = body.partition("@")
    readouts, _, c_puct = search.partition("/")
    checkpoints, hash_, head = body.partition("#")
    overrides = search_options(options)
    if "policy_temperature" not in overrides:
        budget = int(readouts) if readouts else int(default_readouts)
        overrides = {"num_readouts": budget, "min_readouts": budget, **overrides}
        if c_puct:
            overrides.update(c_puct_init=float(c_puct), c_puct_by_round={})
    parts = [_strip_generic(p) for p in checkpoints.split("+")]
    if policy is not None:
        if len(parts) != 1:
            raise ValueError(f"{spec!r}: name the value checkpoint only (the policy is {policy})")
        value = None if parts[0] in ("", "self") else parts[0]
        parts = [policy, value]
    elif len(parts) > 2:
        raise ValueError(f"{spec!r}: <policy>[+<value>]")
    paths = [None if p is None else str(Path(p).resolve()) for p in parts]
    for p in paths:
        if p is not None and not Path(p).is_file():
            raise FileNotFoundError(f"No generic checkpoint at {p}")
    return SearchSpec(
        policy=paths[0],
        value=paths[1] if len(paths) > 1 else None,
        value_head=head if hash_ else default_head,
        overrides=overrides,
    )


def _split_options(body: str) -> tuple:
    """``(checkpoints and budget, ":", options)``: the options start at the first ``:`` after the last
    ``generic:`` prefix's."""
    start = 0
    while True:
        i = body.find(GENERIC_PREFIX, start)
        if i < 0:
            break
        start = i + len(GENERIC_PREFIX)
    i = body.find(":", start)
    return (body, "", "") if i < 0 else (body[:i], ":", body[i + 1 :])


def search_config(**overrides):
    """A ``SelfPlayConfig`` for searched play: no Dirichlet noise, no resignation, the most visited move from the
    first decision (``softpick_move_cutoff`` 0) unless overridden."""
    from rl18xx.agent.alphazero.config import SelfPlayConfig

    settings = dict(dirichlet_noise_weight=0.0, enable_resign=False, noresign_holdout_rate=0.0, softpick_move_cutoff=0)
    settings.update(overrides)
    return SelfPlayConfig(network=None, **settings)


def local_evaluator(policy, value=None, value_head: str = "win", device: str = "cpu"):
    """A ``lockstep.GenericEvaluator`` on in-process networks (``LocalGenericClient``; no compression): ``policy``
    / ``value`` are checkpoint paths, ``LocalGenericServer`` s or loaded models (``value`` None: the policy
    network's own heads)."""
    from rl18xx.agent.alphazero.generic_inference import LocalGenericClient, LocalGenericServer
    from rl18xx.agent.alphazero.lockstep import GenericEvaluator

    def client(part):
        if part is None:
            return None
        if isinstance(part, LocalGenericClient):
            return part
        if isinstance(part, LocalGenericServer):
            return LocalGenericClient(part)
        if isinstance(part, (str, Path)):
            from rl18xx.agent.alphazero import model_generic as mg

            part, _ = mg.load_checkpoint(str(part), device)
        return LocalGenericClient(LocalGenericServer(part.eval()))

    return GenericEvaluator(client(policy), client(value), value_head, compress=False)


class GenericSearchPlayer:
    """One Rust search tree over a generator title's game (module docstring), driven like
    ``rust_mcts_player.RustMCTSPlayer`` by a ``lockstep.LockstepSearcher`` (:meth:`start_search` gives the
    ``TreeSearch``). :meth:`follow` keeps the tree while its root is the game's position (the same player moving
    again right after its own move) and builds a fresh one otherwise; :meth:`play` commits the root's move and
    returns the action dicts the tree logged (the move at its price, then the forced moves after it)."""

    def __init__(self, config):
        self.config = config
        self._rust_player = None
        self.decisions = 0  # decisions this player has played (softpick_move_cutoff counts them)
        self._synced = 0  # the root's action count when it last matched a game
        self.kept = 0  # searches that started from a kept tree

    def reset(self, raw) -> None:
        """A fresh tree at ``raw``'s position (an ``engine_rs.BaseGame`` of a generator title)."""
        import engine_rs

        cfg = self.config
        tree = engine_rs.RustMCTSPlayer(
            raw, float(cfg.pw_c), float(cfg.pw_alpha), int(cfg.min_price_children), float(cfg.price_explore_eps)
        )
        tree.set_search_config(
            float(cfg.c_puct_init),
            float(cfg.c_puct_base),
            {str(k): float(v) for k, v in (cfg.c_puct_by_round or {}).items()},
            int(cfg.max_engine_actions),
        )
        if cfg.mcts_mean_q:
            tree.set_q_config(True, float(cfg.fpu_reduction))
            if cfg.mcts_q_normalize:
                tree.set_q_normalize(True)
        elif cfg.mcts_q_normalize:
            raise ValueError("mcts_q_normalize needs mcts_mean_q")
        if not cfg.mcts_move_widening:
            tree.set_move_widening(False)
        if float(cfg.terminal_score_weight) > 0:
            tree.set_terminal_utility(float(cfg.terminal_score_weight), [float(p) for p in cfg.terminal_score_points])
        if int(cfg.gumbel_k) > 0:
            tree.set_gumbel_config(
                int(cfg.gumbel_k), float(cfg.gumbel_c_visit), float(cfg.gumbel_c_scale), float(cfg.gumbel_noise_scale)
            )
        self._rust_player = tree
        self._synced = len(raw.raw_actions)
        self._log = None  # the root's actions without ignored fields (follow(..., ignore=...)), built on demand

    @staticmethod
    def _stripped(actions, ignore) -> list:
        return [{k: v for k, v in dict(a).items() if k not in ignore} for a in actions]

    def follow(self, raw, ignore: tuple = ()) -> bool:
        """Search from ``raw``'s position: keep the tree when its root is that position (its log matches the
        game's since they last agreed), else build a fresh one. Returns whether the tree was kept. ``ignore``:
        action fields the comparison skips, over the whole log (a site's game: its ``id`` / ``user`` /
        ``created_at``, which the tree's own action dicts lack)."""
        tree = self._rust_player
        if tree is not None and ignore:
            if self._log is None:
                self._log = self._stripped(tree.root_game_object().raw_actions, ignore)
            if self._log == self._stripped(raw.raw_actions, ignore):
                self.kept += 1
                return True
        elif tree is not None and self._synced <= len(raw.raw_actions) and tree.root_log_matches(raw, self._synced):
            self.kept += 1
            return True
        self.reset(raw)
        return False

    @property
    def tree(self):
        return self._rust_player

    def release(self) -> None:
        """Drop the tree (a 400-readout 1867 tree holds ~75 MB of positions); the next :meth:`follow` builds one."""
        self._rust_player = None
        self._log = None

    def legal(self) -> list:
        return [int(i) for i in self._rust_player.legal_action_indices_at_root()]

    def adaptive_readouts(self) -> int:
        if len(self.legal()) <= int(self.config.adaptive_readout_threshold):
            return int(self.config.min_readouts)
        return int(self.config.num_readouts)

    def start_search(self, readouts: Optional[int] = None, key=None):
        """This move's search (``rust_mcts_player.RustMCTSPlayer.start_search``): None at a single legal move with
        no open price; at a single move whose price is open, the root's evaluation alone (its price head prices
        it)."""
        from rl18xx.agent.alphazero.lockstep import TreeSearch

        legal = self.legal()
        if len(legal) == 1:
            if not self._rust_player.root_move_has_open_price(legal[0]):
                return None
            readouts = 0
        return TreeSearch(
            self,
            readouts if readouts is not None else self.adaptive_readouts(),
            key=key,
            budget=getattr(self.config, "readout_budget", "additional"),
        )

    def visits(self) -> dict:
        """Root visits per legal index."""
        tree = self._rust_player
        stats = tree.root_target_stats()
        return {int(i): float(n) for i, n in zip(stats["legal"], stats["child_n"])}

    def pick_move(self, allowed: Optional[set] = None) -> int:
        """The move to play: the most visited (PUCT-tie-broken, ``pick_best_action``) once ``softpick_move_cutoff``
        decisions are played, a visit-proportional sample before; a Gumbel root's own move. ``allowed``: legal
        indices the move must come from (the most visited of them; the bots' Ruby dead ends)."""
        tree = self._rust_player
        legal = self.legal()
        if allowed is not None and not set(legal) <= set(allowed):
            choices = [i for i in legal if i in allowed] or legal
            visits = self.visits()
            return max(choices, key=lambda i: (visits.get(i, 0.0), -choices.index(i)))
        if len(legal) > 1 and bool(tree.gumbel_enabled) and bool(tree.gumbel_search_active):
            return int(tree.gumbel_action())
        if self.decisions >= int(self.config.softpick_move_cutoff) or len(legal) == 1:
            return int(tree.pick_best_action())
        visits = self.visits()
        counts = np.array([visits.get(i, 0.0) for i in legal], dtype=np.float64)
        if counts.sum() <= 0:
            return legal[random.randrange(len(legal))]
        return legal[int(np.searchsorted(np.cumsum(counts / counts.sum()), random.random()).clip(0, len(legal) - 1))]

    def play(self, index: int) -> list:
        """Commit root move ``index`` (an open price: the most visited price grandchild) and return the action
        dicts the tree logged for it: the move, then the forced moves after it."""
        tree = self._rust_player
        if len(self.legal()) > 1:
            self.decisions += 1
        before = int(tree.root_move_number())
        tree.advance_root(int(index), False)
        actions = [dict(a) for a in list(tree.root_game_object().raw_actions)[before:]]
        self._synced = int(tree.root_move_number())
        self._log = None
        return actions


def search_position(
    player: GenericSearchPlayer, raw, evaluator, key="main", readouts: Optional[int] = None, ignore: tuple = ()
):
    """Search ``raw``'s position in-process (one tree, its own ``LockstepSearcher`` round after round): the tree
    kept when it still matches (``GenericSearchPlayer.follow``), else a fresh one. Returns the search's
    ``TreeSearch`` (None: a forced move)."""
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher

    player.follow(raw, ignore)
    search = player.start_search(readouts=readouts, key=key)
    if search is not None:
        LockstepSearcher({key: evaluator}).run([search])
    return search


__all__ = [
    "BOT_READOUTS",
    "BOT_SEARCH",
    "GenericSearchPlayer",
    "SearchSpec",
    "local_evaluator",
    "parse_search_spec",
    "search_config",
    "search_options",
    "search_position",
]
