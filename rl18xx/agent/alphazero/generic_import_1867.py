"""1867's human-game importer for the title-generic supervised data (RL bridge milestone A3;
``docs/rl_model.md`` "1867: data, starts, caps, smoke").

**Matching** (shared with ``scripts/g1867_completeness.py``, which imports it from here): a recorded human
action matches a generated move on its type, its entity and the parameters that change the outcome
(company / corporation / bundle holder and size / hex, tile and rotation / city / train type and seller /
choice / kind), a ranged parameter (bid, company price, price for another corporation's train) by range
*and step* membership. ``run_routes`` matches the one ``RunRoutes`` index by type (the route itself is the
router check's business; the record replays the human's own routes). The layout is injective per state
(``rl_legal_moves_json`` raises otherwise), so exactly one matching move is exactly one layout index.

**Importer** (:class:`Importer1867`, ``generic_data.register_importer("1867", ...)``): raw 18xx.games exports
(``human_games/1867``, ``human_games/scrape_2026_10/1867``) replayed as the corpus audits replay them --
Ruby's filtered actions, recorded auto-actions processed right after their parent, seats renumbered 1..N
("Player k"; no names kept), the engine seed from ``settings.seed`` / the game id. Scope as the generator's
and A2's completeness: finished or archived games, 3-6 players, base rules; 2-player games, ``grid_market``
(or any other optional rule) and abandoned (``active`` / ``new``) games are dropped and counted. Every
human decision (auto-actions and actions after the game's end are not decisions) is matched against the
**rules'** moves (``rl_legal_moves_json("rules")``, as A2's index completeness); the position's legal list
is the **title's cross-player policy**'s (``model_generic.legal_actions``: what self-play and the evals are
offered), so a human move the policy excludes (a train / private bought from another player, a merger add
of another player's minor) is skipped and counted as ``policy``. Misses in a documented replay-only class
(``g1867_remaining_work.md`` "Replay-only") count as ``replay_only:<class>``, other misses ``unmatched``,
several matches ``ambiguous``. A game the replay rejects part-way is kept up to the rejection (its decisions
are human decisions of a legal position; ``finished`` False, ``stopped`` names the rejected action), so its
labels are exactly A2's completeness decisions.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rl18xx.agent.alphazero.generic_data import TitleImporter, new_engine_game

#: Action types the replay processes that are matched to the RunRoutes index by type alone.
TYPE_ONLY = {"run_routes"}
#: Action types the cross-player policy (``GameTitle::generator_cross_player``) can exclude.
POLICY_TYPES = {"buy_train", "buy_company", "merge"}
#: Game statuses in scope (A2's completeness): finished games and the site's archive of them.
STATUSES = ("finished", "archived")
PLAYERS = (3, 6)


# ---------------------------------------------------------------------------------------------------
# Matching (scripts/g1867_completeness.py imports these)
# ---------------------------------------------------------------------------------------------------


def _in_range(value, rng) -> bool:
    if rng is None or value is None:
        return False
    lo, hi, step = rng
    return lo <= value <= hi and (value - lo) % max(step, 1) == 0


def _shares_key(g, a: dict):
    ids = list(a.get("shares") or [])
    owners = sorted(g.g1867_share_owner(s) for s in ids)
    corp = ids[0].rsplit("_", 1)[0] if ids else a.get("corporation")
    pres = any(s.endswith("_0") for s in ids)
    return (corp, tuple(owners), int(a.get("percent") or 0), pres)


def action_key(g, a: dict):
    """The outcome-relevant parameters of a wire action (state BEFORE it)."""
    t = a["type"]
    if t == "bid":
        return (a.get("company"), a.get("corporation"))
    if t in ("buy_shares", "sell_shares"):
        return _shares_key(g, a)
    if t == "par":
        sp = a.get("share_price")
        if isinstance(sp, str):
            sp = int(sp.split(",")[0])
        return (a.get("corporation"), sp)
    if t == "lay_tile":
        return (a.get("hex"), str(a.get("tile")).rsplit("-", 1)[0], int(a.get("rotation") or 0))
    if t == "place_token":
        city = a.get("city")
        return tuple(g.g1867_city_ref(city)) if city else (a.get("hex"),)
    if t == "remove_token":
        return tuple(g.g1867_city_ref(a.get("city"))) + (int(a.get("slot") or 0),)
    if t == "dividend":
        return (a.get("kind"),)
    if t == "choose":
        return (str(a.get("choice")),)
    if t == "buy_train":
        name, owner = g.g1867_train_info(a.get("train", ""))
        exch = a.get("exchange")
        return (name, owner, g.g1867_train_info(exch)[0] if exch else None)
    if t == "discard_train":
        return (g.g1867_train_info(a.get("train", ""))[0],)
    if t == "merge":
        return (a.get("corporation"),)
    if t == "buy_company":
        return (a.get("company"),)
    return ()


def action_price(a: dict):
    return a.get("price") if a["type"] in ("bid", "buy_train", "buy_company") else None


def matching(g, human: dict, moves: list) -> list:
    """The generated moves (``g1867_legal_moves_json`` / ``rl_legal_moves_json`` entries) the human action
    matches (``run_routes``: not handled here, see :func:`matching_indexed`)."""
    t = human["type"]
    ent = str(human.get("entity"))
    hk = action_key(g, human)
    hp = action_price(human)
    out = []
    for m in moves:
        w = m["wire"]
        if w["type"] != t or str(w.get("entity")) != ent:
            continue
        if action_key(g, w) != hk:
            continue
        if m["range"] is not None:
            if not _in_range(hp, m["range"]):
                continue
        elif hp is not None and w.get("price") != hp:
            continue
        out.append(m)
    return out


def matching_indexed(g, human: dict, moves: list) -> list:
    """:func:`matching` over indexed moves (``rl_legal_moves_json``), with ``run_routes`` matched to the
    mover's RunRoutes move by type (A2's index completeness)."""
    if human["type"] in TYPE_ONLY:
        return [
            m
            for m in moves
            if m["wire"]["type"] == human["type"] and str(m["wire"].get("entity")) == str(human.get("entity"))
        ]
    return matching(g, human, moves)


def replay_only_reason(g, human: dict) -> Optional[str]:
    """A missed human action in a documented replay-only class (``docs/g1867_remaining_work.md``
    "Replay-only"), else None."""
    offers = g.g1867_step_offers()
    if not offers:
        return None
    kind, ent, _types = offers[0]
    t, who = human["type"], str(human.get("entity"))
    if kind == "MajorTrainless" and t in ("choose", "pass") and who != ent:
        return "trainless_out_of_order"  # Ruby accepts any queued major; we decide in queue order
    if kind == "PostMergerShares" and t in ("buy_shares", "pass") and who != ent:
        return "dealing_out_of_turn"  # Ruby's actions ignore the dealing order
    if kind == "ReduceTokens" and t in ("buy_shares", "pass") and human.get("entity_type") == "player":
        return "pre_9655_merger_order"  # old records deal shares before removing tokens
    return None


# ---------------------------------------------------------------------------------------------------
# Replay of a raw export
# ---------------------------------------------------------------------------------------------------


def scope_reason(game: dict) -> Optional[str]:
    """Why a raw export is outside the importer's scope (None: in scope). Reasons are the drop keys of
    the conversion's stats: ``title``, ``status_<status>``, ``players_<n>``, ``grid_market``,
    ``optional_rules``, ``no_actions``."""
    if str(game.get("title")) != "1867":
        return "title"
    status = game.get("status")
    if status not in STATUSES:
        return f"status_{status}"
    n = len(game.get("players") or [])
    if not PLAYERS[0] <= n <= PLAYERS[1]:
        return f"players_{n}"
    rules = list((game.get("settings") or {}).get("optional_rules") or [])
    if "grid_market" in rules:
        return "grid_market"
    if rules:
        return "optional_rules"
    if not game.get("actions"):
        return "no_actions"
    return None


def replay_export(game: dict) -> Tuple[list, List[bool], Optional[str], object]:
    """Replay a raw 18xx.games export as ``tests/title_replay_harness.replay_game`` does, flattened:
    ``(engine action log, per logged action whether it was a recorded auto-action, the rejected action's
    description or None, the engine game)``. A rejected action stops the replay; the log holds every
    action before it (the engine logs each processed action once, as processed)."""
    from tests.title_replay_harness import game_seed, ruby_filtered_actions

    n = len(game["players"])
    seat = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    raw = new_engine_game("1867", {i + 1: f"Player {i + 1}" for i in range(n)}, (), game_seed(game))
    actions = json.loads(json.dumps(game["actions"]))
    for a in actions:
        a["_fid"] = a.get("id")
        for au in a.get("auto_actions") or []:
            au["_fid"] = a.get("id")
    auto_flags: List[bool] = []
    stopped = None
    for a in ruby_filtered_actions(actions, mark_auto=True):
        fid = a.pop("_fid", None)
        auto = bool(a.pop("_auto", False))
        if a.get("entity_type") == "player":
            a["entity"] = a["user"] = seat.get(a["entity"], a["entity"])
        before = len(raw.raw_actions)
        try:
            raw.process_action(a)
        except BaseException as exc:  # noqa: BLE001 (pyo3 PanicException included)
            stopped = f"{a.get('type')} at file id {fid}: {str(exc)[:160]}"
            break
        logged = len(raw.raw_actions) - before
        if logged != 1:
            raise RuntimeError(f"file id {fid}: the engine logged {logged} actions for one")
        auto_flags.append(auto)
    return [dict(a) for a in raw.raw_actions], auto_flags, stopped, raw


# ---------------------------------------------------------------------------------------------------
# The importer
# ---------------------------------------------------------------------------------------------------


class Importer1867(TitleImporter):
    """1867 human games (raw 18xx.games exports) -> generic records (module docstring)."""

    title = "1867"

    def __init__(self, legal_policy: Optional[str] = None):
        #: The policy of the positions' legal lists (None: the title's cross-player policy).
        self.legal_policy = legal_policy
        self._game_id: Optional[str] = None
        self._auto: List[bool] = []
        self._stopped: Optional[str] = None

    def players(self, game: dict) -> Dict[int, str]:
        n = len(game.get("players") or [])
        return {i + 1: f"Player {i + 1}" for i in range(n)}

    def seed(self, game: dict) -> Optional[int]:
        from tests.title_replay_harness import game_seed

        return int(game_seed(game))

    def load(self, game: dict) -> Tuple[Optional[list], Optional[str]]:
        self._game_id, self._auto, self._stopped = str(game.get("id")), [], None
        reason = scope_reason(game)
        if reason is not None:
            return None, reason
        actions, auto, stopped, _ = replay_export(game)
        if not actions:
            return None, "replay_rejected_at_start" if stopped else "no_actions"
        self._auto, self._stopped = auto, stopped
        return actions, None

    def is_decision(self, state, pos: int, action: dict) -> Tuple[bool, str]:
        """Auto-actions (programmed passes, the BuyCompanyPreloan auto-pass) and actions after the game's
        end are not decisions."""
        if pos < len(self._auto) and self._auto[pos]:
            return False, "auto"
        if self.raw(state).finished:
            return False, "after_end"
        return True, ""

    def legal_actions(self, state):
        from rl18xx.agent.alphazero import model_generic as mg

        return mg.legal_actions(self.raw(state), policy=self.legal_policy)

    def match(self, state, action: dict, legal_indices: np.ndarray) -> Tuple[Optional[int], str]:
        g = self.raw(state)
        moves = json.loads(g.rl_legal_moves_json("rules"))
        hits = matching_indexed(g, action, moves)
        if not hits:
            reason = replay_only_reason(g, action)
            return None, f"replay_only:{reason}" if reason else "unmatched"
        if len(hits) > 1:
            return None, "ambiguous"
        index = int(hits[0]["index"])
        if index not in set(int(i) for i in legal_indices):
            return None, "policy"
        return index, "ok"

    def record_extra(self, game: dict) -> dict:
        return {"stopped": self._stopped}


def register() -> None:
    from rl18xx.agent.alphazero.generic_data import register_importer

    register_importer("1867", Importer1867)


__all__ = [
    "Importer1867",
    "POLICY_TYPES",
    "TYPE_ONLY",
    "action_key",
    "action_price",
    "matching",
    "matching_indexed",
    "register",
    "replay_export",
    "replay_only_reason",
    "scope_reason",
]
