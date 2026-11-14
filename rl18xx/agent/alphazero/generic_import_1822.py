"""The 1822 family's human-game importer for the title-generic supervised data (1822, 1822PNW, 1822MX,
1822CA; ``docs/rl_model.md`` "1822 family: data, starts, caps and the CPU smoke").

**Matching** (shared with ``scripts/g1822_completeness.py``, which imports it from here): a recorded human
action matches a generated move on its type, its entity and the parameters that change the outcome
(company / corporation / bundle holder and size / hex, tile, rotation and combo partner / city / train type,
seller, depot variant and exchange / choice / kind / assignment target), a ranged parameter (a bid, the price
of another corporation's train, a partial debt payoff's amount) by range *and step* membership.
``run_routes`` matches the mover's one ``RunRoutes`` index by type (the route is the router check's
business; the record replays the human's own routes). The layout is injective per state
(``rl_legal_moves_json`` raises otherwise), so exactly one matching move is exactly one layout index.

**Importer** (:class:`Importer1822Family`, ``generic_data.register_importer(<title>, ...)``): raw
18xx.games exports (``human_games/1822``, ``human_games/scrape_2026_10/<title>``) replayed as the corpus
audits replay them -- Ruby's filtered actions, recorded auto-actions processed right after their parent,
seats renumbered 1001..1000+N ("Player k"; the family's minors have numeric syms, ``SEAT_ID_BASE``), the
engine seed from ``settings.seed`` / the game id. Scope: finished or archived games of the title, at the
pipeline's player counts (``titles.TitleSpec``: 1822 / 1822CA 3-6, 1822PNW / 1822MX 3-5), whose optional rules
are the title's ``TitleSpec.optional_rules`` (1822: 1822+, its No Removals / Single Stack modifiers and the
multi-share tax haven -- any combination --; 1822PNW / 1822MX / 1822CA: base rules only); dropped and counted:
``title``, ``status_<s>``, ``players_<n>``, ``optional_rules``, ``no_actions``. A game is replayed and recorded
under its canonical rules (``TitleSpec.canonical_rules``: a modifier without 1822+ does nothing and is dropped);
the record carries them (``optional_rules``), their combination name (``rules``, ``TitleSpec.rules_name``) and
the export's own list (``settings_optional_rules``).

Every human decision (auto-actions and actions after the game's end are not decisions) is matched against
the **rules'** moves, as A4's index completeness (``scripts/g1822_completeness.py --index``); the position's
legal list is the **title's cross-player policy**'s (``RESTRICTED``: what self-play and the evals are
offered), so a human move the policy excludes (a train from another player's corporation, a private or a
minor another player owns) is skipped as ``policy``. The policy only removes ``buy_train``,
``acquire_company`` and ``merge`` moves (``g1822_cross_player_excluded``; when it would leave nothing it
returns the rules' set), so for every other action type the policy's moves of that type *are* the rules'
moves of that type, in the same order: the importer matches those on the policy's list and asks the
generator for the rules' list only for the three policy types (one generator run per decision instead of
two; ``test_policy_list_matching_equals_rules_matching``). Misses: ``replay_only:out_of_turn`` (no rules
move of the human's entity: Ruby lets a crowded corporation other than the first discard, the tax-haven
owner act out of turn, a corporation that is not operating run routes; ``g1822_remaining_work.md``
"Replay-only"), ``replay_only:reacquire_owned`` (acquiring a company the corporation already owns), other
misses ``unmatched``; several matches ``ambiguous``. Finished games only (``supervised_finished_only``, the
user's rule for the new titles).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from rl18xx.agent.alphazero.generic_data import TitleImporter, new_engine_game

FAMILY = ("1822", "1822PNW", "1822MX", "1822CA")
#: Seat ids start above the minors' numeric syms (``tests/title_replay_harness.SEAT_ID_BASE``).
SEAT_ID_BASE = 1000
#: Action types matched to the mover's RunRoutes index by type alone.
TYPE_ONLY = {"run_routes"}
#: Action types the cross-player policy (``GameTitle::generator_cross_player``) can exclude.
POLICY_TYPES = {"buy_train", "acquire_company", "merge"}
#: Game statuses in scope: finished games and the site's archive of them.
STATUSES = ("finished", "archived")


# ---------------------------------------------------------------------------------------------------
# Matching (scripts/g1822_completeness.py imports these)
# ---------------------------------------------------------------------------------------------------


def _in_range(value, rng) -> bool:
    if rng is None or value is None:
        return False
    lo, hi, step = rng
    return lo <= value <= hi and (value - lo) % max(step, 1) == 0


def _shares_key(g, a: dict):
    ids = list(a.get("shares") or [])
    owners = sorted(g.g1822_share_owner(s) for s in ids)
    corp = ids[0].rsplit("_", 1)[0] if ids else a.get("corporation")
    pres = any(s.endswith("_0") for s in ids)
    return (corp, tuple(owners), int(a.get("percent") or 0), pres)


def action_key(g, a: dict):
    """The outcome-relevant parameters of a wire action (state BEFORE it)."""
    t = a["type"]
    if t == "bid":
        return (a.get("company"),)
    if t in ("buy_shares", "sell_shares"):
        return _shares_key(g, a)
    if t == "par":
        sp = a.get("share_price")
        if isinstance(sp, str):
            sp = int(sp.split(",")[0])
        return (a.get("corporation"), sp)
    if t == "lay_tile":
        tile = str(a.get("tile")).split("-")[0]
        return (
            a.get("hex"),
            tile,
            # (a builder cube's rotation changes nothing)
            0 if tile == "BC" else int(a.get("rotation") or 0),
            tuple(sorted(a.get("combo_entities") or [])),
        )
    if t in ("place_token", "remove_token"):
        # (1822MX's NdeM token auction names the token by its city)
        city = a.get("city")
        return tuple(g.g1822_city_ref(city)) if city else (a.get("hex"),)
    if t == "hex_token":
        return (a.get("hex"),)
    if t == "dividend":
        return (a.get("kind"),)
    if t in ("choose", "choose_ability"):
        return (str(a.get("choice")),)
    if t == "buy_train":
        name, owner = g.g1822_train_info(a.get("train", ""))
        return (name, owner, a.get("variant") if owner == "depot" else None, bool(a.get("exchange")))
    if t == "discard_train":
        return (g.g1822_train_info(a.get("train", ""))[0],)
    if t == "merge":
        return (a.get("corporation"),)
    if t == "acquire_company":
        return (a.get("company"),)
    if t == "assign":
        return (a.get("target"), a.get("target_type"))
    return ()


def action_price(a: dict):
    """The price a human action chose: a bid's, a train bought from another corporation's, a partial debt
    payoff's ``amount``; None for other types."""
    t = a["type"]
    if t == "bid":
        return a.get("price")
    if t == "payoff_player_debt_partial":
        return a.get("amount")
    if t == "buy_train":
        return a.get("price")
    return None


def _matches(g, t: str, ent: str, hk, hp, m: dict) -> bool:
    w = m["wire"]
    if w["type"] != t or str(w.get("entity")) != ent:
        return False
    if action_key(g, w) != hk:
        return False
    if m["range"] is not None:
        return _in_range(hp, m["range"])
    if t == "buy_train" and hp is not None and w.get("price") != hp:
        return False
    if t == "payoff_player_debt_partial":
        return False
    return True


def matching(g, human: dict, moves: list) -> list:
    """The generated moves (``g1822_legal_moves_json`` / ``rl_legal_moves_json`` entries) the human action
    matches (``run_routes``: see :func:`matching_indexed`)."""
    t = human["type"]
    ent = str(human.get("entity"))
    hk = action_key(g, human)
    hp = action_price(human)
    return [m for m in moves if _matches(g, t, ent, hk, hp, m)]


def matches(g, human: dict, moves: list) -> int:
    """How many generated moves the human action matches (:func:`matching`)."""
    return len(matching(g, human, moves))


def matching_indexed(g, human: dict, moves: list) -> list:
    """:func:`matching` over indexed moves (``rl_legal_moves_json``), with ``run_routes`` matched to the
    mover's RunRoutes move by type and entity (A4's index completeness)."""
    if human["type"] in TYPE_ONLY:
        ent = str(human.get("entity"))
        return [m for m in moves if m["wire"]["type"] == human["type"] and str(m["wire"].get("entity")) == ent]
    return matching(g, human, moves)


def replay_only_reason(g, human: dict, rules_moves: list) -> Optional[str]:
    """A missed human action in a documented replay-only class (``docs/g1822_remaining_work.md``
    "Replay-only"), else None: ``out_of_turn`` when the rules offer the human's entity nothing (another
    crowded corporation's discard, the tax-haven owner out of turn, a run by a corporation that is not
    operating), ``reacquire_owned`` for acquiring a company the acquirer already owns."""
    ent = str(human.get("entity"))
    if not any(str(m["wire"].get("entity")) == ent for m in rules_moves):
        return "out_of_turn"
    if human["type"] == "acquire_company":
        company = str(human.get("company"))
        for co in g.companies:
            if co.sym == company and str(co.owner).split(":")[-1] == ent:
                return "reacquire_owned"
    return None


# ---------------------------------------------------------------------------------------------------
# Scope and replay of a raw export
# ---------------------------------------------------------------------------------------------------


def player_range(title: str) -> Tuple[int, int]:
    from rl18xx.agent.alphazero.titles import title_spec

    spec = title_spec(title)
    return spec.min_players, spec.max_players


def scope_reason(game: dict, title: str, allow_optional_rules: bool = False) -> Optional[str]:
    """Why a raw export is outside the importer's scope (None: in scope). Reasons are the drop keys of the
    conversion's stats: ``title``, ``status_<status>``, ``players_<n>``, ``optional_rules`` (a rule outside the
    title's ``TitleSpec.optional_rules``; ``allow_optional_rules`` admits any), ``no_actions``."""
    if str(game.get("title")) != str(title):
        return "title"
    status = game.get("status")
    if status not in STATUSES:
        return f"status_{status}"
    n = len(game.get("players") or [])
    lo, hi = player_range(title)
    if not lo <= n <= hi:
        return f"players_{n}"
    rules = (game.get("settings") or {}).get("optional_rules") or ()
    if rules and not allow_optional_rules:
        from rl18xx.agent.alphazero.titles import title_spec

        if not title_spec(title).rules_in_scope(rules):
            return "optional_rules"
    if not game.get("actions"):
        return "no_actions"
    return None


def seat_players(n: int) -> Dict[int, str]:
    return {SEAT_ID_BASE + i + 1: f"Player {i + 1}" for i in range(n)}


def replay_export(game: dict, title: str, optional_rules=()) -> Tuple[list, List[bool], Optional[str], object]:
    """Replay a raw 18xx.games export as ``tests/title_replay_harness.replay_game`` does, flattened:
    ``(engine action log, per logged action whether it was a recorded auto-action, the rejected action's
    description or None, the engine game)``. A rejected action stops the replay; the log holds every action
    before it (the engine logs each processed action once, as processed)."""
    from tests.title_replay_harness import game_seed, ruby_filtered_actions

    n = len(game["players"])
    seat = {p["id"]: SEAT_ID_BASE + i + 1 for i, p in enumerate(game["players"])}
    raw = new_engine_game(title, seat_players(n), tuple(optional_rules), game_seed(game))
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


class Importer1822Family(TitleImporter):
    """1822-family human games (raw 18xx.games exports) -> generic records (module docstring)."""

    def __init__(self, title: str, legal_policy: Optional[str] = None, allow_optional_rules: bool = False):
        if str(title) not in FAMILY:
            raise ValueError(f"{title!r} is not an 1822-family title ({FAMILY})")
        self.title = str(title)
        #: The policy of the positions' legal lists (None: the title's cross-player policy).
        self.legal_policy = legal_policy
        #: Keep games with any optional rules (measurements only; the pipeline's scope is the title's
        #: ``TitleSpec.optional_rules``).
        self.allow_optional_rules = bool(allow_optional_rules)
        self._auto: List[bool] = []
        self._stopped: Optional[str] = None
        self._finished = False
        #: The current position's moves under ``legal_policy`` (``legal_actions`` -> ``match``).
        self._moves: Optional[list] = None

    def players(self, game: dict) -> Dict[int, str]:
        return seat_players(len(game.get("players") or []))

    def optional_rules(self, game: dict) -> List[str]:
        """The rules a game is replayed and recorded under: its export's, canonical (``TitleSpec.canonical_rules``;
        with ``allow_optional_rules`` rules the pipeline does not play are kept as given)."""
        from rl18xx.agent.alphazero.titles import title_spec

        rules = (game.get("settings") or {}).get("optional_rules") or []
        return list(title_spec(self.title).canonical_rules(rules, strict=False)) if rules else []

    def seed(self, game: dict) -> Optional[int]:
        from tests.title_replay_harness import game_seed

        return int(game_seed(game))

    def load(self, game: dict) -> Tuple[Optional[list], Optional[str]]:
        self._auto, self._stopped, self._finished, self._moves = [], None, False, None
        reason = scope_reason(game, self.title, self.allow_optional_rules)
        if reason is not None:
            return None, reason
        try:
            actions, auto, stopped, raw = replay_export(game, self.title, self.optional_rules(game))
        except ValueError as exc:  # the engine refuses an optional rule
            return None, f"unsupported_rules: {str(exc)[:120]}"
        if not actions:
            return None, "replay_rejected_at_start" if stopped else "no_actions"
        self._auto, self._stopped, self._finished = auto, stopped, bool(raw.finished)
        return actions, None

    def unfinished_reason(self, game: dict) -> Optional[str]:
        """The loaded log stops at a rejected action, or ends before the game's end; None when it reaches
        the end (``finished_only`` conversions)."""
        if self._stopped:
            return f"unfinished_replay_rejected: {self._stopped}"
        return None if self._finished else "unfinished_record_ends"

    def is_decision(self, state, pos: int, action: dict) -> Tuple[bool, str]:
        """Recorded auto-actions (programmed passes) and actions after the game's end are not decisions."""
        if pos < len(self._auto) and self._auto[pos]:
            return False, "auto"
        if self.raw(state).finished:
            return False, "after_end"
        return True, ""

    def legal_actions(self, state):
        from rl18xx.agent.alphazero import model_generic as mg

        g = self.raw(state)
        self._moves = json.loads(g.rl_legal_moves_json(self.legal_policy))
        return mg.legal_actions(g, policy=self.legal_policy, moves=self._moves)

    def match(self, state, action: dict, legal_indices: np.ndarray) -> Tuple[Optional[int], str]:
        g = self.raw(state)
        moves = self._moves if self._moves is not None else json.loads(g.rl_legal_moves_json(self.legal_policy))
        if self.legal_policy != "rules" and action["type"] in POLICY_TYPES:
            moves = json.loads(g.rl_legal_moves_json("rules"))
        hits = matching_indexed(g, action, moves)
        if not hits:
            if self.legal_policy != "rules" and action["type"] not in POLICY_TYPES:
                moves = json.loads(g.rl_legal_moves_json("rules"))  # (a miss is rare: the rules' entities)
            reason = replay_only_reason(g, action, moves)
            return None, f"replay_only:{reason}" if reason else "unmatched"
        if len(hits) > 1:
            return None, "ambiguous"
        index = int(hits[0]["index"])
        if index not in set(int(i) for i in legal_indices):
            return None, "policy"
        return index, "ok"

    def apply(self, state, action: dict) -> None:
        self._moves = None
        state.process_action(action)

    def price(self, action: dict) -> Optional[int]:
        price = action_price(action)
        return None if price is None else int(price)

    def record_extra(self, game: dict) -> dict:
        from rl18xx.agent.alphazero.titles import rules_name

        return {
            "stopped": self._stopped,
            "rules": rules_name(self.title, self.optional_rules(game)),
            "settings_optional_rules": list((game.get("settings") or {}).get("optional_rules") or []),
        }


def register() -> None:
    from rl18xx.agent.alphazero.generic_data import register_importer

    for title in FAMILY:
        register_importer(title, lambda title=title: Importer1822Family(title))


__all__ = [
    "FAMILY",
    "Importer1822Family",
    "POLICY_TYPES",
    "SEAT_ID_BASE",
    "TYPE_ONLY",
    "action_key",
    "action_price",
    "matches",
    "matching",
    "matching_indexed",
    "register",
    "replay_export",
    "replay_only_reason",
    "scope_reason",
    "seat_players",
]
