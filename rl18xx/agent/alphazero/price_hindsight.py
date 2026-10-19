"""Hindsight price learning: counterfactual price branching on finished games.

A human learns prices ten moves later -- "had I paid myself $5 more for that
private, I could buy the share I want now". A search can't: the consequence is
past its horizon, the value is too blunt to see $5, and training only sees the
price that was played. The engine is deterministic, so here the game is rewound:

1. **Constraint events** (:func:`scan_game`, :func:`unaffordable_events`) in a
   finished game, each at the state before action ``i``:

   - ``unaffordable``: the acting entity (a player in a stock round, a
     corporation in an operating round) with ``probe_cash`` more dollars
     (``BaseGame.counterfactual_transfer_cash`` on a clone) has new legal moves,
     and the policy at that richer state puts its top move, with probability >=
     ``p_wanted``, on one of them -- a move it wants and can't afford. The
     shortfall is the least extra cash that makes that move legal (bisection on
     the engine's legal-move enumeration; "unaffordable" is told apart from
     other illegality by construction: only cash changed).
   - ``emr_sale``: a president selling shares in an operating round to fund
     the corporation's forced train (emergency money raising); the shortfall is
     what the sales raised.
   - ``president_contribution``: a train bought for more than the
     corporation's cash (the president paid the rest); the shortfall is the
     difference.
   - ``bankrupt``: a bankruptcy.
   - ``bid_lost``: in an auction played by the agents, a company won by
     another player for at most ``bid_margin`` more than this player's highest
     bid on it.

2. **Back-tracing** (:func:`link_events`): the price decisions -- the price
   head's slots: auction bids, a corporation buying a private, cross-corporation
   train purchases (:func:`price_decision`) -- within ``window`` decisions
   before the event whose money went to or came from the constrained entity
   (for the operating-round emergencies also its president's pocket). Each
   link records which way the price would have helped (payee: higher; payer:
   lower). Categorical money decisions (par prices, dividends) are recorded too
   and can be branched as alternative moves, but only the price head is trained.

3. **Branching** (:class:`BranchRunner`): from the state before the decision, the
   played price and alternatives -- the played price +- ``deltas`` and the legal
   range's ends, clipped and de-duplicated (:func:`price_alternatives`) -- are
   applied and continued two ways:

   - **open loop**: the game's own later actions are replayed up to the event
     where they are still legal (skipped where not); at the event the
     constrained entity's cash, whether the wanted move became legal, and the
     value of the position are recorded;
   - **closed loop**: ``rollouts`` policy-only continuations to the game's end
     (every seat the rollout policy at ``temperature``, prices from its price
     head), with **common random numbers** -- continuation ``r`` of every
     alternative draws from the same seed, so branches only part where the price
     changed something. Each rollout records the final win share, net-worth share
     and placement points of every seat, and the value network's estimate for
     the deciding player at ``horizons`` decisions after the branch.

4. The per-alternative paired differences against the played price give an
   advantage over the price head's cells (:func:`cell_advantages`) and an
   advantage-weighted target for the head (:func:`price_cell_target_from_advantages`).

``scripts/price_hindsight.py`` drives it: ``detect`` -> ``branch`` -> ``report``
-> ``finetune``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

NUM_PLAYERS = 4
PRICE_TYPES = ("bid", "buy_company", "buy_train")  # the price head's action types
MONEY_TYPES = ("par", "dividend")  # categorical money decisions (policy head)
EVENT_KINDS = ("unaffordable", "emr_sale", "president_contribution", "bankrupt", "bid_lost")
DEFAULT_DELTAS = (-20, -10, -5, -1, 1, 5, 10, 20)
DEFAULT_HORIZONS = (0, 10, 30, 60, 100)
OUTCOME_KEYS = ("win", "worth", "place")


# ---------------------------------------------------------------------------
# Entities and money
# ---------------------------------------------------------------------------


def player_seats(rs) -> dict:
    """Player id -> seat (players sorted by id: the order of outcomes and values)."""
    return {pid: seat for seat, pid in enumerate(sorted(int(p.id) for p in rs.players))}


def norm_entity(entity) -> str:
    """``"player:<id>"`` / ``"corp:<sym>"`` for an action's entity (an int player id,
    a numeric string, a corporation symbol, or an already prefixed id)."""
    if isinstance(entity, int):
        return f"player:{entity}"
    s = str(entity)
    if s.startswith(("player:", "corp:")):
        return s
    return f"player:{s}" if s.isdigit() else f"corp:{s}"


def controlling_player(rs, entity: Optional[str]) -> Optional[int]:
    """The player deciding for ``entity``: the player itself, or a corporation's president."""
    if not entity:
        return None
    entity = norm_entity(entity)
    if entity.startswith("player:"):
        return int(entity.split(":", 1)[1])
    corp = rs.corporation_by_id(entity.split(":", 1)[1])
    owner = corp.owner_id_str if corp is not None else ""
    return int(owner.split(":", 1)[1]) if owner.startswith("player:") else None


def mover_seat(rs) -> int:
    """The acting player's seat -- the encoder's rotation (``BaseGame.mover_index``):
    the active player, or in an operating round the active corporation's president."""
    pid = controlling_player(rs, rs.current_entity_id)
    return player_seats(rs).get(pid, 0) if pid is not None else 0


def entity_cash(rs, entity: str) -> int:
    entity = norm_entity(entity)
    if entity.startswith("player:"):
        pid = int(entity.split(":", 1)[1])
        return int(next((p.cash for p in rs.players if int(p.id) == pid), 0))
    corp = rs.corporation_by_id(entity.split(":", 1)[1])
    return int(corp.cash) if corp is not None else 0


def train_owner(rs, train_id: str) -> Optional[str]:
    """``"corp:<sym>"`` of the corporation holding ``train_id`` (None: the depot)."""
    for corp in rs.corporations:
        for train in corp.trains:
            if train.id == train_id:
                return f"corp:{corp.sym}"
    return None


def value_for_seat(canonical_value, rotation: int, seat: int) -> float:
    """A canonical value vector (the mover first) read for ``seat``."""
    v = np.asarray(canonical_value, dtype=np.float64)
    return float(v[(seat - rotation) % NUM_PLAYERS])


# ---------------------------------------------------------------------------
# Decisions of a recorded game
# ---------------------------------------------------------------------------


def _matches(action: dict, decoded: dict) -> bool:
    kind = action["type"]
    if decoded.get("type") != kind:
        return False
    if kind in ("bid", "buy_company"):
        return decoded.get("company") == action.get("company")
    if kind == "buy_train":
        return decoded.get("train") == action.get("train")
    if kind == "par":
        return decoded.get("corporation") == action.get("corporation") and str(decoded.get("share_price")) == str(
            action.get("share_price", "")
        ).split(",")[0]
    if kind == "dividend":
        return decoded.get("kind") == action.get("kind")
    return False


def price_decision(rs, action: dict, legal: Iterable[int]) -> Optional[dict]:
    """The price-head decision a recorded ``action`` is, at ``rs`` (the state before
    it), or None: ``{index, price, slot, lo, hi, payer, payee}`` for an auction bid,
    a corporation buying a private, or a train bought from another corporation,
    where the price was open (``lo < hi``)."""
    kind = action.get("type")
    if kind not in PRICE_TYPES or action.get("price") is None:
        return None
    price = int(action["price"])
    for ix in legal:
        slot = rs.price_head_slot_for_index(int(ix))
        if slot is None:
            continue
        slot_index, lo, hi = int(slot[0]), int(slot[1]), int(slot[2])
        if lo >= hi:
            continue
        decoded = rs.decode_index_to_map(int(ix), price if lo <= price <= hi else lo)
        if not _matches(action, decoded):
            continue
        entity = norm_entity(action["entity"])
        if kind == "bid":
            payer, payee = entity, None
        elif kind == "buy_company":
            payer, payee = entity, norm_entity(rs.company_by_id(action["company"]).owner)
        else:
            payer, payee = entity, train_owner(rs, action["train"])
            if payee is None:
                return None  # a depot train has a fixed price
        return {
            "index": int(ix), "price": price, "slot": slot_index, "lo": lo, "hi": hi,
            "payer": payer, "payee": payee,
        }
    return None


def money_decision(rs, action: dict, legal: Iterable[int]) -> Optional[dict]:
    """A par or dividend choice: ``{index, alternatives}`` -- the legal indices of the
    same kind (the other par prices of the same corporation; payout / withhold)."""
    kind = action.get("type")
    if kind not in MONEY_TYPES:
        return None
    played, same_kind = None, []
    for ix in legal:
        decoded = rs.decode_index_to_map(int(ix), None)
        if decoded.get("type") != kind:
            continue
        if kind == "par" and decoded.get("corporation") != action.get("corporation"):
            continue
        same_kind.append(int(ix))
        if played is None and _matches(action, decoded):
            played = int(ix)
    if played is None or len(same_kind) < 2:
        return None
    return {"index": played, "alternatives": [ix for ix in same_kind if ix != played]}


def is_decision(rs, legal: list) -> bool:
    """Several legal moves, or one whose price is still open (``policy_selfplay._advance``)."""
    from rl18xx.agent.alphazero.policy_selfplay import has_price_choice

    return len(legal) > 1 or (len(legal) == 1 and has_price_choice(rs, legal[0]))


@dataclass
class Scan:
    """What one pass over a recorded game finds (action indices into its log)."""

    game_id: str
    n_actions: int
    first_agent_action: int  # actions before it are the start position's (not the agents')
    decision_index: dict  # action index -> running count of decision points before it
    decisions: list  # price / money decisions (dicts)
    events: list  # constraint events found from the log (dicts; "unaffordable" added later)
    probes: list  # (action index, entity, probe game, base legal, probe legal) for unaffordable detection
    error: Optional[str] = None


def scan_game(
    game_id: str,
    actions: list,
    first_agent_action: Optional[int] = 0,
    probe_cash: int = 100,
    bid_margin: int = 10,
    probe: bool = True,
) -> Scan:
    """Replay ``actions`` (a saved game's log) and collect its price / money
    decisions, the log-visible constraint events, and the cash probes for
    :func:`unaffordable_events`. Only actions from ``first_agent_action`` on are
    the agents' (a start position's prefix is not); None: from the first action
    outside the opening auction (a game started at the first Stock Round)."""
    from rl18xx.agent.alphazero.start_positions import new_game

    g = new_game(NUM_PLAYERS)
    rs = g._game
    scan = Scan(game_id, len(actions), first_agent_action if first_agent_action is not None else -1, {}, [], [], [])
    n_decisions = 0
    emr_open: Optional[dict] = None
    bids: dict = {}  # company -> {player: highest bid}
    try:
        for i, action in enumerate(actions):
            if g.finished:
                break
            legal = [int(x) for x in rs.factored_legal_indices()]
            decision = is_decision(rs, legal)
            scan.decision_index[i] = n_decisions
            round_type = rs.round.round_type
            kind = action.get("type")
            if scan.first_agent_action < 0 and round_type != "Auction":
                scan.first_agent_action = i
            agent = 0 <= scan.first_agent_action <= i
            entity = rs.current_entity_id
            mover = controlling_player(rs, entity)
            seats = player_seats(rs)
            base = {"i": i, "decision_number": n_decisions, "round": round_type, "entity": entity,
                    "player": mover, "seat": seats.get(mover) if mover is not None else None}

            if agent and decision:
                d = price_decision(rs, action, legal)
                if d is not None:
                    what = {k: action[k] for k in ("company", "train", "corporation") if k in action}
                    scan.decisions.append({**base, "kind": "price", "type": kind, **what, **d})
                else:
                    m = money_decision(rs, action, legal)
                    if m is not None:
                        scan.decisions.append({**base, "kind": kind, "type": kind, **m})
                if probe and round_type in ("Stock", "Operating"):
                    p = rs.pickle_clone()
                    p.counterfactual_transfer_cash(entity, int(probe_cash))
                    probe_legal = [int(x) for x in p.factored_legal_indices()]
                    new = sorted(set(probe_legal) - set(legal))
                    if new:
                        scan.probes.append((i, entity, p, legal, probe_legal, new, dict(base)))

            # Log-visible constraint events (the state before action i).
            if agent and kind == "sell_shares" and round_type == "Operating":
                player = norm_entity(action["entity"])
                if emr_open is None:
                    emr_open = {**base, "kind": "emr_sale", "constrained": [entity, player],
                                "player": int(player.split(":")[1]), "cash_before": entity_cash(rs, player)}
                    emr_open["seat"] = seats.get(emr_open["player"])
            elif emr_open is not None:
                player = f"player:{emr_open['player']}"
                emr_open["shortfall"] = int(entity_cash(rs, player) - emr_open.pop("cash_before"))
                scan.events.append(emr_open)
                emr_open = None
            if agent and kind == "buy_train" and round_type == "Operating":
                corp_cash = entity_cash(rs, entity)
                if int(action.get("price", 0)) > corp_cash:
                    scan.events.append({**base, "kind": "president_contribution",
                                        "constrained": [entity, f"player:{mover}"],
                                        "shortfall": int(action["price"]) - corp_cash})
            if agent and kind == "bankrupt":
                scan.events.append({**base, "kind": "bankrupt", "constrained": [entity, f"player:{mover}"],
                                    "shortfall": None})
            if kind == "bid" and round_type == "Auction":
                company = action.get("company")
                bids.setdefault(company, {})
                bidder = norm_entity(action["entity"])
                bids[company][bidder] = max(bids[company].get(bidder, 0), int(action.get("price", 0)))
                bids[company]["_last"] = i

            owners_before = {c.sym: c.owner for c in rs.companies} if round_type == "Auction" else None
            rs.process_action(dict(action))
            if decision:
                n_decisions += 1
            if owners_before is not None and agent:
                for c in rs.companies:
                    if owners_before.get(c.sym) != c.owner and str(c.owner).startswith("player:"):
                        won_by, price = c.owner, None
                        offers = bids.get(c.sym, {})
                        price = offers.get(won_by)
                        if price is None:
                            continue
                        for loser, best in offers.items():
                            if loser in ("_last", won_by) or best <= 0:
                                continue
                            if price - best <= bid_margin:
                                pid = int(loser.split(":")[1])
                                scan.events.append({
                                    "i": i + 1, "decision_number": n_decisions, "round": "Auction",
                                    "entity": loser, "player": pid, "seat": player_seats(rs).get(pid),
                                    "kind": "bid_lost", "constrained": [loser], "shortfall": int(price - best),
                                    "company": c.sym,
                                })
    except Exception as e:  # a game recorded before an engine rule change
        scan.error = repr(e)
    if emr_open is not None and scan.error is None:
        emr_open.pop("cash_before", None)
        emr_open["shortfall"] = None
        scan.events.append(emr_open)
    for e in scan.events:
        e["game_id"] = game_id
    for d in scan.decisions:
        d["game_id"] = game_id
    return scan


def shortfall_for(rs, entity: str, move: int, limit: int, offset: int = 0) -> Optional[int]:
    """The least extra cash (1..``limit``) for ``entity`` that makes ``move`` legal
    (bisection; None if even ``limit`` doesn't), at the state ``rs`` minus
    ``offset`` dollars of the entity's -- a probe that already holds ``offset``
    extra (``scan_game``'s) can be bisected without the original state."""

    def legal_with(x: int) -> bool:
        p = rs.pickle_clone()
        p.counterfactual_transfer_cash(entity, int(x) - int(offset))
        return int(move) in set(int(v) for v in p.factored_legal_indices())

    if not legal_with(limit):
        return None
    lo, hi = 0, int(limit)  # legal at hi, not at lo (0: the move was illegal)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if legal_with(mid):
            hi = mid
        else:
            lo = mid
    return hi


def unaffordable_events(scan: Scan, priors: list, p_wanted: float, probe_cash: int) -> list:
    """``unaffordable`` events from a scan's probes, given each probe's policy priors
    (over its probe legal moves): the probe policy's top move is newly legal with
    probability >= ``p_wanted``; its shortfall is bisected on the probe itself."""
    out = []
    for (i, entity, probe, legal, probe_legal, new, base), p in zip(scan.probes, priors):
        p = np.asarray(p, dtype=np.float64)
        if p.sum() <= 0:
            continue
        p = p / p.sum()
        top = int(np.argmax(p))
        move = int(probe_legal[top])
        new_set = set(new)
        if move not in new_set or p[top] < p_wanted:
            continue
        shortfall = shortfall_for(probe, entity, move, probe_cash, offset=probe_cash)
        decoded = probe.decode_index_to_map(move, None)
        out.append({
            **base, "kind": "unaffordable", "game_id": scan.game_id, "constrained": [entity],
            "wanted": move, "wanted_type": decoded.get("type"), "p_wanted": float(p[top]),
            "p_new": float(sum(p[k] for k, ix in enumerate(probe_legal) if ix in new_set)),
            "shortfall": shortfall,
        })
    return out


def link_events(decisions: list, events: list, window: int) -> list:
    """``(event, decision, helps)`` for every price decision within ``window``
    decisions before an event whose money went to or came from the event's
    constrained entities; ``helps`` = +1 if a higher price would have given them
    cash (they were paid), -1 if a lower one (they paid), 0 if the money moved
    between two of them (a corporation buying its president's private before an
    emergency both pay for: the pool is unchanged, only where the cash sat). A lost
    bid links to the loser's own bids on that company, where a higher bid helps (and,
    as cash, to what the loser paid elsewhere)."""
    from bisect import bisect_left

    links = []
    by_game: dict = {}
    for d in decisions:
        if d["kind"] == "price":
            by_game.setdefault(d["game_id"], []).append(d)
    for ds in by_game.values():
        ds.sort(key=lambda d: d["i"])
    starts = {g: [d["i"] for d in ds] for g, ds in by_game.items()}
    for e in events:
        constrained = set(e.get("constrained") or [])
        ds = by_game.get(e["game_id"], [])
        for d in ds[: bisect_left(starts.get(e["game_id"], []), e["i"])]:
            if e["decision_number"] - d["decision_number"] > window:
                continue
            if (
                e["kind"] == "bid_lost" and d["type"] == "bid" and d.get("payer") in constrained
                and d.get("company") == e.get("company")
            ):
                links.append((e, d, +1))  # its own bid on the lost company: higher would have won
                continue
            paid, paying = d.get("payee") in constrained, d.get("payer") in constrained
            if paid or paying:
                links.append((e, d, int(paid) - int(paying)))
    return links


def price_alternatives(price: int, lo: int, hi: int, deltas=DEFAULT_DELTAS, extremes: bool = True) -> list:
    """The played price first, then ``price + delta`` for each delta and (``extremes``)
    the range's ends, clipped to ``[lo, hi]`` and de-duplicated, in ascending order."""
    alts = {min(max(int(price) + int(d), int(lo)), int(hi)) for d in deltas}
    if extremes:
        alts |= {int(lo), int(hi)}
    alts.discard(int(price))
    return [int(price)] + sorted(alts)


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------


@dataclass
class _Rollout:
    """A policy-only continuation (``policy_selfplay._advance`` reads game /
    decisions / termination / price_row / uid)."""

    game: Any  # RustGameAdapter
    uid: str
    alt: int
    r: int
    rng: np.random.Generator
    decisions: int = 0
    termination: Optional[str] = None
    price_row: Optional[np.ndarray] = None
    values: dict = field(default_factory=dict)  # horizon -> deciding seat's value


def outcomes(game) -> dict:
    """Per seat: win share, net-worth share, placement points (1/0.5/0.2/0)."""
    from rl18xx.agent.alphazero.policy_gradient import game_facts, outcome, placement_points

    win, worth = outcome(game)
    net_worth, _, _ = game_facts(game)
    return {
        "win": [round(float(x), 4) for x in win],
        "worth": [round(float(x), 5) for x in worth],
        "place": [round(float(x), 3) for x in placement_points(net_worth)],
    }


def replay(actions: list, upto: int):
    """A fresh game with ``actions[:upto]`` applied (a RustGameAdapter)."""
    from rl18xx.agent.alphazero.start_positions import new_game

    g = new_game(NUM_PLAYERS)
    rs = g._game
    for action in actions[:upto]:
        rs.process_action(dict(action))
    return g


def apply_alternative(rs, job: dict, alt: dict) -> None:
    """Play the job's decision with an alternative (``{"index", "price"}``)."""
    rs.apply_action_index(int(alt["index"]), None if alt.get("price") is None else int(alt["price"]))


def job_alternatives(rs, job: dict, deltas=DEFAULT_DELTAS, extremes: bool = True) -> list:
    """The alternatives of a job's decision at ``rs`` (the state before it): for a
    price decision the played index at :func:`price_alternatives`; for a par /
    dividend the played index and the others of its kind (``price`` from the
    engine where fixed). The played one first."""
    d = job["decision"]
    if d["kind"] == "price":
        return [{"index": d["index"], "price": p, "delta": p - d["price"]}
                for p in price_alternatives(d["price"], d["lo"], d["hi"], deltas, extremes)]
    out = []
    for ix in [d["index"]] + list(d.get("alternatives", [])):
        pr = rs.price_range_for_index(int(ix))
        out.append({"index": int(ix), "price": None if pr is None else int(pr[0]), "delta": None})
    return out


def open_loop(base_rs, actions: list, job: dict, alt: dict) -> tuple:
    """Replay the game's own actions after the decision up to the job's event on the
    alternative's branch: ``(rs at the event or None, summary)``. Actions that are no
    longer legal are skipped (counted); the summary has the constrained entity's
    cash there, whether the event's entity is acting again, and (``unaffordable``)
    whether the wanted move is legal."""
    event = job.get("event")
    if not event:
        return None, {}
    rs = base_rs.pickle_clone()
    d = job["decision"]
    apply_alternative(rs, job, alt)
    skipped = 0
    for action in actions[d["i"] + 1 : event["i"]]:
        try:
            rs.process_action(dict(action))
        except Exception:
            skipped += 1
    if rs.finished:
        return None, {"skipped": skipped, "finished": True}
    entity = event["constrained"][0]
    summary = {
        "skipped": skipped,
        "same_entity": rs.current_entity_id == event["entity"],
        "cash": entity_cash(rs, entity),
        "cash_pool": sum(entity_cash(rs, e) for e in event["constrained"]),
    }
    if event["kind"] == "unaffordable" and event.get("wanted") is not None:
        summary["wanted_legal"] = int(event["wanted"]) in set(int(x) for x in rs.factored_legal_indices())
    return rs, summary


class BranchRunner:
    """Runs branching jobs in lockstep: every alternative of every job, ``rollouts``
    continuations each, all positions of a round in one request per network.
    ``evaluators``: ``{"policy": ..., "value": ...}`` (``lockstep.evaluator_for``
    backends); the value network is asked only at the horizons, the event
    (open loop) and the branch points."""

    def __init__(self, evaluators: dict, settings: dict):
        from rl18xx.agent.alphazero.lockstep import LockstepSearcher

        self.settings = settings
        self.searcher = LockstepSearcher(evaluators, dedupe_maps=bool(settings.get("dedupe_maps", True)))
        self.has_value = "value" in evaluators

    def _values(self, states: list) -> list:
        """The value network's canonical values at ``states`` (engine games, not over)."""
        from rl18xx.agent.alphazero.lockstep import PolicyRequest

        key = "value" if self.has_value else "policy"
        reqs = [PolicyRequest(rs, [int(x) for x in rs.factored_legal_indices()] or [0], key=key) for rs in states]
        reqs_live = [r for r in reqs if len(r.legal)]
        if reqs_live:
            self.searcher.round([], reqs_live)
        return [r.value for r in reqs]

    def run(self, jobs: list) -> list:
        from rl18xx.agent.alphazero import price_pmf
        from rl18xx.agent.alphazero.lockstep import PolicyRequest
        from rl18xx.agent.alphazero.policy_gradient import choose_price
        from rl18xx.agent.alphazero.policy_selfplay import _advance
        from rl18xx.rust_adapter import RustGameAdapter

        s = self.settings
        started = time.time()
        rollouts, results = [], []
        temperature = float(s.get("temperature", 0.5))
        horizons = tuple(int(h) for h in s.get("horizons", DEFAULT_HORIZONS))
        settings = {"max_decisions": int(s.get("max_decisions", 700)), "price_eps": 0.0}
        for j, job in enumerate(jobs):
            res = {"job": job["job_id"], "alts": [], "error": None}
            results.append(res)
            try:
                actions = job["actions"]
                d = job["decision"]
                base = replay(actions, d["i"])
                rs = base._game
                legal = [int(x) for x in rs.factored_legal_indices()]
                if int(d["index"]) not in legal:
                    raise RuntimeError("the decision's move is not legal on replay")
                seat = player_seats(rs).get(controlling_player(rs, rs.current_entity_id))
                res["seat"] = seat
                # The rollout policy's view of the decision itself.
                req = PolicyRequest(rs, legal, key="policy")
                self.searcher.round([], [req])
                pos = legal.index(int(d["index"]))
                res["prior_played"] = float(req.priors[pos])
                if d["kind"] == "price" and req.price_row is not None:
                    cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[d["slot"]], d["lo"], d["hi"])
                    res["cell_probs"] = [round(float(x), 6) for x in cells.cell_probs(req.price_row[d["slot"]])]
                alts = job_alternatives(rs, job, s.get("deltas", DEFAULT_DELTAS), s.get("extremes", True))
                # Open loop up to the event, and the value at each branch point and at the event.
                branch_states, event_states = [], []
                for k, alt in enumerate(alts):
                    b = rs.pickle_clone()
                    try:
                        apply_alternative(b, job, alt)
                    except Exception as e:
                        alt = {**alt, "error": repr(e)}
                        res["alts"].append(alt)
                        continue
                    if d["kind"] == "price":
                        alt["cell"] = int(price_pmf.price_cells(price_pmf.SLOT_TYPES[d["slot"]], d["lo"], d["hi"]).cell_of(alt["price"]))
                    else:
                        alt["prior"] = float(req.priors[legal.index(int(alt["index"]))])
                    ev_rs, ol = open_loop(rs, actions, job, alt)
                    alt["open_loop"] = ol
                    alt["rollouts"] = []
                    res["alts"].append(alt)
                    branch_states.append((alt, b))
                    if ev_rs is not None:
                        event_states.append((alt, ev_rs))
                    for r in range(int(s["rollouts"])):
                        rng = np.random.default_rng([int(job["seed"]), r])
                        ro = _Rollout(game=RustGameAdapter(b.pickle_clone()), uid=f"{job['job_id']}:{k}:{r}",
                                      alt=len(res["alts"]) - 1, r=r, rng=rng)
                        ro.job_index = j
                        rollouts.append(ro)
                live = [(alt, st) for alt, st in event_states if not st.finished]
                if live:
                    values = self._values([st for _, st in live])
                    for (alt, st), v in zip(live, values):
                        if v is not None:
                            alt["open_loop"]["value"] = round(value_for_seat(v, mover_seat(st), seat), 5)
            except Exception as e:
                res["error"] = repr(e)
                rollouts = [ro for ro in rollouts if getattr(ro, "job_index", -1) != j]

        # Closed loop: every rollout of every job in lockstep.
        active = list(rollouts)
        decisions = 0
        while active:
            policy_reqs, value_reqs, stepping = [], [], []
            for ro in active:
                legal = _advance(ro, settings, ro.rng)
                if legal is None:
                    continue
                rs = ro.game._game
                if ro.decisions in horizons and ro.decisions not in ro.values:
                    value_reqs.append((ro, PolicyRequest(rs, legal, key="value" if self.has_value else "policy")))
                req = PolicyRequest(rs, legal, key="policy")
                policy_reqs.append(req)
                stepping.append((ro, legal, req))
            active = [ro for ro, _, _ in stepping]
            if not stepping:
                break
            self.searcher.round([], policy_reqs + [r for _, r in value_reqs])
            for ro, req in value_reqs:
                seat = results[ro.job_index]["seat"]
                ro.values[ro.decisions] = round(value_for_seat(req.value, mover_seat(ro.game._game), seat), 5)
            for ro, legal, req in stepping:
                p = np.asarray(req.priors, dtype=np.float64)
                if temperature != 1.0:
                    p = np.power(np.clip(p, 1e-12, None), 1.0 / temperature)
                p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
                move = int(legal[int(ro.rng.choice(len(legal), p=p))])
                ro.price_row = req.price_row
                price, _, _ = choose_price(ro.game, move, req.price_row, ro.rng, 0.0)
                ro.game._game.apply_action_index(move, price)
                ro.decisions += 1
                decisions += 1
        for ro in rollouts:
            res = results[ro.job_index]
            if res["error"]:
                continue
            res["alts"][ro.alt]["rollouts"].append({
                "r": ro.r, **outcomes(ro.game), "values": {str(h): v for h, v in ro.values.items()},
                "decisions": ro.decisions, "termination": ro.termination,
            })
        for res in results:
            res["seconds"] = round(time.time() - started, 1)
        return results, decisions


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def alt_outcomes(alt: dict, seat: int, key: str = "win") -> np.ndarray:
    """The deciding seat's ``key`` outcome per rollout of an alternative, by rollout index."""
    rows = sorted(alt.get("rollouts", []), key=lambda r: r["r"])
    return np.array([float(r[key][seat]) for r in rows])


def paired_advantages(result: dict, key: str = "win") -> list:
    """Per alternative (the played one excluded): ``(alt, mean paired difference
    against the played price, its standard error, the per-rollout differences)``
    -- rollout r of both share its seed."""
    all_alts = result.get("alts", [])
    if not all_alts or not all_alts[0].get("rollouts"):
        return []
    alts = [all_alts[0]] + [a for a in all_alts[1:] if a.get("rollouts")]
    if len(alts) < 2:
        return []
    seat = result["seat"]
    played = alt_outcomes(alts[0], seat, key)
    out = []
    for alt in alts[1:]:
        y = alt_outcomes(alt, seat, key)
        n = min(len(y), len(played))
        if n < 2:
            continue
        diff = y[:n] - played[:n]
        out.append((alt, float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(n)), diff))
    return out


def horizon_advantages(result: dict, horizon: int) -> dict:
    """Per alternative index (the played one excluded): the mean paired difference of
    the value network's estimate at ``horizon`` decisions after the branch."""
    alts = [a for a in result.get("alts", []) if a.get("rollouts")]
    if len(alts) < 2:
        return {}
    key = str(horizon)

    def per_r(alt):
        return {r["r"]: r["values"][key] for r in alt["rollouts"] if key in r.get("values", {})}

    played = per_r(alts[0])
    out = {}
    for k, alt in enumerate(alts[1:], start=1):
        mine = per_r(alt)
        common = sorted(set(mine) & set(played))
        if common:
            out[k] = float(np.mean([mine[r] - played[r] for r in common]))
    return out


def cell_advantages(result: dict, key: str = "win") -> dict:
    """Price cell -> (advantage over the played price, standard error, prices tested)
    for a price job: the mean of the paired differences of the alternatives in the
    cell (the played price contributes 0 to its own cell)."""
    out: dict = {}
    alts = result.get("alts", [])
    if not alts or not alts[0].get("rollouts") or "cell" not in alts[0]:
        return out
    by_cell: dict = {}
    by_cell.setdefault(alts[0]["cell"], []).append((0.0, 0.0, alts[0]["price"]))
    for alt, adv, se, _ in paired_advantages(result, key):
        by_cell.setdefault(alt["cell"], []).append((adv, se, alt["price"]))
    for cell, rows in by_cell.items():
        advs = np.array([r[0] for r in rows])
        ses = np.array([r[1] for r in rows])
        adv = float(advs.mean())
        se = float(np.sqrt((ses**2).sum()) / len(rows))
        out[int(cell)] = (adv, se, [int(r[2]) for r in rows])
    return out


def price_cell_target_from_advantages(probs, cell_adv: dict, beta: float, z: float = 0.0) -> np.ndarray:
    """An advantage-weighted target over the price head's cells (AWR / MPO style):
    target(c) proportional to probs(c) * exp(A(c) / beta), where A(c) is the cell's
    advantage over the played price if measured (and, with ``z`` > 0, only where
    |A| >= z * SE), else 0. The played price's cell counts as measured with A = 0."""
    p = np.asarray(probs, dtype=np.float64)
    a = np.zeros_like(p)
    for cell, (adv, se, _) in cell_adv.items():
        if z > 0 and abs(adv) < z * se:
            continue
        a[int(cell)] = adv
    live = p > 0
    logits = np.where(live, np.log(np.clip(p, 1e-12, None)) + a / beta, -np.inf)
    logits -= logits[live].max()
    t = np.where(live, np.exp(logits), 0.0)
    return t / t.sum()


def expected_under(probs, cell_q: dict) -> Optional[float]:
    """E over ``probs`` restricted (renormalized) to the cells with a measurement
    ``cell_q`` (cell -> value): how a price head would do on the tested prices."""
    p = np.asarray(probs, dtype=np.float64)
    cells = [c for c in cell_q if p[int(c)] > 0]
    if not cells:
        return None
    w = np.array([p[int(c)] for c in cells])
    return float((w * np.array([cell_q[c] for c in cells])).sum() / w.sum())


def split_half_gain(result: dict, key: str = "win") -> Optional[tuple]:
    """Pick the best alternative on the even rollouts (its advantage > 0, else keep the
    played price), score that choice on the odd ones: ``(chosen alt's delta or 0,
    its odd-half advantage)`` -- an unbiased estimate of what switching to the
    nearby price the first half prefers gains."""
    best, best_adv = None, 0.0
    for alt, _, _, diff in paired_advantages(result, key):
        even = diff[0::2]
        if len(even) and even.mean() > best_adv:
            best, best_adv = (alt, diff), float(even.mean())
    if best is None:
        return (0, 0.0)
    alt, diff = best
    odd = diff[1::2]
    if not len(odd):
        return None
    return (alt.get("delta") if alt.get("delta") is not None else alt["price"], float(odd.mean()))


def to_json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(type(o))

    return json.dumps(obj, default=default)
