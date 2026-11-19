"""M1 of docs/robustness_test_plan.md: the regression set -- positions from Caleb's games against the bots on the
local site, just before each bot mistake, scored for any checkpoint (docs/robustness_measurement.md).

Each position (``data/regression_set.json``) names a game (``data/caleb_games/<id>.json.gz``: the site game as the
advisor's importer played it in the Rust engine, with each engine action's site action id), how many of its engine
actions to replay (``action_index``), the mover, the move the bot played, the bad moves and the preferred ones (as
engine action dicts), and how a model passes:

- ``avoid_bad``: the search must not choose a bad move (a buy into a company about to be looted and dumped, a
  sell-out handing over a treasury, a doomed buy, a buy into a suitcase, a raise in a bid war);
- ``choose_preferred``: the search must choose a preferred move (contest Caleb's CA / MH in the auction).

Weaknesses (docs/bot_weaknesses_2026_11.md): ``auction`` (1), ``loot_dump`` (2), ``sellout`` (3), ``doomed`` (4),
``suitcase`` (5).

    # score a legacy system (policy + value, as composite_model.load_policy_value / eval_head_to_head take it)
    uv run python scripts/robustness/regression_set.py score \
        --model <policy.pth>+<critic.pth>[#score|#blend=0.5] [--search @400/0.05:mean,dedupe,par=8,ptemp=0.5] \
        --out logs/robustness/m1_<name>.jsonl
    uv run python scripts/robustness/regression_set.py score --model generic:<ckpt.pth> --no-search  # policy+value
    # a generic learner + critic searched on the critic's placement-points head (eval_head_to_head's syntax)
    uv run python scripts/robustness/regression_set.py score \
        --model generic:<run>/learner/<u>.pth+generic:<run>/critic/<u>.pth#score --out logs/robustness/m1_<name>.jsonl
    uv run python scripts/robustness/regression_set.py summary logs/robustness/m1_*.jsonl
    uv run python scripts/robustness/regression_set.py list
    # rebuild the fixture (data/regression_set.json) from the curated list below
    uv run python scripts/robustness/regression_set.py build
    # (re)convert site exports (18xx.games-style, e.g. from the local site's database) into data/caleb_games
    uv run python scripts/robustness/regression_set.py convert <export.json>...

A move matches a fixture move when applying it gives the same position (the canonical encoding): share
certificate ids and the like don't matter; a move with an open price (a bid) is applied at the fixture move's price,
so "bid on CA" matches whatever the price.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DATA = HERE / "data"
GAMES_DIR = DATA / "caleb_games"
FIXTURE = DATA / "regression_set.json"
WEAKNESSES = ("auction", "loot_dump", "sellout", "doomed", "suitcase")
# The live bots' search (rl18xx/agent/advisor/advisor.py SEARCH_OVERRIDES, priors at gen6's learner temperature 0.5),
# at 400 readouts, in eval_head_to_head's spec syntax. Fresh tree per position (the bots keep trees between moves).
DEFAULT_SEARCH = "@400/0.05:mean,dedupe,par=8,ptemp=0.5"
# Which networks the bots played with (UTC, the site's timestamps): docs/bot_weaknesses_2026_11.md.
ERAS = [
    ("2026-11-12T02:54:26", "gen4", "gen4 rab_place/1800 + lr1 v4 (play), gen4 critic 1800 (auction)"),
    ("2026-11-13T09:03:18", "gen5", "gen5 u2800 + lr1 v4 (play), gen5 critic (auction)"),
    ("2026-11-14T09:36:15", "gen6+lr1v4", "gen6 u2400 + lr1 v4 (play), gen6 critic 2400 (auction)"),
    ("9999", "gen6", "gen6 u2400 + gen6 critic 2400"),
]

# ------------------------------------------------------------------------------------- curated positions
# (game, site action id, weakness, kind, note). Kinds: "contest" (bad = the played move, preferred = the mover's bids
# on CA / MH: Caleb's uncontested openers), "war" (bad = the raise, preferred = pass), "buy" (bad = buying any share
# of the played company, preferred = pass), "sellout" (bad = any sale of the played company that hands its
# presidency over, preferred = pass). Picked from the analysis' dossiers (dumps_user.txt, float_sell_user.txt,
# doomed.py, the auction logs); the site ids are those in docs/bot_weaknesses_2026_11.md.
CURATED = [
    # 1. auction: Caleb's CA / MH openers never contested ...
    (21, 2, "auction", "contest", "after Caleb's MH $115 (CA still free): bids CS $45; probe: policy 0.55 on CA $165"),
    (21, 3, "auction", "contest", "after Caleb's MH $115 and a CS bid: bids DH $75 (starts a DH war to $215)"),
    (21, 6, "auction", "contest", "after Caleb's MH $115 + CA $165: buys SV"),
    (20, 4, "auction", "contest", "after Caleb's MH $115: bids CS $50 (Caleb later takes CA at $160 unopposed)"),
    (19, 3, "auction", "contest", "after Caleb's MH $115: passes"),
    (19, 4, "auction", "contest", "after Caleb's MH $115: bids CS $50 (a CS war to $150 follows)"),
    (19, 7, "auction", "contest", "after Caleb's MH $115 + CA $165: buys SV"),
    (18, 3, "auction", "contest", "after Caleb's CA $165: bids DH $75 (starts a DH war to $195)"),
    (18, 4, "auction", "contest", "after Caleb's CA $165 and a DH bid: bids DH $80"),
    (13, 6, "auction", "contest", "after Caleb's DH $75 / CA $165: passes"),
    # one position opened games 5, 10, 11, 14 and 15 (Caleb first to act, CA $165): kept once (build rejects repeats)
    (10, 2, "auction", "contest", "Caleb's CA $165 opening (games 5, 10, 11, 14, 15): passes"),
    (10, 3, "auction", "contest", "Caleb's CA $165, a pass: bids DH $75 (games 5, 10, 11; DH wars to $185)"),
    (14, 3, "auction", "contest", "Caleb's CA $165, a CS bid: bids CS $50 (games 14, 15; CS wars to $110)"),
    # ... while the bots bid each other's DH / CS far over face
    (21, 20, "auction", "war", "DH war: raises to $150 (face $70); probe: critic prefers pass"),
    (21, 33, "auction", "war", "DH war: raises to $215; probe: prior 0.996 on the raise beat the critic's pass"),
    (20, 29, "auction", "war", "CS war: raises to $175 (face $40)"),
    (19, 26, "auction", "war", "CS war: raises to $150"),
    (18, 27, "auction", "war", "DH war: raises to $190"),
    (15, 14, "auction", "war", "CS war: raises to $100"),
    (5, 30, "auction", "war", "DH war: raises to $185"),
    (11, 40, "auction", "war", "DH war: raises to $170"),
    # 2. buy-ins before a loot-and-dump (the company is emptied, then dumped trainless on the buyer)
    (21, 132, "loot_dump", "buy", "buys C&O (Caleb's; $510, 2+2); looted OR5.1, dumped [343], $630 contributed"),
    (21, 235, "loot_dump", "buy", "buys C&O (Caleb's; $680, a 2); probe: prior 0.998; dumped [343]"),
    (19, 297, "loot_dump", "buy", "buys NYC (Caleb's; $700, a 4); looted OR6, dumped [436], buyer bankrupt"),
    (19, 319, "loot_dump", "buy", "buys NYC again (3rd share); dumped [436], buyer bankrupt"),
    (19, 334, "loot_dump", "buy", "buys NYC again (5th share); dumped [436], buyer bankrupt"),
    (15, 402, "loot_dump", "buy", "buys NYC (Caleb's; $640, a 3); looted OR7.2, dumped [507], buyer bankrupt"),
    (15, 420, "loot_dump", "buy", "buys NYC again; dumped [507], buyer bankrupt"),
    (15, 425, "loot_dump", "buy", "buys NYC again (3rd share); dumped [507], buyer bankrupt"),
    (10, 377, "loot_dump", "buy", "buys NYC (Caleb's; $1000); looted OR8.1 ($1000 2-train), dumped [576], bankrupt"),
    (10, 412, "loot_dump", "buy", "buys NYC again; dumped [576], buyer bankrupt"),
    (5, 633, "loot_dump", "buy", "buys C&O (Caleb's; $186, a 4 that rusts on the D) -- dumped two actions later"),
    # 3. sell-outs handing a treasury to Caleb
    (21, 398, "sellout", "sellout", "sells NYC ($900, parred this SR): presidency to Bot 2, who passes it to Caleb"),
    (21, 408, "sellout", "sellout", "sells NYC ($900): Caleb becomes president; probe: prior 0.000, search sold"),
    (20, 198, "sellout", "sellout", "sells PRR 40% ($670, no trains): Caleb becomes president"),
    (19, 117, "sellout", "sellout", "sells NYNH ($390): Caleb becomes president"),
    (18, 101, "sellout", "sellout", "sells C&O ($590): Caleb becomes president"),
    (15, 155, "sellout", "sellout", "sells C&O ($590): Caleb becomes president"),
    (15, 345, "sellout", "sellout", "sells B&M ($340): Caleb becomes president"),
    (11, 175, "sellout", "sellout", "sells C&O ($590): Caleb becomes president"),
    (10, 318, "sellout", "sellout", "sells C&O ($590): Caleb becomes president"),
    # 4. doomed buys: shares (or the presidency) of a floated, trainless company that can't afford the next train
    (21, 360, "doomed", "buy", "buys C&O (Bot 3's; $1, trainless, phase 4)"),
    (21, 399, "doomed", "buy", "buys C&O (Bot 3's; $1, trainless, phase 4)"),
    (21, 416, "doomed", "buy", "buys C&O again (Bot 3's; $1, trainless)"),
    (19, 432, "doomed", "buy", "buys B&O ($26, trainless, phase 6, next a D) and becomes its president"),
    (19, 441, "doomed", "buy", "buys NYC (Bot 1's; $1, trainless, phase 6)"),
    (19, 443, "doomed", "buy", "buys NYC (Bot 1's; $1, trainless, phase 6)"),
    (15, 522, "doomed", "buy", "buys NYNH (Bot 1's; $100, trainless, phase 6, next a D)"),
    (14, 720, "doomed", "buy", "buys B&O ($1, trainless, phase D) and becomes its president"),
    (10, 599, "doomed", "buy", "buys NYC (Bot 3's; $0, trainless, phase 5)"),
    (12, 1020, "doomed", "buy", "6 players: buys B&O (Bot 2's; $1, trainless, phase D)"),
    # 5. suitcases: Caleb's CPR, trainless with no route (no train obligation), bought into and taken over
    (18, 512, "suitcase", "buy", "buys CPR (Caleb's suitcase, trainless 9 ORs; $102, no route)"),
    (18, 662, "suitcase", "buy", "buys CPR (Caleb's suitcase; $102, no route)"),
    (14, 615, "suitcase", "buy", "buys CPR (Caleb's; $0, no route); dumped on it [624], then Caleb lays its route"),
    (14, 620, "suitcase", "buy", "buys CPR again; dumped [624], forced to buy a D after Caleb's route [697]"),
    (12, 908, "suitcase", "buy", "6 players: buys CPR (Caleb's; $1, no route)"),
    (12, 919, "suitcase", "buy", "6 players: buys CPR and takes it over; Caleb lays its route [996]"),
    (8, 528, "suitcase", "buy", "buys CPR (Caleb's; $1, no route)"),
    (8, 560, "suitcase", "buy", "buys CPR and takes it over (Caleb's suitcase)"),
]


def era_of(timestamp: Optional[str]) -> tuple:
    for end, short, desc in ERAS:
        if timestamp and timestamp < end:
            return short, desc
    return ERAS[-1][1], ERAS[-1][2]


# ------------------------------------------------------------------------------------------- engine helpers
def load_game_record(game_id) -> dict:
    from robust_games import read_json

    return read_json(GAMES_DIR / f"{game_id}.json.gz")


def clean_action(action: dict) -> dict:
    return {k: v for k, v in action.items() if k not in ("created_at", "user")}


def rust_game_at(record: dict, k: int, no_buyback_rule: bool = True):
    """The game of ``record`` (an engine-format game) after its first ``k`` actions, as the bots searched it: the Rust
    engine, the real auction rules, the rule against train shuffling on (``autoplay.bot_position``)."""
    from engine_rs import BaseGame as RustGame

    from rl18xx.rust_adapter import RustGameAdapter

    n = len(record["players"])
    rules = [r for r in (record.get("settings") or {}).get("optional_rules") or []
             if r in ("multiple_brown_from_ipo", "optional_6_train")]
    rust = RustGame({i: f"Player {i}" for i in range(1, n + 1)}, optional_rules=rules)
    rust.set_auction_unlock(False)
    rust.set_no_buyback_rule(bool(no_buyback_rule))
    game = RustGameAdapter(rust)
    for a in record["actions"][:k]:
        game.process_action(clean_action(a))
    return game


def acting_player(game) -> Optional[int]:
    players = game.active_players()
    return int(players[0].id) if players else None


def signature(game) -> str:
    """A digest of a position (its canonical encoding): equal for moves that lead to the same position."""
    gs, nf, round_type, rotation, num_players = game._game.encode_canonical_py()
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(gs).tobytes())
    h.update(np.ascontiguousarray(nf).tobytes())
    h.update(f"{round_type}:{rotation}:{num_players}".encode())
    return h.hexdigest()


def legal_moves(game) -> list:
    """``[(index, (lo, hi) or None)]`` of the position's legal moves."""
    out = []
    for index in game._game.factored_legal_indices():
        pr = game._game.price_range_for_index(int(index))
        out.append((int(index), (int(pr[0]), int(pr[1])) if pr is not None else None))
    return out


def apply_index(game, index: int, price: Optional[int] = None):
    """``(position after, the action it logged)`` of playing ``index`` (at ``price`` if it has a price range)."""
    probe = game.pickle_clone()
    n0 = len(probe.raw_actions)
    probe._game.apply_action_index(int(index), None if price is None else int(price))
    logged = list(probe.raw_actions)
    return probe, (dict(logged[n0]) if len(logged) > n0 else None)


def apply_action(game, action: dict):
    probe = game.pickle_clone()
    probe.process_action(clean_action(action))
    return probe


def match_indices(game, action: dict, moves: Optional[list] = None) -> list:
    """The legal indices that play ``action`` (an engine action dict): same resulting position, an open price played
    at the action's price. Usually one."""
    try:
        target = signature(apply_action(game, action))
    except Exception:
        return []
    price = action.get("price")
    out = []
    for index, pr in moves if moves is not None else legal_moves(game):
        if pr is not None and pr[0] != pr[1]:
            if price is None or not pr[0] <= int(price) <= pr[1]:
                continue
            p = int(price)
        else:
            p = pr[0] if pr is not None else None
        try:
            after, _ = apply_index(game, index, p)
        except Exception:
            continue
        if signature(after) == target:
            out.append(index)
    return out


def describe(action: Optional[dict]) -> str:
    """A short description of an engine action dict."""
    if not action:
        return "-"
    t = action.get("type")
    if t == "bid":
        return f"bid {action.get('company')} ${action.get('price')}"
    if t in ("buy_shares", "sell_shares"):
        shares = action.get("shares") or []
        corp = shares[0].split("_")[0] if shares else action.get("corporation")
        verb = "buy" if t == "buy_shares" else "sell"
        return f"{verb} {action.get('percent', 10 * len(shares))}% {corp}"
    if t == "par":
        return f"par {action.get('corporation')} {str(action.get('share_price', '')).split(',')[0]}"
    if t == "buy_train":
        return f"{action.get('entity')} buys {action.get('variant') or action.get('train')} ${action.get('price')}"
    if t == "lay_tile":
        return f"{action.get('entity')} lays {action.get('tile')} at {action.get('hex')}"
    if t == "buy_company":
        return f"{action.get('entity')} buys {action.get('company')} ${action.get('price')}"
    return f"{t}" + (f" ({action.get('entity')})" if action.get("entity_type") != "player" else "")


def move_action(game, index: int, moves: Optional[list] = None) -> Optional[dict]:
    """The action a legal index plays (an open price at the bottom of its range)."""
    pr = dict(moves or legal_moves(game)).get(index)
    try:
        _, action = apply_index(game, index, pr[0] if pr is not None else None)
    except Exception:
        return None
    return action


# ------------------------------------------------------------------------------------------------- building
def convert(paths, out_dir=GAMES_DIR) -> None:
    """Site exports -> engine-format games (``robust_games.convert_site_export``), with each player's name."""
    from robust_games import convert_site_export, read_json, write_json

    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json"), key=lambda f: (len(f.name), f.name)) if p.is_dir() else [p]
    for path in files:
        export = read_json(path)
        record = convert_site_export(export)
        record["id"] = f"caleb-{export.get('id')}"
        record["rl18xx"]["source"] = "local 18xx.games site (Caleb vs the bots), converted with the advisor's importer"
        out = Path(out_dir) / f"{export.get('id')}.json.gz"
        write_json(out, record, compress=True)
        print(f"{path} -> {out}: {len(record['actions'])} engine actions, finished {record['rl18xx']['finished']}, "
              f"error {record['rl18xx']['conversion_error']}")


def _corp_of(action: dict) -> Optional[str]:
    shares = action.get("shares") or []
    return shares[0].split("_")[0] if shares else action.get("corporation")


def _context(game, corp: Optional[str], mover: int) -> dict:
    phase = getattr(game, "phase", None)
    rnd = game._game.round
    desc = {"Auction": "Auction", "Stock": f"Stock Round {game._game.turn}"}.get(
        rnd.round_type, f"Operating Round {game._game.turn}.{rnd.round_num}")
    ctx = {"round": desc, "phase": str(getattr(phase, "name", phase))}
    p = next(p for p in game.players if int(p.id) == int(mover))
    ctx["mover_cash"] = int(p.cash)
    if corp:
        c = game.corporation_by_id(corp)
        owner = c.owner
        ctx["corp"] = {"name": corp, "cash": int(c.cash), "trains": [t.name for t in c.trains],
                       "president": int(owner.id) if owner is not None and owner.is_player() else None,
                       "share_price": int(c.share_price.price) if c.share_price else None}
    return ctx


def build_position(record: dict, gid: int, site_id: int, weakness: str, kind: str, note: str) -> dict:
    site_ids = record["rl18xx"]["site_action_ids"]
    if site_id not in site_ids:
        raise ValueError(f"game {gid}: no engine action for site action {site_id}")
    k = site_ids.index(site_id)
    played = clean_action(record["actions"][k])
    game = rust_game_at(record, k)
    mover = acting_player(game)
    if played.get("entity_type") != "player" or int(played["entity"]) != mover:
        raise ValueError(f"game {gid} [{site_id}]: the played action {played} is not the acting player {mover}'s")
    moves = legal_moves(game)
    played_idx = match_indices(game, played, moves)
    if len(played_idx) != 1:
        raise ValueError(f"game {gid} [{site_id}]: the played move matches indices {played_idx}")
    actions = {index: move_action(game, index, moves) for index, _ in moves}
    pass_moves = [a for a in actions.values() if a and a.get("type") == "pass"]
    corp = _corp_of(played) if played.get("type") in ("buy_shares", "sell_shares") else None
    if kind == "contest":
        bad = [played]
        preferred = [a for a in actions.values() if a and a.get("type") == "bid" and a.get("company") in ("CA", "MH")]
        criterion = "choose_preferred"
    elif kind == "war":
        bad = [played]
        preferred = pass_moves
        criterion = "avoid_bad"
    elif kind == "buy":
        bad = [a for a in actions.values() if a and a.get("type") == "buy_shares" and _corp_of(a) == corp]
        preferred = pass_moves
        criterion = "avoid_bad"
    elif kind == "sellout":
        bad = []
        for index, pr in moves:
            a = actions[index]
            if not a or a.get("type") != "sell_shares" or _corp_of(a) != corp:
                continue
            after, _ = apply_index(game, index, None)
            owner = after.corporation_by_id(corp).owner
            if owner is None or not owner.is_player() or int(owner.id) != mover:
                bad.append(a)
        preferred = pass_moves
        criterion = "avoid_bad"
    else:
        raise ValueError(f"unknown kind {kind!r}")
    if not any(match_indices(game, b, moves) == played_idx for b in bad) and kind != "contest":
        raise ValueError(f"game {gid} [{site_id}]: the played move is not among the bad moves")
    if not preferred:
        raise ValueError(f"game {gid} [{site_id}]: no preferred move")
    names = record["rl18xx"].get("names") or {}
    era, era_desc = era_of(record["actions"][k].get("created_at"))
    return {
        "id": f"g{gid}-a{site_id}",
        "weakness": weakness,
        "kind": kind,
        "criterion": criterion,
        "game": gid,
        "file": f"caleb_games/{gid}.json.gz",
        "site_action_id": site_id,
        "action_index": k,
        "num_players": len(record["players"]),
        "mover": {"player_id": mover, "name": names.get(str(mover))},
        "era": era,
        "bots_played_with": era_desc,
        "played": played,
        "played_desc": describe(played),
        "bad_moves": bad,
        "preferred_moves": preferred,
        "context": _context(game, corp, mover),
        "note": note,
    }


def build(out=FIXTURE) -> list:
    positions = []
    records = {}
    seen = {}
    for gid, site_id, weakness, kind, note in CURATED:
        if gid not in records:
            records[gid] = load_game_record(gid)
        pos = build_position(records[gid], gid, site_id, weakness, kind, note)
        key = signature(rust_game_at(records[gid], pos["action_index"]))  # one entry per position
        if key in seen:
            raise ValueError(f"{pos['id']} is the same position as {seen[key]}")
        seen[key] = pos["id"]
        positions.append(pos)
        print(f"{pos['id']:12s} {weakness:9s} {kind:8s} {pos['era']:10s} {pos['mover']['name']:6s} "
              f"played {pos['played_desc']:16s} bad {len(pos['bad_moves'])} preferred {len(pos['preferred_moves'])} "
              f"| {pos['context']['round']}", flush=True)
    data = {
        "version": 1,
        "description": "M1 regression set (docs/robustness_test_plan.md, docs/robustness_measurement.md): "
                       "positions from Caleb's games against the bots just before each bot mistake.",
        "default_search": DEFAULT_SEARCH,
        "positions": positions,
    }
    Path(out).write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(positions)} positions -> {out}")
    return positions


def load_fixture(path=FIXTURE) -> list:
    return json.loads(Path(path).read_text())["positions"]


# ------------------------------------------------------------------------------------------------- models
def _abs_seat(game, player_id: int) -> int:
    return sorted(int(p.id) for p in game.players).index(int(player_id))


class LegacySystem:
    """A legacy (``model_transformer``) policy + value: ``<policy.pth>[+<value.pth>][#score|#blend=<w>]``, searched
    with ``RustMCTSPlayer`` under the search settings of ``eval_head_to_head``'s spec syntax."""

    kind = "legacy"
    can_search = True

    def __init__(self, spec: str, search: str = DEFAULT_SEARCH, device: str = "cpu"):
        from eval_head_to_head import parse_player

        from rl18xx.agent.alphazero.composite_model import load_policy_value

        self.spec = spec
        self.search_spec = search
        self.full_spec = spec if "@" in spec else spec + search
        self.paths, self.overrides = parse_player(self.full_spec, default_readouts=400, num_players=4)
        self.overrides_by_count = {}
        self.net = load_policy_value(self.paths)
        self.net.eval()
        self.value_model = getattr(self.net, "value_model", None) or getattr(self.net, "model", None) or self.net
        self.prior_temperature = float(self.overrides.get("prior_temperature", 1.0))
        self.readouts = int(self.overrides.get("num_readouts", 400))

    @classmethod
    def from_network(cls, net, overrides: dict, label: str = "network") -> "LegacySystem":
        """A system around an already built network (anything with ``run_many_encoded`` / ``run_encoded``, as the
        search uses one) and ``SelfPlayConfig`` search overrides, for every player count (tests, notebooks)."""
        self = cls.__new__(cls)
        self.spec = self.full_spec = label
        self.search_spec = ""
        self.paths = None
        self.overrides = dict(overrides)
        self.overrides_by_count = {n: dict(overrides) for n in range(2, 7)}
        self.net = net
        self.value_model = getattr(net, "value_model", None) or getattr(net, "model", None) or net
        self.prior_temperature = float(overrides.get("prior_temperature", 1.0))
        self.readouts = int(overrides.get("num_readouts", 400))
        return self

    def _overrides(self, num_players: int) -> dict:
        """Search overrides for a game of ``num_players`` (the score head's terminal utility depends on the count)."""
        if num_players not in self.overrides_by_count:
            from eval_head_to_head import parse_player

            self.overrides_by_count[num_players] = parse_player(self.full_spec, 400, num_players)[1]
        return self.overrides_by_count[num_players]

    def policy(self, game, legal: list) -> np.ndarray:
        """Probabilities (T = 1) over ``legal`` (indices)."""
        import torch

        from rl18xx.agent.alphazero.mcts import _rust_encode

        with torch.no_grad():
            probs, _, _ = self.net.run_many_encoded([_rust_encode(game)])
        p = np.asarray(probs[0].float().cpu().numpy() if hasattr(probs[0], "cpu") else probs[0], dtype=np.float64)
        p = np.clip(p[np.asarray(legal, dtype=np.int64)], 0.0, None)
        return p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))

    def values(self, game, player_id: int) -> dict:
        """The value network's estimates for ``player_id``: win share, score head, and the search's value."""
        import torch

        from rl18xx.agent.alphazero.mcts import _rust_encode, unrotate_value

        e = _rust_encode(game)
        seat = _abs_seat(game, player_id)
        n = int(e[7])
        out = {}
        with torch.no_grad():
            vm = self.value_model
            if hasattr(vm, "run_values_and_scores_encoded"):
                win, score = vm.run_values_and_scores_encoded([e])
                win = unrotate_value(win[0].float().cpu().numpy(), e[6], n)[:n]
                score = unrotate_value(score[0].float().cpu().numpy(), e[6], n)[:n]
                out["win"] = float(win[seat] / win.sum()) if win.sum() > 0 else 0.0
                out["score"] = float(score[seat])
            _, _, v = self.net.run_many_encoded([e])
        v = unrotate_value(np.asarray(v[0].float().cpu().numpy() if hasattr(v[0], "cpu") else v[0]), e[6], n)[:n]
        out["search"] = float(v[seat])
        return out

    def search(self, game, readouts: Optional[int] = None) -> dict:
        """A fresh search at ``game`` as the bots ran theirs: root evaluated, then batches until the root has
        ``readouts`` visits. Returns the root statistics (legal, visits, Q, priors, chosen index)."""
        import torch

        from rl18xx.agent.alphazero.config import SelfPlayConfig
        from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
        from rl18xx.agent.alphazero.self_play import _slice_price_components

        n = len(game.players)
        overrides = dict(self._overrides(n))
        readouts = int(readouts or overrides.get("num_readouts", self.readouts))
        overrides.update(num_readouts=readouts, min_readouts=readouts)
        config = SelfPlayConfig(
            network=self.net, dirichlet_noise_weight=0.0, enable_resign=False, noresign_holdout_rate=0.0,
            player_count_distribution={n: 1.0}, auction_unlock=False, softpick_move_cutoff=0, **overrides,
        )
        player = RustMCTSPlayer(config)
        player.initialize_game(game.pickle_clone())
        tree = player._rust_player
        with torch.no_grad():
            leaf = player.root.select_leaf()
            leaf.ensure_encoded()
            probs, _, value = self.net.run_encoded(leaf.encoded_game_state)
            leaf.incorporate_results(probs, value, leaf,
                                     price_components=_slice_price_components(self.net.last_price_components, 0))
            while tree.n_at_root() < readouts:
                before = tree.n_at_root()
                player.tree_search()
                if tree.n_at_root() <= before:
                    break
        st = tree.root_target_stats()
        return {
            "legal": [int(x) for x in st["legal"]],
            "child_n": [float(x) for x in st["child_n"]],
            "child_q": [float(x) for x in st["child_q"]],
            "child_prior": [float(x) for x in st["child_prior"]],
            "root_value": float(st["root_value"]),
            "raw_value": float(st["raw_value"]),
            "root_n": float(st["root_n"]),
            "chosen": int(tree.pick_best_action()),
        }


def _prior_temperature_suffix(spec: str):
    """``(spec without it, T)`` when ``spec`` ends in ``@<T>`` with a decimal point (the M1 scorer's older generic
    syntax ``generic:<ckpt>@0.5``: the prior temperature, the search settings from ``--search``), else
    ``(spec, None)``; ``@<readouts>...`` is eval_head_to_head's search spec."""
    head, at, tail = spec.rpartition("@")
    if at and "." in tail:
        try:
            return head, float(tail)
        except ValueError:
            pass
    return spec, None


def _with_prior_temperature(search: str, temperature: float) -> str:
    """``search`` (eval_head_to_head's ``@<n>[/<c>][:options]``) with its ``ptemp`` set to ``temperature``."""
    budget, colon, options = search.partition(":")
    kept = [o for o in options.split(",") if o and not o.startswith("ptemp=")]
    return budget + ":" + ",".join(kept + [f"ptemp={temperature}"])


class GenericSystem:
    """Title-generic checkpoints (``model_generic``): ``generic:<policy.pth>[+generic:<value.pth>][#win|#score|
    #blend=<w>]``, the syntax of ``eval_head_to_head``'s generic players, searched in-process (``game_network``) with
    the legacy Rust tree under the same search settings as :class:`LegacySystem` (``--search``, default the live bots'
    ``@400/0.05:mean,dedupe,par=8,ptemp=0.5``; a spec may carry its own ``@<n>...``). No ``+value``: the policy
    network's own value heads. The older ``generic:<ckpt>@<T>`` (a decimal T) keeps meaning the prior temperature,
    with the rest of the search from ``--search``.

    Policy: probabilities at temperature 1 over the legal moves (1830's legal indices are the factored layout's, the
    legacy model's). Values: the value network's win and score heads for the mover, and the search's head
    (``search``). Search: :class:`game_network.GameNetwork` (priors at temperature 1, the tree tempering them with
    ``ptemp``, as eval_head_to_head's generic players; the price head's cells as the legacy price rows; values in
    absolute seat order) through ``RustMCTSPlayer.tree_search``'s ``evaluates_games`` hook, the root evaluated by a
    one-leaf batch as the advisor's generic searches; ``#score`` backs up placement points at finished games when the
    value's run recorded a placement reward (``parse_player``). ~12-17 ms of CPU per leaf for default-size networks
    (docs/rl_model.md "Head-to-head evals of generic searches"): ~5 s a position at 400 readouts."""

    kind = "generic"

    def __init__(self, spec: str, search: str = DEFAULT_SEARCH, device: str = "cpu"):
        from eval_head_to_head import generic_parts, parse_player

        from rl18xx.agent.alphazero.game_network import GameNetwork, GenericNetwork

        self.spec = spec
        body, temperature = _prior_temperature_suffix(spec)
        if temperature is not None:
            search = _with_prior_temperature(search, temperature)
        self.search_spec = search
        own = body.replace("+generic:", "+").partition("generic:")[2]  # the spec past its parts' prefixes
        self.full_spec = body if ("@" in own or ":" in own) else body + search
        key, self.overrides = parse_player(self.full_spec, default_readouts=400, num_players=4)
        self.overrides_by_count = {4: dict(self.overrides)}
        parts = generic_parts(key)
        self.paths = parts
        # "sample=<t>" (the policy alone in eval_head_to_head): no search, priors at t
        self.can_search = "policy_temperature" not in self.overrides
        self.prior_temperature = float(
            self.overrides.get("prior_temperature", self.overrides.get("policy_temperature", 1.0)))
        self.readouts = int(self.overrides.get("num_readouts", 0)) if self.can_search else 0
        self.policy_net = GenericNetwork(parts["policy"], device)
        self.value_net = GenericNetwork(parts["value"], device) if parts["value"] else self.policy_net
        self.value_head = parts["value_head"]
        # priors at temperature 1: the tree tempers them with the spec's ptemp (SelfPlayConfig.prior_temperature)
        self.net = GameNetwork(self.policy_net, self.value_net, policy_temperature=1.0, value_head=self.value_head)

    def _overrides(self, num_players: int) -> dict:
        if num_players not in self.overrides_by_count:
            from eval_head_to_head import parse_player

            self.overrides_by_count[num_players] = parse_player(self.full_spec, 400, num_players)[1]
        return self.overrides_by_count[num_players]

    def policy(self, game, legal: list) -> np.ndarray:
        out = self.policy_net.forward([game], [legal])[0]
        order = {int(i): k for k, i in enumerate(out.legal)}
        return np.asarray([out.probs[order[int(i)]] for i in legal], dtype=np.float64)

    def values(self, game, player_id: int) -> dict:
        out = self.value_net.forward([game])[0]  # value-only; absolute order (players sorted by id)
        seat = _abs_seat(game, player_id)
        search = self.net._head(out.win, out.score)
        return {"win": float(out.win[seat]), "score": float(out.score[seat]), "search": float(search[seat])}

    def search(self, game, readouts: Optional[int] = None) -> dict:
        """A fresh search at ``game`` (as :meth:`LegacySystem.search`): the root evaluated, then batches until it
        has ``readouts`` visits; the root's statistics."""
        if not self.can_search:
            raise ValueError(f"{self.spec}: a policy-only spec (sample=<t>) has no search")
        import torch

        from rl18xx.agent.alphazero.config import SelfPlayConfig
        from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer

        n = len(game.players)
        overrides = dict(self._overrides(n))
        readouts = int(readouts or overrides.get("num_readouts", self.readouts))
        overrides.update(num_readouts=readouts, min_readouts=readouts)
        config = SelfPlayConfig(
            network=self.net, dirichlet_noise_weight=0.0, enable_resign=False, noresign_holdout_rate=0.0,
            player_count_distribution={n: 1.0}, auction_unlock=False, softpick_move_cutoff=0, **overrides,
        )
        player = RustMCTSPlayer(config)
        player.initialize_game(game.pickle_clone())
        tree = player._rust_player
        with torch.no_grad():
            player.tree_search(parallel_readouts=1)  # the root alone (the advisor's generic root evaluation)
            while tree.n_at_root() < readouts:
                before = tree.n_at_root()
                player.tree_search()
                if tree.n_at_root() <= before:
                    break
        st = tree.root_target_stats()
        return {
            "legal": [int(x) for x in st["legal"]],
            "child_n": [float(x) for x in st["child_n"]],
            "child_q": [float(x) for x in st["child_q"]],
            "child_prior": [float(x) for x in st["child_prior"]],
            "root_value": float(st["root_value"]),
            "raw_value": float(st["raw_value"]),
            "root_n": float(st["root_n"]),
            "chosen": int(tree.pick_best_action()),
        }


class MixedSystem(GenericSystem):
    """A legacy policy searched with a title-generic value: ``<legacy policy.pth>+generic:<value.pth>[#win|#score|
    #blend=<w>]`` (``game_network.GameNetwork`` mixes the parts, as the advisor's ``--play-*`` options do). Policy: the
    legacy model's (:meth:`LegacySystem.policy`); values: the generic network's heads; search: as
    :class:`GenericSystem`, with ``--search``'s settings (a spec carries no ``@<n>...`` of its own)."""

    kind = "mixed"

    def __init__(self, spec: str, search: str = DEFAULT_SEARCH, device: str = "cpu"):
        from eval_head_to_head import parse_player

        from rl18xx.agent.alphazero.composite_model import load_policy_value
        from rl18xx.agent.alphazero.game_network import GameNetwork, GenericNetwork
        from rl18xx.agent.alphazero.generic_policy import GENERIC_PREFIX

        self.spec = spec
        self.search_spec = search
        legacy, _, rest = spec.partition("+" + GENERIC_PREFIX)
        value_path, _, head = rest.partition("#")
        self.full_spec = legacy + search
        _, self.overrides = parse_player(self.full_spec, default_readouts=400, num_players=4)
        self.overrides_by_count = {4: dict(self.overrides)}
        self.paths = {"policy": legacy, "value": value_path, "value_head": head or "win"}
        self.can_search = True
        self.prior_temperature = float(self.overrides.get("prior_temperature", 1.0))
        self.readouts = int(self.overrides.get("num_readouts", 400))
        self.legacy = LegacySystem.from_network(load_policy_value(legacy), self.overrides, label=legacy)
        self.policy_net = self.legacy.net
        self.value_net = GenericNetwork(value_path, device)
        self.value_head = self.paths["value_head"]
        # priors at temperature 1: the tree tempers them with the search's ptemp (as GenericSystem)
        self.net = GameNetwork(self.policy_net, self.value_net, policy_temperature=1.0, value_head=self.value_head)

    def policy(self, game, legal: list) -> np.ndarray:
        return self.legacy.policy(game, legal)


def load_system(spec: str, search: str = DEFAULT_SEARCH, device: str = "cpu"):
    from rl18xx.agent.alphazero.generic_policy import GENERIC_PREFIX

    if spec.startswith(GENERIC_PREFIX):
        return GenericSystem(spec, search, device)
    if "+" + GENERIC_PREFIX in spec:
        return MixedSystem(spec, search, device)
    return LegacySystem(spec, search, device)


# ------------------------------------------------------------------------------------------------- scoring
def score_position(system, pos: dict, record: dict, do_search: bool = True, readouts: Optional[int] = None) -> dict:
    t0 = time.time()
    game = rust_game_at(record, pos["action_index"])
    mover = pos["mover"]["player_id"]
    if acting_player(game) != mover:
        raise ValueError(f"{pos['id']}: acting player {acting_player(game)} is not the mover {mover}")
    moves = legal_moves(game)
    legal = [i for i, _ in moves]
    bad = sorted({i for a in pos["bad_moves"] for i in match_indices(game, a, moves)})
    pref = sorted({i for a in pos["preferred_moves"] for i in match_indices(game, a, moves)})
    played = match_indices(game, pos["played"], moves)
    if not bad or not pref:
        raise ValueError(f"{pos['id']}: bad {bad} / preferred {pref} moves not found among the legal moves")
    p = system.policy(game, legal)
    t = float(system.prior_temperature)
    prior = np.power(p, 1.0 / t) if t != 1.0 else p.copy()
    prior = prior / prior.sum()
    k = {index: i for i, index in enumerate(legal)}
    top = np.argsort(-p, kind="stable")[:3]
    res = {
        "id": pos["id"], "weakness": pos["weakness"], "kind": pos["kind"], "criterion": pos["criterion"],
        "game": pos["game"], "site_action_id": pos["site_action_id"], "era": pos["era"], "model": system.spec,
        "num_legal": len(legal),
        "policy": {
            "p_bad": float(sum(p[k[i]] for i in bad)), "p_preferred": float(sum(p[k[i]] for i in pref)),
            "p_played": float(sum(p[k[i]] for i in played)),
            "prior_bad": float(sum(prior[k[i]] for i in bad)), "prior_preferred": float(sum(prior[k[i]] for i in pref)),
            "prior_temperature": t,
            "top": [{"move": describe(move_action(game, legal[j], moves)), "p": float(p[j]),
                     "bad": legal[j] in bad, "preferred": legal[j] in pref} for j in top],
            "argmax_bad": legal[int(np.argmax(p))] in bad, "argmax_preferred": legal[int(np.argmax(p))] in pref,
        },
    }
    before = system.values(game, mover)
    after_bad = system.values(apply_action(game, pos["played"]), mover)
    after_pref = system.values(apply_action(game, pos["preferred_moves"][0]), mover)
    res["value"] = {
        "before": before, "after_bad": after_bad, "after_preferred": after_pref,
        "delta_bad": {h: after_bad[h] - before[h] for h in before},
        "delta_preferred": {h: after_pref[h] - before[h] for h in before},
        # > 0: the value prefers the preferred move to the bad one, 1 ply deep
        "pref_minus_bad": {h: after_pref[h] - after_bad[h] for h in before},
    }
    if do_search:
        s = system.search(game, readouts)
        n = np.asarray(s["child_n"], dtype=np.float64)
        q = np.asarray(s["child_q"], dtype=np.float64)
        sk = {index: i for i, index in enumerate(s["legal"])}
        total = n.sum() or 1.0
        chosen = s["chosen"]

        def best_q(indices):
            vals = [q[sk[i]] for i in indices if i in sk and n[sk[i]] > 0]
            return float(max(vals)) if vals else None

        order = np.argsort(-n, kind="stable")[:3]
        res["search"] = {
            "readouts": int(s["root_n"]), "chosen": describe(move_action(game, chosen, moves)),
            "chosen_bad": chosen in bad, "chosen_preferred": chosen in pref,
            "share_chosen": float(n[sk[chosen]] / total),
            "share_bad": float(sum(n[sk[i]] for i in bad if i in sk) / total),
            "share_preferred": float(sum(n[sk[i]] for i in pref if i in sk) / total),
            "q_bad": best_q(bad), "q_preferred": best_q(pref), "root_value": s["root_value"],
            "top": [{"move": describe(move_action(game, s["legal"][j], moves)), "share": float(n[j] / total),
                     "q": float(q[j]) if n[j] > 0 else None} for j in order],
        }
    res["pass"] = passed(res)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def passed(res: dict) -> Optional[bool]:
    """Whether the position passes: the search's choice when there was a search, else the policy's argmax."""
    s = res.get("search")
    if s is not None:
        return not s["chosen_bad"] if res["criterion"] == "avoid_bad" else bool(s["chosen_preferred"])
    p = res["policy"]
    return not p["argmax_bad"] if res["criterion"] == "avoid_bad" else bool(p["argmax_preferred"])


def summarize(results: list) -> dict:
    """Per weakness: positions, pass rate, share where the search (else the policy's argmax) chose a bad move,
    mean policy probability of the bad and preferred moves, mean 1-ply value gap (preferred - bad)."""
    by = defaultdict(list)
    for r in results:
        by[r["weakness"]].append(r)
        by["all"].append(r)
    out = {}
    for w in list(WEAKNESSES) + ["all"]:
        rs = by.get(w)
        if not rs:
            continue
        searched = [r for r in rs if r.get("search")]
        row = {
            "n": len(rs),
            "pass": statistics.mean(1.0 if r["pass"] else 0.0 for r in rs),
            "policy_argmax_bad": statistics.mean(1.0 if r["policy"]["argmax_bad"] else 0.0 for r in rs),
            "p_bad": statistics.mean(r["policy"]["p_bad"] for r in rs),
            "p_preferred": statistics.mean(r["policy"]["p_preferred"] for r in rs),
            "prior_bad": statistics.mean(r["policy"]["prior_bad"] for r in rs),
            "value_pref_minus_bad": statistics.mean(r["value"]["pref_minus_bad"]["search"] for r in rs),
            "value_pref_minus_bad_win": statistics.mean(r["value"]["pref_minus_bad"].get("win", float("nan"))
                                                        for r in rs),
        }
        if searched:
            row.update(
                search_chose_bad=statistics.mean(1.0 if r["search"]["chosen_bad"] else 0.0 for r in searched),
                search_chose_preferred=statistics.mean(1.0 if r["search"]["chosen_preferred"] else 0.0
                                                       for r in searched),
                search_share_bad=statistics.mean(r["search"]["share_bad"] for r in searched),
                search_share_preferred=statistics.mean(r["search"]["share_preferred"] for r in searched),
            )
        out[w] = row
    return out


def format_summary(summary: dict, label: str = "") -> str:
    lines = [f"== {label}" if label else "== regression set",
             "weakness  | n  | pass | search chose bad | search chose pref | policy argmax bad | p(bad) | p(pref) | "
             "prior(bad) | value pref-bad (search head / win)"]
    for w, r in summary.items():
        sb = f"{r['search_chose_bad']:.2f}" if "search_chose_bad" in r else "-"
        sp = f"{r['search_chose_preferred']:.2f}" if "search_chose_preferred" in r else "-"
        lines.append(f"{w:9s} | {r['n']:2d} | {r['pass']:.2f} | {sb:16s} | {sp:17s} | {r['policy_argmax_bad']:17.2f} | "
                     f"{r['p_bad']:.3f}  | {r['p_preferred']:.3f}   | {r['prior_bad']:.3f}      | "
                     f"{r['value_pref_minus_bad']:+.3f} / {r['value_pref_minus_bad_win']:+.3f}")
    return "\n".join(lines)


def format_positions(by_label: dict) -> str:
    """One row per position, one column per label: PASS / fail, p(bad) and the search's share on the bad moves."""
    labels = list(by_label)
    rows = defaultdict(dict)
    order = []
    for label, results in by_label.items():
        for r in results:
            if r["id"] not in rows:
                order.append(r["id"])
            rows[r["id"]][label] = r
    lines = [" | ".join(["position", "weakness"] + [f"[{i}] {label}" for i, label in enumerate(labels)])]
    for pid in order:
        cells = [pid, next(iter(rows[pid].values()))["weakness"]]
        for label in labels:
            r = rows[pid].get(label)
            if r is None:
                cells.append("-")
                continue
            cell = ("PASS" if r["pass"] else "fail") + f" p(bad) {r['policy']['p_bad']:.2f}"
            if r.get("search"):
                cell += f" N(bad) {r['search']['share_bad']:.2f}"
            cells.append(cell)
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def format_result(r: dict) -> str:
    p = r["policy"]
    v = r["value"]
    s = r.get("search")
    line = (f"{r['id']:11s} {r['weakness']:9s} {'PASS' if r['pass'] else 'fail'} p(bad) {p['p_bad']:.3f} "
            f"p(pref) {p['p_preferred']:.3f} | value(mover) {v['before']['search']:.3f} -> bad "
            f"{v['after_bad']['search']:.3f} / pref {v['after_preferred']['search']:.3f}")
    if s:
        line += (f" | search -> {s['chosen']} ({s['share_chosen']:.2f}); bad {s['share_bad']:.2f} pref "
                 f"{s['share_preferred']:.2f}")
    return line + f" [{r['seconds']}s]"


def score(args) -> list:
    import torch

    torch.set_num_threads(int(args.threads))
    positions = load_fixture(args.fixture)
    if args.weakness:
        positions = [p for p in positions if p["weakness"] in args.weakness]
    if args.ids:
        positions = [p for p in positions if p["id"] in args.ids]
    if args.limit:
        positions = positions[: args.limit]
    system = load_system(args.model, args.search, args.device)
    do_search = not args.no_search and system.can_search
    if not args.no_search and not system.can_search:
        print("note: a policy-only spec (sample=<t>) -- policy and value only", flush=True)
    records = {}
    results = []
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "a") if args.out else None
    for pos in positions:
        gid = pos["game"]
        if gid not in records:
            records[gid] = load_game_record(gid)
        r = score_position(system, pos, records[gid], do_search, args.readouts)
        r["label"] = args.label or args.model
        settings = args.search if system.kind == "legacy" else "@" + system.full_spec.partition("@")[2]
        r["search_settings"] = settings if do_search else None
        results.append(r)
        print(format_result(r), flush=True)
        if out:
            out.write(json.dumps(r) + "\n")
            out.flush()
    if out:
        out.close()
    print(format_summary(summarize(results), args.label or args.model))
    secs = [r["seconds"] for r in results]
    if secs:
        print(f"{len(results)} positions, {statistics.mean(secs):.1f} s per position")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score a model on the regression set")
    s.add_argument("--model", required=True,
                   help="<policy.pth>+<value.pth>[#score|#blend=w] (legacy) or "
                   "generic:<policy.pth>[+generic:<value.pth>][#score|#blend=w] (generic; generic:<ckpt.pth>@<T>: "
                   "prior temperature T), or <legacy policy.pth>+generic:<value.pth>[#score|#blend=w] (mixed)")
    s.add_argument("--search", default=DEFAULT_SEARCH,
                   help=f"search settings in eval_head_to_head's syntax (default {DEFAULT_SEARCH}: the live bots')")
    s.add_argument("--readouts", type=int, default=None, help="override the search budget")
    s.add_argument("--no-search", action="store_true", help="policy and value only")
    s.add_argument("--weakness", nargs="*", choices=WEAKNESSES)
    s.add_argument("--ids", nargs="*")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--fixture", default=str(FIXTURE))
    s.add_argument("--device", default="cpu",
                   help="generic checkpoints' device; legacy ones load where the checkpointer puts them (the GPU when "
                   "one is visible: CUDA_VISIBLE_DEVICES= for CPU)")
    s.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4") or 4))
    s.add_argument("--label", default=None)
    s.add_argument("--out", default=None, help="append each position's result to this JSONL")
    sm = sub.add_parser("summary", help="summarize result JSONL files (one table per label)")
    sm.add_argument("results", nargs="+")
    sm.add_argument("--positions", action="store_true", help="also a per-position table: pass / fail per label")
    sub.add_parser("build", help="rebuild data/regression_set.json from the curated list")
    ls = sub.add_parser("list", help="list the positions")
    ls.add_argument("--fixture", default=str(FIXTURE))
    cv = sub.add_parser("convert", help="site exports -> data/caleb_games/<id>.json.gz")
    cv.add_argument("exports", nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "score":
        score(args)
    elif args.cmd == "summary":
        import gzip

        by_label = defaultdict(list)
        for path in args.results:
            for line in (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)):
                if line.strip():
                    r = json.loads(line)
                    by_label[r.get("label") or r.get("model")].append(r)
        for label, rs in by_label.items():
            print(format_summary(summarize(rs), label))
        if args.positions:
            print(format_positions(by_label))
    elif args.cmd == "build":
        build()
    elif args.cmd == "list":
        for p in load_fixture(args.fixture):
            print(f"{p['id']:12s} {p['weakness']:9s} {p['kind']:8s} {p['criterion']:16s} {p['era']:10s} "
                  f"{p['mover']['name']:6s} {p['played_desc']:16s} | {p['context']['round']} | {p['note']}")
    elif args.cmd == "convert":
        convert(args.exports)


if __name__ == "__main__":
    main()
