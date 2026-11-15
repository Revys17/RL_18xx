"""Replay an 1830 game in the Python reference engine and extract the events behind the bots' five weaknesses
(docs/bot_weaknesses_2026_11.md): auction bids and sales, presidency changes (dumps, take-overs), share buys and
sales, train sales between corporations, corporation turns (trainless ORs), forced contributions and bankruptcies.

A port of the 2026-11-14 analysis (session 48dd27c6: ``timeline.py``, ``events.py``, ``corpus.py``); the numbers in
docs/bot_weaknesses_2026_11.md came from that code, so keep the logic in step with it (tests compare game 21 of the
regression set's games with the analysis' counts). Players are "Player N" with ids 1..N (every engine game here is
built so); events name players by id. Log lines are the Python engine's.

    rows, error, game = replay_rows(actions, num_players, optional_rules)
    events = extract(rows, is_bot=lambda pid: False)
"""

from __future__ import annotations

import copy
import re
from typing import Callable, Iterable, Optional

FACE = {"SV": 20, "CS": 40, "DH": 70, "MH": 110, "CA": 160, "BO": 220}
CO_BY_NAME = {
    "Schuylkill Valley": "SV",
    "Champlain & St.Lawrence": "CS",
    "Delaware & Hudson": "DH",
    "Mohawk & Hudson": "MH",
    "Camden & Amboy": "CA",
    "Baltimore & Ohio": "BO",
}
CORPS = ["PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M"]
TRAIN_PRICE = {"2": 80, "3": 180, "4": 300, "5": 450, "6": 630, "D": 1100}
PHASE_ORDER = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "D": 7}

_P = r"Player (\d+)"
RE_BID = re.compile(rf"^{_P} bids \$(\d+) for (.+)$")
RE_WIN = re.compile(rf"^{_P} wins the auction for (.+) with (?:a|the only) bid of \$(\d+)$")
RE_BUYCO = re.compile(rf"^{_P} buys (.+) for \$(\d+)$")
RE_OPERATES = re.compile(rf"^{_P} operates (\S+)$")
RE_CONTRIB = re.compile(rf"^{_P} contributes \$(\d+)$")
RE_BANKRUPT = re.compile(rf"^-- {_P} goes bankrupt")
RE_SELL = re.compile(rf"^{_P} sells (?:a 10% share|(\d+) shares) of (\S+) and receives \$(\d+)$")
RE_TRAIN = re.compile(r"^(\S+) buys an? (\w+) train for \$(\d+) from (.+)$")
RE_CO_TO_CORP = re.compile(rf"^(\S+) buys (.+) from {_P} for \$(\d+)$")
RE_RUST = re.compile(r"^-- Event: (\w+) trains rust \((.*)\) --$")
RE_PHASE = re.compile(r"^-- Phase (\w+)")
RE_RUNS = re.compile(r"^(\S+) runs an? (\w+) train for \$(\d+)")
RE_HEADING = re.compile(r"^-- (Stock Round|Operating Round) ([\d.]+)")
RE_SR = re.compile(r"^-- Stock Round (\d+)")
RE_SHARE_BUY = re.compile(r"^Player \d+ buys an? (\d+)% share of (\S+) from the (IPO|market) for \$(\d+)")
_DEBUG = re.compile(r"^(adding share|setting <)| object at 0x")
ENGINE_RULES = ("multiple_brown_from_ipo", "optional_6_train")


def _call(v):
    return v() if callable(v) else v


# ------------------------------------------------------------------------------------------- replay
def new_python_game(num_players: int, optional_rules: Iterable[str] = (), auction_unlock: bool = False):
    """The Python reference engine's 1830 with players "Player 1".."Player N"."""
    from rl18xx.game.gamemap import GameMap

    rules = [r for r in (optional_rules or []) if r in ENGINE_RULES]
    game = GameMap().game_by_title("1830")({i: f"Player {i}" for i in range(1, num_players + 1)}, optional_rules=rules)
    game.auction_unlock = bool(auction_unlock)
    return game


def state(game) -> dict:
    """The position after an action, as the event extraction reads it."""
    corps = list(game.corporations)
    players = []
    for p in sorted(game.players, key=lambda p: p.id):
        players.append(
            {
                "id": p.id,
                "cash": p.cash,
                "value": game.player_value(p),
                "shares": {c.name: p.percent_of(c) for c in corps if p.percent_of(c)},
                "pres": [c.name for c in corps if c.owner is p],
                "companies": [c.sym for c in p.companies],
                "bankrupt": bool(getattr(p, "bankrupt", False)),
            }
        )
    crows = {}
    for c in corps:
        price = c.share_price
        par = c.par_price()
        try:
            ipo = int(c.num_ipo_shares() * c.share_percent)
        except Exception:
            ipo = None
        try:
            market = int(c.num_market_shares() * c.share_percent)
        except Exception:
            market = None
        crows[c.name] = {
            "ipoed": bool(c.ipoed),
            "floated": bool(_call(c.floated)),
            "closed": bool(_call(getattr(c, "closed", False))),
            "pres": (c.owner.id if c.owner is not None and c.owner.is_player() else None),
            "par": par.price if par else None,
            "price": price.price if price else None,
            "cash": c.cash,
            "ipo": ipo,
            "market": market,
            "trains": [t.name for t in c.trains],
        }
    rnd = game.round
    kind = type(rnd).__name__ if rnd else None
    return {
        "round": kind,
        "phase": game.phase.name,
        "players": players,
        "corps": crows,
        "depot": [t.name for t in game.depot.depot_trains()],
        "finished": bool(game.finished),
    }


def actor_of(game, action: dict):
    """``(entity name, player id behind it)`` of an action before it is applied."""
    et = action.get("entity_type")
    ent = action.get("entity")
    if et == "player":
        return str(ent), int(ent)
    if et == "corporation":
        c = game.corporation_by_id(ent)
        return ent, (c.owner.id if c and c.owner is not None and c.owner.is_player() else None)
    if et == "company":
        c = game.company_by_id(ent)
        o = c.owner if c else None
        if o is not None and hasattr(o, "is_player") and o.is_player():
            return ent, o.id
        if o is not None and hasattr(o, "owner") and o.owner is not None:
            return ent, o.owner.id
        return ent, None
    return str(ent), None


def replay_rows(actions: list, num_players: int, optional_rules: Iterable[str] = (), auction_unlock: bool = False,
                site_ids: Optional[list] = None):
    """``(rows, error, game)``: the game replayed action by action; ``rows[0]`` is the initial state (``k`` -1),
    then one row per action: the action, its actor, the round it was played in, its log lines and the state after.
    ``site_ids`` (one per action) label the rows (``site_id``; else the action's index). ``error`` names the
    first action the engine rejected (the rows stop there)."""
    from rl18xx.game.engine.actions import BaseAction

    py = new_python_game(num_players, optional_rules, auction_unlock)
    rows = [{"k": -1, "state": state(py)}]
    logn = len(py.log)
    err = None
    for k, a in enumerate(actions):
        try:
            ent, pl = actor_of(py, a)
        except Exception:
            ent, pl = None, None
        pre_round = type(py.round).__name__ if py.round else None
        try:
            desc = py.round_description(pre_round) if pre_round in ("Stock", "Operating", "Auction") else None
        except Exception:
            desc = None
        clean = {kk: vv for kk, vv in copy.deepcopy(a).items() if kk not in ("created_at", "user")}
        try:
            py.process_action(BaseAction.action_from_dict(clean, py))
        except Exception as e:
            err = f"action {k} ({a.get('type')}): {type(e).__name__}: {e}"
            break
        lines = [str(e.message) for e in py.log[logn:] if not _DEBUG.search(str(e.message))]
        logn = len(py.log)
        row = {"k": k, "action": a, "entity": ent, "player": pl, "round_before": pre_round,
               "round_desc_before": desc, "log": lines, "state": state(py)}
        if site_ids is not None:
            row["site_id"] = site_ids[k]
        rows.append(row)
    return rows, err, py


def start_k(rows: list, start: Optional[str]) -> Optional[int]:
    """The first action index the recorded game's own players chose: 0 for a game from the auction (or a game with
    no start label), the first action of Stock Round 1 for a human / random post-auction start, and of Stock Round
    N for ``midgame:...:srN``; None if the game never got there."""
    if not start or start == "auction":
        return 0
    if start.startswith("human") or start.startswith("random"):
        sr = 1
    elif start.startswith("midgame"):
        m = re.search(r"sr(\d+)", start)
        sr = int(m.group(1)) if m else 1
    else:
        return 0
    for r in rows[1:]:
        for ln in r["log"]:
            m = RE_SR.match(ln)
            if m and int(m.group(1)) == sr:
                return r["k"] + 1
    return None


# ------------------------------------------------------------------------------------------- events
def _depot_next(st):
    d = st.get("depot") or []
    return d[0] if d else None


def _corp_snapshot(st, c):
    x = st["corps"][c]
    return {"cash": x["cash"], "trains": list(x["trains"]), "price": x["price"], "pres": x["pres"],
            "floated": x["floated"], "ipo": x["ipo"], "market": x["market"]}


def _player(st, pid):
    for p in st["players"]:
        if p["id"] == pid:
            return p
    return None


def _holdings(st, pid, corp):
    p = _player(st, pid)
    return p["shares"].get(corp, 0) if p else 0


def extract(rows: list, is_bot: Callable[[int], bool] = lambda pid: False) -> dict:
    """The events of a replayed game (``replay_rows``' rows). ``is_bot(pid)`` tags actors (the site games' bots).
    Events carry ``site_id`` (the row's ``site_id``, else its action index ``k``)."""
    ev = {"auction": {}, "auction_end": None, "pres_changes": [], "contributions": [], "or_sales": [],
          "bankruptcies": [], "own_sales": [], "buys": [], "train_sales": [], "private_sales": [], "corp_turns": [],
          "rusts": [], "phases": [], "pars": [], "game_end": None}
    bids = {}  # company -> [(pid, price, site id)]
    cur_corp = cur_or = cur_sr = None
    prev = rows[0]["state"]
    turn_trains_start, turn_runs, turn_sid = None, [], None

    def end_turn(st):
        if cur_corp is None or st is None or cur_corp not in st["corps"]:
            return
        x = st["corps"][cur_corp]
        ev["corp_turns"].append({
            "site_id": turn_sid, "corp": cur_corp, "or": cur_or, "pres": x["pres"],
            "pres_bot": is_bot(x["pres"]) if x["pres"] else None, "trains_start": turn_trains_start,
            "trains_end": list(x["trains"]), "ran": bool(turn_runs), "revenue": sum(v for _, v in turn_runs),
            "cash_end": x["cash"], "price": x["price"], "phase": st["phase"],
        })

    for r in rows[1:]:
        st = r["state"]
        a = r["action"]
        actor = r.get("player")
        sid = r.get("site_id", r["k"])
        lines = r["log"]
        # --- auction
        for ln in lines:
            m = RE_BID.match(ln)
            if m:
                co = CO_BY_NAME.get(m.group(3))
                bids.setdefault(co, []).append((int(m.group(1)), int(m.group(2)), sid))
                continue
            m = RE_WIN.match(ln)
            if m:
                co = CO_BY_NAME.get(m.group(2))
                ev["auction"][co] = {"buyer": int(m.group(1)), "price": int(m.group(3)), "site_id": sid, "how": "won"}
                continue
            if r["round_before"] == "Auction" or r["round_before"] is None:
                m = RE_BUYCO.match(ln)
                if m and m.group(2) in CO_BY_NAME:
                    co = CO_BY_NAME[m.group(2)]
                    ev["auction"][co] = {"buyer": int(m.group(1)), "price": int(m.group(3)), "site_id": sid,
                                         "how": "bought"}
        if ev["auction_end"] is None and st["round"] == "Stock":
            ev["auction_end"] = {"k": r["k"], "site_id": sid, "players": {
                p["id"]: {"cash": p["cash"], "companies": p["companies"], "value": p["value"]} for p in st["players"]}}
        # --- headings, operating turns, money
        for ln in lines:
            m = RE_HEADING.match(ln)
            if m:
                if cur_corp is not None:
                    end_turn(st)
                    cur_corp = None
                if m.group(1) == "Stock Round":
                    cur_sr, cur_or = m.group(2), None
                else:
                    cur_or = m.group(2)
            m = RE_OPERATES.match(ln)
            if m:
                if cur_corp is not None:
                    end_turn(st)  # the previous corporation's turn ended with this row's action
                cur_corp = m.group(2)
                turn_trains_start = list(st["corps"][cur_corp]["trains"]) if cur_corp in st["corps"] else []
                turn_runs, turn_sid = [], sid
            m = RE_RUNS.match(ln)
            if m:
                turn_runs.append((m.group(2), int(m.group(3))))
            m = RE_CONTRIB.match(ln)
            if m:
                pid = int(m.group(1))
                ev["contributions"].append({"site_id": sid, "player": pid, "bot": is_bot(pid),
                                            "amount": int(m.group(2)), "corp": cur_corp, "or": cur_or,
                                            "player_cash_before": (_player(prev, pid) or {}).get("cash")})
            m = RE_BANKRUPT.match(ln)
            if m:
                pid = int(m.group(1))
                ev["bankruptcies"].append({"site_id": sid, "player": pid, "bot": is_bot(pid), "corp": cur_corp,
                                           "or": cur_or, "corp_state": _corp_snapshot(prev, cur_corp)
                                           if cur_corp in prev["corps"] else None})
            m = RE_SELL.match(ln)
            if m and r["round_before"] == "Operating":
                pid = int(m.group(1))
                ev["or_sales"].append({"site_id": sid, "player": pid, "bot": is_bot(pid), "corp_sold": m.group(3),
                                       "n": int(m.group(2)) if m.group(2) else 1, "cash": int(m.group(4)),
                                       "for_corp": cur_corp, "or": cur_or})
            m = RE_TRAIN.match(ln)
            if m:
                buyer, ttype, price, seller = m.group(1), m.group(2), int(m.group(3)), m.group(4)
                if seller != "The Depot":
                    ev["train_sales"].append({
                        "site_id": sid, "buyer": buyer, "seller": seller, "train": ttype, "price": price,
                        "buyer_pres": prev["corps"].get(buyer, {}).get("pres"),
                        "seller_pres": prev["corps"].get(seller, {}).get("pres"),
                        "buyer_cash_before": prev["corps"].get(buyer, {}).get("cash"), "or": cur_or,
                        "phase": prev["phase"]})
            m = RE_CO_TO_CORP.match(ln)
            if m and m.group(2) in CO_BY_NAME:
                co = CO_BY_NAME[m.group(2)]
                ev["private_sales"].append({"site_id": sid, "corp": m.group(1), "company": co,
                                            "seller": int(m.group(3)), "seller_bot": is_bot(int(m.group(3))),
                                            "price": int(m.group(4)), "face": FACE[co],
                                            "corp_pres": prev["corps"].get(m.group(1), {}).get("pres")})
            m = RE_RUST.match(ln)
            if m:
                ev["rusts"].append({"site_id": sid, "train": m.group(1), "victims": m.group(2), "or": cur_or})
            m = RE_PHASE.match(ln)
            if m:
                ev["phases"].append({"site_id": sid, "phase": m.group(1)})
        # --- presidency changes
        for c in CORPS:
            old, new = prev["corps"][c]["pres"], st["corps"][c]["pres"]
            if old == new:
                continue
            if old is None:
                if a.get("type") == "par":
                    ev["pars"].append({"site_id": sid, "corp": c, "player": new, "bot": is_bot(new),
                                       "price": st["corps"][c]["par"], "sr": cur_sr})
                continue
            if new is None:
                continue
            typ = a.get("type")
            if typ == "sell_shares" and actor == old:
                cause = "sold"  # the old president sold: a dump onto the new one
            elif typ == "buy_shares" and actor == new:
                cause = "bought"
            elif any(RE_BANKRUPT.match(x) for x in lines):
                cause = "bankruptcy"
            else:
                cause = typ or "other"
            ev["pres_changes"].append({
                "site_id": sid, "k": r["k"], "corp": c, "from": old, "to": new, "from_bot": is_bot(old),
                "to_bot": is_bot(new), "cause": cause, "round": r["round_before"], "round_desc": r["round_desc_before"],
                "sr": cur_sr, "or": cur_or, "phase": st["phase"], "corp_before": _corp_snapshot(prev, c),
                "corp_after": _corp_snapshot(st, c), "to_cash": (_player(st, new) or {}).get("cash"),
                "to_value": (_player(st, new) or {}).get("value"), "to_pct": _holdings(st, new, c),
                "from_pct_after": _holdings(st, old, c), "depot_next": _depot_next(st),
            })
        # --- share sales and buys in the stock round
        if a.get("type") == "sell_shares" and actor is not None:
            c = a.get("shares", [""])[0].split("_")[0] if a.get("shares") else a.get("corporation")
            if c in prev["corps"]:
                ev["own_sales"].append({
                    "site_id": sid, "k": r["k"], "player": actor, "bot": is_bot(actor), "corp": c,
                    "was_pres": prev["corps"][c]["pres"] == actor, "pct_before": _holdings(prev, actor, c),
                    "pct_after": _holdings(st, actor, c), "pres_after": st["corps"][c]["pres"],
                    "round": r["round_before"], "sr": cur_sr, "corp_before": _corp_snapshot(prev, c),
                    "phase": prev["phase"]})
        if a.get("type") == "buy_shares" and actor is not None:
            shares = a.get("shares") or []
            c = shares[0].split("_")[0] if shares else None
            if c in prev["corps"]:
                src, paid = None, None
                for ln in lines:
                    mm = RE_SHARE_BUY.match(ln)
                    if mm:
                        src, paid = mm.group(3), int(mm.group(4))
                pres = prev["corps"][c]["pres"]
                ev["buys"].append({
                    "src": src, "paid": paid, "site_id": sid, "k": r["k"], "player": actor, "bot": is_bot(actor),
                    "corp": c, "pres": pres, "pres_bot": is_bot(pres) if pres else None,
                    "pct_before": _holdings(prev, actor, c), "pct_after": _holdings(st, actor, c),
                    "pres_after": st["corps"][c]["pres"], "corp_before": _corp_snapshot(prev, c),
                    "phase": prev["phase"], "sr": cur_sr, "depot_next": _depot_next(prev),
                    "round": r["round_before"]})
        prev = st
    last = rows[-1]["state"]
    ev["game_end"] = {"finished": last["finished"],
                      "players": {p["id"]: {"cash": p["cash"], "value": p["value"], "bankrupt": p["bankrupt"]}
                                  for p in last["players"]}}
    for co, info in ev["auction"].items():
        b = bids.get(co, [])
        info["bidders"] = sorted({pid for pid, _, _ in b})
        info["nbids"] = len(b)
        info["contested"] = len(info["bidders"]) >= 2
        info["face"] = FACE.get(co)
        info["buyer_bot"] = is_bot(info["buyer"])
    return ev
