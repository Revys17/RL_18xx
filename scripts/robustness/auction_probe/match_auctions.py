"""Auction consistency per private, by side: eval collections (games.jsonl + games/) or the human corpus."""
import json, sys, collections
from pathlib import Path
FACE = {"SV": 20, "CS": 40, "DH": 70, "MH": 110, "CA": 160, "BO": 220}
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

def eval_games(E):
    for l in open(E / "games.jsonl"):
        r = json.loads(l)
        yield json.load(open(E / "games" / f"{r['game_idx']}.json")), (lambda e: int(e) - 1), r["seats"], r["win_share"]

def human_games(D):
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        if not isinstance(d, dict) or "players" not in d or not d.get("finished", True):
            continue
        if len(d["players"]) != 4 or not d.get("result") or any(a.get("type") == "undo" for a in d["actions"]):
            continue
        ids = [p["id"] for p in d["players"]]
        res = [float(d["result"][str(i)]) for i in ids]
        best = max(res); w = [1.0 if v >= best else 0.0 for v in res]; t = sum(w)
        yield d["actions"], (lambda e, ids=ids: ids.index(int(e))), ["H"] * 4, [x / t for x in w]

def analyse(name, games):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    n = inc = 0; side_win = collections.defaultdict(list); bargain_side = collections.Counter()
    for acts, seat_of, seats, win in games:
        n += 1
        for s in range(4): side_win[seats[s]].append(win[s])
        bids = collections.defaultdict(list)
        for a in acts:
            if a.get("type") == "bid" and a.get("entity_type") == "player":
                bids[a["company"]].append((seat_of(a["entity"]), int(a["price"])))
        sold = {}
        for c in FACE:
            if not bids[c]: continue
            seat, price = max(bids[c], key=lambda x: x[1])
            bidders = {s for s, _ in bids[c]}
            con = len(bidders) >= 2
            sold[c] = (seat, price, con)
            d = per[c]; side = seats[seat]
            d["con"].append(con); d["over"].append(price - FACE[c])
            d[f"win_{side}_{'con' if con else 'unc'}"].append(win[seat])
            d[f"n_{side}"].append(1)
        barg = [(c, s) for c, (s, p, con) in sold.items() if c != "SV" and not con and p <= FACE[c] + 5]
        over = [c for c, (s, p, con) in sold.items() if p >= FACE[c] + 30]
        if barg and over:
            inc += 1
            for c, s in barg: bargain_side[seats[s]] += 1
    sides = sorted(side_win)
    print(f"== {name}: {n} games; side win rates " + ", ".join(f"{s} {mean(side_win[s]):.3f}" for s in sides))
    hdr = "   private contested  over  " + "  ".join(f"{s}: buys  win unc (n)   win con (n)" for s in sides)
    print(hdr)
    for c in FACE:
        d = per[c]
        if not d["con"]: continue
        cols = []
        for s in sides:
            u, k = d[f"win_{s}_unc"], d[f"win_{s}_con"]
            cols.append(f"{s}: {len(d[f'n_{s}'])/len(d['con']):4.0%}  {mean(u):.2f} ({len(u):3d})   {mean(k):.2f} ({len(k):3d})")
        print(f"   {c:7} {mean(d['con']):9.0%} {mean(d['over']):5.0f}  " + "  ".join(cols))

for arg in sys.argv[1:]:
    p = Path(arg)
    analyse(p.name, eval_games(p) if (p / "games.jsonl").exists() else human_games(p))
