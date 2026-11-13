"""Auction prices in gen5's saved games (1 in 20), split by buyer: the learner's seats vs the
league's. history.jsonl's auction_price_* mixes both, and the frozen league members overpay
(MH 177 vs the learner's 138 over u1-110), so it can't show whether the learner moves.
Usage: python3 learner_prices.py [window=50]"""
import collections, json, sys

G = "/home/revys/longrun/work/lr1/model_checkpoints_pg/gen5_20261107"
HUMANS = {"BO": 222, "CA": 208, "CS": 48, "DH": 87, "MH": 135}
W = int(sys.argv[1]) if len(sys.argv) > 1 else 50
pr = collections.defaultdict(list)
cnt = collections.Counter()
for line in open(G + "/games.jsonl"):
    r = json.loads(line)
    if r.get("start") != "auction" or r.get("termination") == "auction_stall" or r["update"] < 1:
        continue
    d = json.load(open(f"{G}/games/{r['name']}.json"))
    pid = [p["id"] for p in d["players"]]
    sales = {}
    for a in d["actions"]:
        if a.get("type") == "bid":
            co, p = a["company"], int(a["price"])
            if co not in sales or p > sales[co][1]:
                sales[co] = [a["entity"], p]
    w = (r["update"] - 1) // W
    ls = set(r["learner_seats"])
    for co, (e, p) in sales.items():
        who = "L" if pid.index(e) in ls else "O"
        pr[(w, who, co)].append(p)
        cnt[(w, co, who)] += 1
        cnt[(w, co, "seat")] += len(ls) / 4
print(f"buyer  updates    " + " ".join(f"{c:>9s}" for c in HUMANS) + "   (price, share of sales; humans "
      + " ".join(str(v) for v in HUMANS.values()) + ")")
for w in sorted({k[0] for k in pr}):
    for who, name in (("L", "learner"), ("O", "league")):
        cells = []
        for co in HUMANS:
            v = pr[(w, who, co)]
            n = cnt[(w, co, "L")] + cnt[(w, co, "O")]
            cells.append(f"{sum(v)/len(v):4.0f} {len(v)/n:.2f}" if v else "    -    ")
        print(f"{name:7s} u{w*W+1:4d}-{w*W+W:<4d} " + " ".join(f"{c:>9s}" for c in cells))
