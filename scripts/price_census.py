"""Census of human price decisions (Bid / BuyTrain / BuyCompany) in 1830.

Replays every cleaned human game through the Rust engine and, at each
price-bearing action, records the price the human paid alongside the legal
``price_range`` the factored enumerator hands MCTS. Writes one CSV row per
decision and prints the summary used in docs/arbitrary_price_actions.md.

    uv run python scripts/price_census.py --games-dir human_games/1830_clean --out price_census.csv
"""

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

TRAIN_FACE = {"2": 80, "3": 180, "4": 300, "5": 450, "6": 630, "D": 1100}
COMPANY_FACE = {"SV": 20, "CS": 40, "DH": 70, "MH": 110, "CA": 160, "BO": 220}


def _owner_sym(owner):
    if isinstance(owner, str):
        return owner
    if getattr(owner, "name", "") == "The Depot":
        return "The Depot"
    return getattr(owner, "sym", None) or getattr(owner, "id", None) or str(owner)


def census_game(path):
    from engine_rs import BaseGame as RustGame
    from rl18xx.rust_adapter import RustGameAdapter

    g = json.load(open(path))
    if "players" not in g:  # cleaning-rejection stub
        return [], None
    adapter = RustGameAdapter(RustGame({int(p["id"]): str(p["name"]) for p in g["players"]}))
    game_id = os.path.basename(path)[: -len(".json")]
    rows = []
    for a in g["actions"]:
        t = a.get("type")
        if t in ("bid", "buy_train", "buy_company") and a.get("price") is not None:
            choices = adapter._game.get_factored_choices()
            row = {"game": game_id, "np": len(g["players"]), "type": t, "price": int(a["price"])}
            match = None
            if t in ("bid", "buy_company"):
                want = "Bid" if t == "bid" else "BuyCompany"
                row["what"], row["face"] = a["company"], COMPANY_FACE.get(a["company"])
                match = next(
                    (c for c in choices if c["type"] == want and c["entity"].get("private") == a["company"]), None
                )
            else:
                name = a["train"].split("-")[0]
                seller = _owner_sym(adapter.train_by_id(a["train"]).owner)
                row.update(what=name, face=TRAIN_FACE.get(name), seller=seller, buyer=a.get("entity"))
                for c in choices:
                    src = c["entity"].get("source")
                    if c["type"] == "BuyTrain" and c["entity"].get("train") == name and (
                        src == seller or (seller == "The Depot" and src == "depot")
                    ):
                        match = c
                        break
                rg = adapter._game
                buyer = rg.corporation_by_id(a["entity"])
                row.update(buyer_cash=buyer.cash, buyer_pres=buyer.owner_id_str, buyer_ntrains=len(buyer.trains))
                if seller != "The Depot":
                    sc = rg.corporation_by_id(seller)
                    row.update(seller_cash=sc.cash, seller_pres=sc.owner_id_str, seller_ntrains=len(sc.trains))
                row["next_depot"] = rg.depot.trains[0].price if rg.depot.trains else None
            if match is not None and match.get("price_range") is not None:
                row["lo"], row["hi"] = int(match["price_range"][0]), int(match["price_range"][1])
            rows.append(row)
        try:
            adapter.process_action(a)
        except Exception as e:
            return rows, f"{game_id}: {type(e).__name__}: {e}"
    return rows, None


def summarize(d):
    d = d.dropna(subset=["lo", "hi"])
    free = d[d.hi > d.lo]
    print("decisions:", d.type.value_counts().to_dict(), " with a free price:", free.type.value_counts().to_dict())

    b = free[free.type == "bid"]
    over = b.price - b.lo
    print(
        f"\nBid (n={len(b)}): at min bid {np.mean(over == 0):.3f}; on the $5 ladder {np.mean(over % 5 == 0):.3f}; "
        f"max over-min ${over.max():.0f}; at all-in {np.mean(b.price == b.hi):.4f}; "
        f"median range ${(b.hi - b.lo).median():.0f}"
    )

    t = free[(free.type == "buy_train") & (free.seller != "The Depot")]
    mode = np.select([t.price == t.lo, t.price == t.hi, t.price == t.hi - 1], ["min", "max", "max-1"], "interior")
    print(f"\nCross-corp BuyTrain (n={len(t)}): same president {np.mean(t.buyer_pres == t.seller_pres):.3f}")
    print("  mode shares:", pd.Series(mode).value_counts(normalize=True).round(3).to_dict())
    u = (t.price - t.lo) / (t.hi - t.lo)
    print("  normalized position (p-lo)/(hi-lo), 10 bins:", np.histogram(u, bins=10, range=(0, 1))[0].tolist())

    c = free[free.type == "buy_company"]
    print(f"\nBuyCompany (n={len(c)}): at max {np.mean(c.price == c.hi):.3f}; at min {np.mean(c.price == c.lo):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", default="human_games/1830_clean")
    ap.add_argument("--out", default="price_census.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.games_dir, "*.json")))[: args.limit]
    all_rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows, err in ex.map(census_game, paths, chunksize=8):
            all_rows.extend(rows)
            if err:
                errors.append(err)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"games={len(paths)} decisions={len(df)} replay errors={len(errors)} -> {args.out}")
    for e in errors[:10]:
        print("  ", e)
    summarize(df)
