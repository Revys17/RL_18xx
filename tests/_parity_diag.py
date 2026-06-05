"""Ad-hoc parity diagnostic for the Rust vs Python factored action helper.

NOT a pytest test (underscore prefix => not collected). Walks the parity seeds
in lockstep and classifies every divergence by category and direction, with
exact price-range comparison (the strict test's _canonical_key drops price).

Usage:
    uv run python tests/_parity_diag.py            # summary across all seeds
    uv run python tests/_parity_diag.py --detail    # + first-occurrence context per category
"""

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.action_helper import ActionHelper  # noqa: E402
from rl18xx.game.factored_action_helper import FactoredActionHelper  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]


def _key(la):
    """Categorical key + price_range. Works on both LegalAction (py, attrs)
    and dict (rust adapter). Returns (catkey_tuple, price_range_or_None)."""
    if isinstance(la, dict):
        t = la.get("type")
        e = la.get("entity", {})
        p = la.get("params", {})
        pr = la.get("price_range")
    else:
        t = la.type
        e = la.entity
        p = la.params
        pr = la.price_range
    pr = tuple(pr) if pr is not None else None

    if t == "Pass":
        k = ("Pass",)
    elif t == "Bid":
        k = ("Bid", e.get("private"))
    elif t == "Par":
        k = ("Par", e.get("corp"), p.get("par_price"))
    elif t == "BuyShares":
        k = ("BuyShares", e.get("corp"), p.get("source"), int(p.get("percent", 0)))
    elif t == "CompanyBuyShares":
        k = ("CompanyBuyShares", e.get("private"), e.get("corp"), p.get("source"))
    elif t == "SellShares":
        k = ("SellShares", e.get("corp"), int(p.get("count", 0)))
    elif t == "PlaceToken":
        k = ("PlaceToken", p.get("hex"), p.get("city"), p.get("slot"))
    elif t == "LayTile":
        k = ("LayTile", p.get("hex"), p.get("tile"), p.get("rotation"))
    elif t == "BuyTrain":
        k = ("BuyTrain", e.get("source"), e.get("train"), e.get("exchange"))
    elif t == "DiscardTrain":
        k = ("DiscardTrain", p.get("train"))
    elif t == "Dividend":
        k = ("Dividend", p.get("kind"))
    elif t == "BuyCompany":
        k = ("BuyCompany", e.get("private"))
    elif t == "RunRoutes":
        k = ("RunRoutes",)
    elif t == "Bankrupt":
        k = ("Bankrupt",)
    else:
        k = (t,)
    return k, pr


def _category(catkey):
    """Bucket a categorical key into a coarse category name for aggregation."""
    return catkey[0]


def walk(seed, max_steps):
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    py = GameMap().game_by_title("1830")(names)
    rust = RustGame(names)
    adapter = RustGameAdapter(rust)
    hp = FactoredActionHelper()
    ah = ActionHelper()

    # category -> {"py_only": count, "rust_only": count, "price": count}
    cat = defaultdict(lambda: defaultdict(int))
    first = {}  # (category, direction) -> context string

    for step in range(max_steps):
        if py.finished:
            break
        py_list = hp.get_choices(py)
        rust_list = adapter.get_factored_choices()
        py_map = {}
        for la in py_list:
            k, pr = _key(la)
            py_map[k] = pr
        rust_map = {}
        for la in rust_list:
            k, pr = _key(la)
            rust_map[k] = pr

        py_keys = set(py_map)
        rust_keys = set(rust_map)
        py_only = py_keys - rust_keys
        rust_only = rust_keys - py_keys
        price_mismatch = {k for k in (py_keys & rust_keys) if py_map[k] != rust_map[k]}

        if py_only or rust_only or price_mismatch:
            rnd = type(py.round).__name__
            stp = type(py.active_step()).__name__ if py.active_step() else None
            phase = getattr(getattr(py, "phase", None), "name", "?")
            ent = py.current_entity
            ent_s = getattr(ent, "name", getattr(ent, "id", str(ent)))
            ctx = f"seed={seed} step={step} [{rnd}/{stp}] phase={phase} entity={ent_s}"
            for k in py_only:
                c = _category(k)
                cat[c]["py_only"] += 1
                first.setdefault((c, "py_only"), (ctx, sorted(py_only), sorted(rust_only)))
            for k in rust_only:
                c = _category(k)
                cat[c]["rust_only"] += 1
                first.setdefault((c, "rust_only"), (ctx, sorted(py_only), sorted(rust_only)))
            for k in price_mismatch:
                c = _category(k)
                cat[c]["price"] += 1
                first.setdefault((c, "price"), (ctx, [(k, "py=", py_map[k], "rust=", rust_map[k])], []))

        legacy = ah.get_all_choices_limited(py)
        if not legacy:
            break
        action = rng.choice(legacy).to_dict()
        try:
            py = py.process_action(action)
        except Exception as exc:
            cat["__PY_ERR__"]["py_only"] += 1
            first.setdefault(("__PY_ERR__", "py_only"), (f"seed={seed} step={step}: {exc}", [], []))
            break
        try:
            adapter.process_action(action)
        except Exception as exc:
            cat["__RUST_ERR__"]["rust_only"] += 1
            first.setdefault(("__RUST_ERR__", "rust_only"), (f"seed={seed} step={step}: {exc}", [], []))
            break
    return cat, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args()

    total = defaultdict(lambda: defaultdict(int))
    all_first = {}
    per_seed_steps = {}
    for seed in args.seeds:
        cat, first = walk(seed, args.steps)
        seed_total = sum(sum(d.values()) for d in cat.values())
        per_seed_steps[seed] = seed_total
        for c, d in cat.items():
            for dir_, n in d.items():
                total[c][dir_] += n
        for k, v in first.items():
            all_first.setdefault(k, v)

    print("=" * 70)
    print(f"PARITY DIVERGENCE SUMMARY (seeds={args.seeds}, max_steps={args.steps})")
    print("=" * 70)
    print(f"{'category':<22} {'py_only':>8} {'rust_only':>10} {'price':>7}")
    print("-" * 70)
    grand = 0
    for c in sorted(total):
        d = total[c]
        po, ro, pm = d.get("py_only", 0), d.get("rust_only", 0), d.get("price", 0)
        grand += po + ro + pm
        print(f"{c:<22} {po:>8} {ro:>10} {pm:>7}")
    print("-" * 70)
    print(f"{'TOTAL divergence-instances':<22} {grand:>27}")
    print(f"per-seed totals: {per_seed_steps}")

    if args.detail:
        print("\n" + "=" * 70)
        print("FIRST OCCURRENCE PER (category, direction)")
        print("=" * 70)
        for (c, dir_) in sorted(all_first):
            ctx, py_only, rust_only = all_first[(c, dir_)]
            print(f"\n### {c} / {dir_}")
            print(f"    {ctx}")
            if py_only:
                print(f"    py_only  : {py_only}")
            if rust_only:
                print(f"    rust_only: {rust_only}")


if __name__ == "__main__":
    main()
