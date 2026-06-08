"""Behavioral decode-parity: native Rust decode vs Python ActionMapper decode.

At every state of a self-advanced game, for EVERY legal policy index, apply the
action two ways to independent clones and assert the resulting engine state is
identical:

  OLD path: Python ``ActionMapper.map_index_to_action[_with_price]`` ->
            ``Action.to_dict()`` -> ``RustGame.process_action(dict)``.
  NEW path: native ``RustGame.apply_action_index(idx, price)``.

State equality is checked via the deterministic full-state encoding
(``encode_for_gnn``) — same encoding ⟺ same cash/shares/trains/tokens/tiles/phase.

The trajectory is advanced through the OLD (known-good) path so divergences are
attributable to the NEW decode. Action choice is deterministic (seeded by step),
no RNG.
"""

import logging
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine_rs import BaseGame as RustGame
from rl18xx.rust_adapter import RustGameAdapter
from rl18xx.agent.alphazero.action_mapper import ActionMapper


def _enc(game):
    gs, nf, _sz, _h, _n = game.encode_for_gnn()
    return (tuple(gs), tuple(nf))


def _decode_old(am, state_game, idx, price):
    adapter = RustGameAdapter(state_game)
    if price is not None:
        act = am.map_index_to_action_with_price(idx, adapter, int(price))
    else:
        act = am.map_index_to_action(idx, adapter)
    return act.to_dict()


def run_game(seed_players, max_steps, am):
    game = RustGame(seed_players)
    stats = {"states": 0, "checks": 0, "skipped_old_raise": 0, "mismatch": 0,
             "new_raise": 0, "by_type": {}}
    mismatches = []

    for step in range(max_steps):
        if game.finished:
            break
        adapter = RustGameAdapter(game)
        try:
            indices, price_ranges, types_by_idx = am.get_legal_actions_factored(adapter)
        except Exception as exc:  # enumeration failure is not a decode bug
            mismatches.append(f"step{step}: enumerate raised {type(exc).__name__}: {exc}")
            break
        if not indices:
            break
        stats["states"] += 1

        for idx in indices:
            pr = price_ranges.get(idx)
            atype = types_by_idx.get(idx, "?")
            # Exercise both ends of a price range (continuous-price slots), plus
            # the categorical (no-price) case. Each price is checked behaviorally.
            if pr and pr[0] != pr[1]:
                prices = [pr[0], pr[1], (pr[0] + pr[1]) // 2]
            elif pr:
                prices = [pr[0]]
            else:
                prices = [None]

            for price in prices:
                old_clone = game.pickle_clone()
                new_clone = game.pickle_clone()

                # OLD path (oracle). If Python decode itself raises (e.g.
                # enumerated but no concrete share), that is not a NEW bug — skip.
                try:
                    old_dict = _decode_old(am, old_clone, idx, price)
                    old_clone.process_action(old_dict)
                except Exception:
                    stats["skipped_old_raise"] += 1
                    continue

                # NEW path.
                try:
                    new_clone.apply_action_index(idx, int(price) if price is not None else None)
                except Exception as exc:
                    stats["new_raise"] += 1
                    if len(mismatches) < 25:
                        mismatches.append(
                            f"step{step} idx{idx} type={atype} price={price}: NEW raised "
                            f"{type(exc).__name__}: {exc}; old_dict={old_dict}"
                        )
                    continue

                stats["checks"] += 1
                stats["by_type"][atype] = stats["by_type"].get(atype, 0) + 1
                if _enc(old_clone) != _enc(new_clone):
                    stats["mismatch"] += 1
                    if len(mismatches) < 25:
                        try:
                            new_map = game.decode_index_to_map(idx, int(price) if price is not None else None)
                        except Exception as exc:
                            new_map = f"<decode raised {exc}>"
                        mismatches.append(
                            f"step{step} idx{idx} type={atype} price={price}: STATE mismatch\n"
                            f"      old={ {k: old_dict.get(k) for k in ('type','train','price','exchange','shares')} }\n"
                            f"      new={new_map}"
                        )

        # Advance the real game via the OLD path (known-good), deterministic pick.
        choice = indices[step % len(indices)]
        pr = price_ranges.get(choice)
        price = pr[0] if pr else None
        try:
            d = _decode_old(am, game, choice, price)
            game.process_action(d)
        except Exception:
            # Picked index Python can't materialize here; try the first index.
            advanced = False
            for choice in indices:
                pr = price_ranges.get(choice)
                price = pr[0] if pr else None
                try:
                    d = _decode_old(am, game, choice, price)
                    game.process_action(d)
                    advanced = True
                    break
                except Exception:
                    continue
            if not advanced:
                break

    return stats, mismatches


def main():
    am = ActionMapper()
    configs = [
        {1: "P1", 2: "P2", 3: "P3", 4: "P4"},
        {1: "A", 2: "B", 3: "C"},
        {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"},
        {1: "A", 2: "B"},
        {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"},
    ]
    total = {"states": 0, "checks": 0, "skipped_old_raise": 0, "mismatch": 0, "new_raise": 0}
    by_type = {}
    all_mismatches = []
    for gi, players in enumerate(configs):
        stats, mismatches = run_game(players, max_steps=1500, am=am)
        for k in total:
            total[k] += stats[k]
        for t, c in stats["by_type"].items():
            by_type[t] = by_type.get(t, 0) + c
        if mismatches:
            all_mismatches.append((gi, len(players), mismatches))
        print(f"game[{gi}] {len(players)}p: {stats}", flush=True)

    print("\n==== TOTAL ====")
    print(total)
    print("checks by action type:", by_type)
    if all_mismatches:
        print(f"\n!! {sum(len(m) for _,_,m in all_mismatches)} divergence samples:")
        for gi, np_, ms in all_mismatches:
            for m in ms[:10]:
                print(f"  game[{gi}]({np_}p) {m}")
        sys.exit(1)
    print("\nALL CLEAR — native decode behaviorally matches Python decode.")


if __name__ == "__main__":
    main()
