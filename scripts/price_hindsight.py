"""Hindsight price learning, offline (``rl18xx/agent/alphazero/price_hindsight.py``).

``detect``: replay saved games (``eval_head_to_head`` / ``eval_policy_only`` /
policy-gradient collections), find their price decisions and constraint events --
the ``unaffordable`` cash probes evaluated by ``--policy`` on an inference server --
and write ``decisions.jsonl``, ``events.jsonl``, ``games.jsonl``.

``jobs``: link each event to the price decisions within ``--window`` decisions before
it whose money touched the constrained entity, and write one branching job per
linked decision (the nearest event its open-loop target), plus ``--control-rate``
of the unlinked price decisions as a control group (and ``--money-jobs`` par /
dividend decisions, branched over the other moves of their kind).

``branch``: every job's alternatives (played price +- ``--deltas``, the range's
ends), each continued ``--rollouts`` times by the policy alone with common random
numbers, plus the open-loop replay to the event; writes ``branch.jsonl``.

``report``: event counts, how often a nearby price wins and by how much against the
continuations' noise, by price type, link direction and event kind.

``finetune``: fit only the price head of ``--init`` toward advantage-weighted cell
targets from the training games' jobs, evaluate on the held-out games' jobs (the
expected branch outcome under the old and the new head), save a checkpoint.

Usage::

    uv run python scripts/price_hindsight.py detect --collection logs/games_src/az6_u200_sys200 \\
        --policy model_checkpoints_pg/rab_place_20261014/learner/1800.pth --out logs/price_hindsight/d1
    uv run python scripts/price_hindsight.py jobs --data logs/price_hindsight/d1 --window 40
    uv run python scripts/price_hindsight.py branch --data logs/price_hindsight/d1 \\
        --policy model_checkpoints_pg/rab_place_20261014/learner/1800.pth \\
        --value model_checkpoints_search/az6_20261019/learner/200.pth --rollouts 8 --workers 12
    uv run python scripts/price_hindsight.py report --data logs/price_hindsight/d1
    uv run python scripts/price_hindsight.py finetune --data logs/price_hindsight/d1 \\
        --init model_checkpoints_pg/rab_place_20261014/learner/1800.pth --out model_checkpoints_ph/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rl18xx.agent.alphazero import price_hindsight as ph  # noqa: E402

LOGGER = logging.getLogger("price_hindsight")
MIDGAME_FILE = "human_games/midgame_positions_1830_4p.jsonl"


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------


def collection_index(directory: Path) -> list:
    """``[(game id, game file, start label)]`` of a saved-game collection: an
    ``eval_head_to_head`` one (``games/<game_idx>.json``) or a ``make_game_record``
    one (``game_file`` / ``name``)."""
    out = []
    for line in (directory / "games.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "game_idx" in r:
            path = directory / "games" / f"{r['game_idx']}.json"
            gid = f"{directory.name}/{r['game_idx']}"
        elif "game_file" in r:
            path = directory / r["game_file"]
            gid = f"{directory.name}/{path.stem}"
        elif "name" in r:
            path = directory / "games" / f"{r['name']}.json"
            gid = f"{directory.name}/{r['name']}"
        else:
            continue
        if path.exists():
            out.append((gid, str(path), r.get("start")))
    return out


def load_actions(path) -> list:
    actions = json.loads(Path(path).read_text())
    return actions.get("actions", []) if isinstance(actions, dict) else actions


_MIDGAME_CUTS: dict = {}


def agent_start(actions: list, start, midgame_file: str = MIDGAME_FILE):
    """The first action the agents played: 0 for a game started at the auction, the
    recorded prefix's length for a mid-game start, None (the first action outside
    the auction) for a first-Stock-Round start; -1 if a mid-game prefix is unknown."""
    label = str(start or "")
    if label == "auction":
        return 0
    if label.startswith("midgame:"):
        if not _MIDGAME_CUTS and Path(midgame_file).exists():
            for line in open(midgame_file):
                if line.strip():
                    rec = json.loads(line)
                    _MIDGAME_CUTS[str(rec["id"])] = {int(k): int(v) for k, v in rec["cuts"].items()}
        _, gid, sr = label.split(":")
        return _MIDGAME_CUTS.get(gid, {}).get(int(sr.lstrip("sr")), -1)
    return None


def heldout(game_id: str, fraction: float) -> bool:
    digest = hashlib.sha256(f"price_hindsight:{game_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


# ---------------------------------------------------------------------------
# Servers and workers
# ---------------------------------------------------------------------------


def start_servers(specs: dict, workers: int, batch_size: int = 512) -> dict:
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.inference_server import start_inference_server

    return {
        name: start_inference_server(
            num_workers=workers, model_factory=pg.load_server_model, checkpoint_path=spec,
            batch_size=batch_size, autocast_device=None,
        )
        for name, spec in specs.items()
    }


def run_pool(task, chunks: list, settings: dict, handles: dict, workers: int, on_result) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.inference_server import wait_checking_servers

    queues = {name: (h.request_q, h.reply_qs, h.ticket_q) for name, h in handles.items()}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=pg.worker_init, initargs=(queues,),
    ) as pool:
        pending = list(chunks)
        futures = set()
        while pending and len(futures) < 2 * workers:
            futures.add(pool.submit(task, pending.pop(0), settings))
        while futures:
            done, futures = wait_checking_servers(futures, handles, pool)
            for future in done:
                on_result(future.result())
                if pending:
                    futures.add(pool.submit(task, pending.pop(0), settings))


def _searcher(keys):
    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.lockstep import ClientEvaluator, LockstepSearcher

    return LockstepSearcher({k: ClientEvaluator(pg._CLIENTS[k]) for k in keys}, dedupe_maps=True)


def detect_task(games: list, settings: dict) -> dict:
    """Pool task: scan games, evaluate their cash probes, return decisions / events."""
    import torch

    from rl18xx.agent.alphazero.lockstep import PolicyRequest

    torch.set_num_threads(1)
    searcher = _searcher(["policy"])
    out = {"decisions": [], "events": [], "games": []}
    for game_id, path, start in games:
        t0 = time.time()
        actions = load_actions(path)
        first = agent_start(actions, start)
        if first == -1:
            out["games"].append({"game_id": game_id, "error": "unknown mid-game prefix"})
            continue
        scan = ph.scan_game(game_id, actions, first, settings["probe_cash"], settings["bid_margin"])
        if scan.error:
            out["games"].append({"game_id": game_id, "error": scan.error})
            continue
        priors = []
        reqs = [PolicyRequest(p, probe_legal, key="policy") for (_, _, p, _, probe_legal, _, _) in scan.probes]
        for k in range(0, len(reqs), 256):
            searcher.round([], reqs[k : k + 256])
        priors = [r.priors for r in reqs]
        events = scan.events + ph.unaffordable_events(scan, priors, settings["p_wanted"], settings["probe_cash"])
        for d in scan.decisions:
            d["path"] = path
        for e in events:
            e["path"] = path
        out["decisions"].extend(scan.decisions)
        out["events"].extend(events)
        out["games"].append({
            "game_id": game_id, "path": path, "start": start, "actions": len(actions),
            "first_agent_action": scan.first_agent_action, "decisions": len(scan.decision_index),
            "probes": len(scan.probes), "seconds": round(time.time() - t0, 2),
        })
    return out


def branch_task(jobs: list, settings: dict) -> dict:
    """Pool task: run branching jobs in lockstep."""
    import torch

    from rl18xx.agent.alphazero import policy_gradient as pg
    from rl18xx.agent.alphazero.lockstep import ClientEvaluator

    torch.set_num_threads(1)
    keys = ["policy"] + (["value"] if settings.get("has_value") else [])
    runner = ph.BranchRunner({k: ClientEvaluator(pg._CLIENTS[k]) for k in keys}, settings)
    loaded = []
    for job in jobs:
        loaded.append({**job, "actions": load_actions(job["path"])})
    results, decisions = runner.run(loaded)
    return {"results": results, "decisions": decisions}


# ---------------------------------------------------------------------------
# detect / jobs / branch
# ---------------------------------------------------------------------------


def cmd_detect(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    games = []
    for c in args.collection:
        index = collection_index(Path(c))
        games.extend(index[args.offset : args.offset + args.games] if args.games else index[args.offset :])
    LOGGER.info(f"{len(games)} games from {len(args.collection)} collections")
    chunks = [games[k : k + args.games_per_task] for k in range(0, len(games), args.games_per_task)]
    settings = {"probe_cash": args.probe_cash, "p_wanted": args.p_wanted, "bid_margin": args.bid_margin}
    handles = start_servers({"policy": args.policy}, args.workers)
    files = {k: (out / f"{k}.jsonl").open("w") for k in ("decisions", "events", "games")}
    counts, started = Counter(), time.time()

    def on_result(result):
        for k in ("decisions", "events", "games"):
            for row in result[k]:
                files[k].write(ph.to_json(row) + "\n")
            counts[k] += len(result[k])
        for f in files.values():
            f.flush()
        LOGGER.info(f"{counts['games']}/{len(games)} games, {counts['events']} events, {time.time() - started:.0f}s")

    try:
        run_pool(detect_task, chunks, settings, handles, args.workers, on_result)
    finally:
        for h in handles.values():
            h.shutdown()
        for f in files.values():
            f.close()
    meta = {**vars(args), "seconds": round(time.time() - started, 1), "counts": dict(counts)}
    (out / "detect_meta.json").write_text(json.dumps(meta, indent=2))


def read_jsonl(path) -> list:
    return [json.loads(line) for line in open(path) if line.strip()]


def cmd_jobs(args) -> None:
    data = Path(args.data)
    decisions = read_jsonl(data / "decisions.jsonl")
    events = read_jsonl(data / "events.jsonl")
    kinds = set(args.event_kinds.split(",")) if args.event_kinds else set(ph.EVENT_KINDS)
    events = [e for e in events if e["kind"] in kinds]
    if args.max_shortfall is not None:
        events = [e for e in events if e.get("shortfall") is None or e["shortfall"] <= args.max_shortfall]
    links = ph.link_events(decisions, events, args.window)
    rng = random.Random(args.seed)
    by_decision = defaultdict(list)
    for e, d, helps in links:
        by_decision[(d["game_id"], d["i"])].append((e, d, helps))
    jobs = []
    for key, items in by_decision.items():
        # The primary event: a lost bid on the decision's own company, else the nearest.
        items.sort(key=lambda x: (not (x[0]["kind"] == "bid_lost" and x[2] > 0), x[0]["i"]))
        e, d, helps = items[0]
        jobs.append({
            "group": "linked", "game_id": d["game_id"], "path": d["path"], "decision": d,
            "event": e, "helps": helps,
            "events": [{"kind": x[0]["kind"], "i": x[0]["i"], "helps": x[2], "shortfall": x[0].get("shortfall")} for x in items],
        })
    linked = set(by_decision)
    unlinked = [d for d in decisions if d["kind"] == "price" and (d["game_id"], d["i"]) not in linked]
    controls = [d for d in unlinked if rng.random() < args.control_rate]
    for d in controls:
        jobs.append({"group": "control", "game_id": d["game_id"], "path": d["path"], "decision": d,
                     "event": None, "helps": 0, "events": []})
    if args.money_jobs:
        for kind in ph.MONEY_TYPES:
            money = [d for d in decisions if d["kind"] == kind]
            rng.shuffle(money)
            for d in money[: args.money_jobs]:
                jobs.append({"group": f"money:{kind}", "game_id": d["game_id"], "path": d["path"], "decision": d,
                             "event": None, "helps": 0, "events": []})
    rng.shuffle(jobs)
    if args.per_stratum:
        # At most N jobs per (group, decision type, primary event kind): rare kinds keep all theirs.
        kept, seen = [], Counter()
        for job in jobs:
            key = (job["group"], job["decision"]["type"], job["event"]["kind"] if job["event"] else None)
            if seen[key] < args.per_stratum:
                kept.append(job)
                seen[key] += 1
        jobs = kept
        LOGGER.info("strata: " + ", ".join(f"{k}: {v}" for k, v in sorted(seen.items(), key=str)))
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    for k, job in enumerate(jobs):
        job["job_id"] = k
        job["seed"] = rng.randrange(2**31)
        job["heldout"] = heldout(job["game_id"], args.heldout_fraction)
    out = Path(args.out) if args.out else data / "jobs.jsonl"
    with open(out, "w") as f:
        for job in jobs:
            f.write(ph.to_json(job) + "\n")
    c = Counter(j["group"] for j in jobs)
    LOGGER.info(f"{len(links)} links over {len(by_decision)} decisions; jobs {dict(c)} -> {out}")


def cmd_branch(args) -> None:
    data = Path(args.data)
    jobs = read_jsonl(Path(args.jobs) if args.jobs else data / "jobs.jsonl")
    if args.groups:
        groups = set(args.groups.split(","))
        jobs = [j for j in jobs if j["group"] in groups or j["group"].split(":")[0] in groups]
    out = Path(args.out) if args.out else data / "branch.jsonl"
    done = set()
    if args.resume and out.exists():
        done = {r["job"] for r in read_jsonl(out)}
    jobs = [j for j in jobs if j["job_id"] not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    LOGGER.info(f"{len(jobs)} jobs to branch ({len(done)} done)")
    settings = {
        "rollouts": args.rollouts, "temperature": args.temperature, "max_decisions": args.max_decisions,
        "horizons": [int(h) for h in args.horizons.split(",")], "deltas": [int(x) for x in args.deltas.split(",")],
        "extremes": not args.no_extremes, "has_value": bool(args.value), "dedupe_maps": True,
    }
    specs = {"policy": args.policy, **({"value": args.value} if args.value else {})}
    handles = start_servers(specs, args.workers)
    chunks = [jobs[k : k + args.jobs_per_task] for k in range(0, len(jobs), args.jobs_per_task)]
    counts, started = Counter(), time.time()
    f = out.open("a")

    def on_result(result):
        for r in result["results"]:
            f.write(ph.to_json(r) + "\n")
            counts["jobs"] += 1
            counts["errors"] += int(bool(r.get("error")))
        counts["decisions"] += result["decisions"]
        f.flush()
        el = time.time() - started
        LOGGER.info(f"{counts['jobs']}/{len(jobs)} jobs ({counts['errors']} errors), "
                    f"{counts['decisions'] / el:.0f} rollout decisions/s, {el:.0f}s")

    try:
        run_pool(branch_task, chunks, settings, handles, args.workers, on_result)
    finally:
        for h in handles.values():
            h.shutdown()
        f.close()
    meta = {**vars(args), "seconds": round(time.time() - started, 1), "counts": dict(counts)}
    (data / f"branch_meta_{int(started)}.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _mean_se(xs) -> tuple:
    xs = np.asarray(list(xs), dtype=np.float64)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    se = float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
    return float(xs.mean()), se, len(xs)


def _fmt(m, se, n, scale=1.0, digits=4) -> str:
    if n == 0 or not np.isfinite(m):
        return "--"
    return f"{m * scale:+.{digits}f} +- {se * scale:.{digits}f} (n={n})"


def price_type(job: dict) -> str:
    d = job["decision"]
    if d["kind"] != "price":
        return d["kind"]
    if d["type"] == "buy_train":
        return "train (cross-corp)"
    return {"buy_company": "private (corp buys)", "bid": "auction bid"}[d["type"]]


def job_stats(job: dict, result: dict, key: str) -> dict:
    """One job's summary: the best alternative's advantage and SE, the split-half
    gain, the noise, the direction that helps."""
    adv = ph.paired_advantages(result, key)
    if not adv:
        return {}
    best = max(adv, key=lambda a: a[1])
    worst = min(adv, key=lambda a: a[1])
    sh = ph.split_half_gain(result, key)
    rollout_sd = float(np.mean([np.std(d, ddof=1) for _, _, _, d in adv]))
    played = result["alts"][0]
    o = ph.alt_outcomes(played, result["seat"], key)
    out = {
        "best_adv": best[1], "best_se": best[2], "best_delta": best[0].get("delta"),
        "worst_adv": worst[1], "any_sig_better": any(a[1] > 2.0 * a[2] and a[2] > 0 for a in adv),
        "any_sig_worse": any(a[1] < -2.0 * a[2] and a[2] > 0 for a in adv),
        "split_gain": sh[1] if sh else None, "split_choice": sh[0] if sh else None,
        "pair_sd": rollout_sd, "outcome_sd": float(np.std(o, ddof=1)) if len(o) > 1 else float("nan"),
        "advs": [(a[0].get("delta"), a[0]["price"], a[1], a[2]) for a in adv],
        "identical_share": float(np.mean([np.mean(d == 0) for _, _, _, d in adv])),
    }
    # Direction: the alternatives that move money toward the constrained entity.
    helps = job.get("helps", 0)
    if helps and job["decision"]["kind"] == "price":
        p0 = job["decision"]["price"]
        toward = [a for a in adv if (a[0]["price"] - p0) * helps > 0]
        away = [a for a in adv if (a[0]["price"] - p0) * helps < 0]
        out["toward_adv"] = float(np.mean([a[1] for a in toward])) if toward else None
        out["away_adv"] = float(np.mean([a[1] for a in away])) if away else None
    return out


def cmd_report(args) -> None:
    data = Path(args.data)
    jobs = {j["job_id"]: j for j in read_jsonl(Path(args.jobs) if args.jobs else data / "jobs.jsonl")}
    results = [r for r in read_jsonl(Path(args.branch) if args.branch else data / "branch.jsonl") if not r.get("error")]
    report = {}
    lines = []

    # Events.
    events = read_jsonl(data / "events.jsonl")
    games = read_jsonl(data / "games.jsonl")
    ok_games = [g for g in games if not g.get("error")]
    lines.append(f"games scanned: {len(games)} ({len(games) - len(ok_games)} failed to replay)")
    ev = Counter(e["kind"] for e in events)
    lines.append("events by kind (per game): " + ", ".join(f"{k} {v} ({v / max(len(ok_games), 1):.2f})" for k, v in ev.most_common()))
    sub = Counter(e.get("wanted_type") for e in events if e["kind"] == "unaffordable")
    lines.append("unaffordable, wanted move: " + ", ".join(f"{k} {v}" for k, v in sub.most_common()))
    buckets = Counter()
    for e in events:
        s = e.get("shortfall")
        b = "?" if s is None else "<=5" if s <= 5 else "<=10" if s <= 10 else "<=20" if s <= 20 else "<=50" if s <= 50 else "<=100" if s <= 100 else ">100"
        buckets[(e["kind"], b)] += 1
    for kind in ev:
        lines.append(f"  {kind} shortfall: " + ", ".join(f"{b} {buckets[(kind, b)]}" for b in ("<=5", "<=10", "<=20", "<=50", "<=100", ">100", "?") if buckets[(kind, b)]))
    decisions = read_jsonl(data / "decisions.jsonl")
    dc = Counter((d["kind"], d["type"]) for d in decisions)
    lines.append("decisions: " + ", ".join(f"{k[1]} {v}" for k, v in dc.most_common()))
    where = Counter()
    for d in decisions:
        if d["kind"] == "price":
            pos = "lo" if d["price"] == d["lo"] else "hi" if d["price"] == d["hi"] else "interior"
            where[(d["type"], pos)] += 1
    for t in sorted({k[0] for k in where}):
        n = sum(v for k, v in where.items() if k[0] == t)
        lines.append(f"  played {t}: " + ", ".join(f"{p} {where[(t, p)] / n:.3f}" for p in ("lo", "interior", "hi")) + f" (n={n})")
    report["events"] = dict(ev)
    report["decisions"] = {f"{k[0]}:{k[1]}": v for k, v in dc.items()}
    report["played_position"] = {f"{k[0]}:{k[1]}": v for k, v in where.items()}

    # Branching.
    for key in args.keys.split(","):
        rows = []
        for r in results:
            job = jobs.get(r["job"])
            if job is None:
                continue
            s = job_stats(job, r, key)
            if s:
                rows.append((job, r, s))
        lines.append("")
        lines.append(f"== outcome: {key} ({len(rows)} jobs branched)")
        groups = defaultdict(list)
        for job, r, s in rows:
            groups[(job["group"].split(":")[0], price_type(job))].append((job, r, s))
            if job["event"]:
                groups[(f"linked:{job['event']['kind']}", price_type(job))].append((job, r, s))
        hdr = (f"{'group':34s} {'type':20s} {'n':>5s} {'alts':>5s} {'identical':>9s} {'pair sd':>8s} {'best adv (raw)':>15s} "
               f"{'sig better':>10s} {'sig worse':>10s} {'split-half gain':>26s} {'toward - away':>26s}")
        lines.append(hdr)
        rep = {}
        for (g, t), items in sorted(groups.items()):
            n = len(items)
            ss = [s for _, _, s in items]
            split = [s["split_gain"] for s in ss if s.get("split_gain") is not None]
            ta = [s["toward_adv"] - s["away_adv"] for s in ss if s.get("toward_adv") is not None and s.get("away_adv") is not None]
            m, se, k = _mean_se(split)
            tm, tse, tk = _mean_se(ta)
            row = {
                "n": n, "alts": float(np.mean([len(s["advs"]) for s in ss])),
                "identical_share": float(np.mean([s["identical_share"] for s in ss])),
                "pair_sd": float(np.mean([s["pair_sd"] for s in ss])),
                "best_adv_raw": float(np.mean([s["best_adv"] for s in ss])),
                "sig_better": float(np.mean([s["any_sig_better"] for s in ss])),
                "sig_worse": float(np.mean([s["any_sig_worse"] for s in ss])),
                "split_gain": (m, se, k), "toward_minus_away": (tm, tse, tk),
            }
            rep[f"{g}|{t}"] = row
            lines.append(
                f"{g:34s} {t:20s} {n:5d} {row['alts']:5.1f} {row['identical_share']:9.3f} {row['pair_sd']:8.4f} "
                f"{row['best_adv_raw']:+15.4f} {row['sig_better']:10.3f} {row['sig_worse']:10.3f} "
                f"{_fmt(m, se, k):>26s} {_fmt(tm, tse, tk):>26s}"
            )
        # By delta: the mean advantage of each price move, over jobs.
        by_delta = defaultdict(list)
        for job, r, s in rows:
            if job["decision"]["kind"] != "price":
                continue
            for delta, price, a, se in s["advs"]:
                d = job["decision"]
                tag = "lo" if price == d["lo"] else "hi" if price == d["hi"] else f"{delta:+d}"
                by_delta[(price_type(job), tag)].append(a)
        lines.append(f"mean advantage by price move ({key}):")
        for (t, tag), xs in sorted(by_delta.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            m, se, k = _mean_se(xs)
            lines.append(f"   {t:20s} {tag:>4s} {_fmt(m, se, k)}")
        # Signal vs noise: the variance of true advantages = var(observed) - mean(se^2).
        a_obs = np.array([a for _, _, s in rows for (_, _, a, _) in s["advs"]])
        se2 = np.array([se**2 for _, _, s in rows for (_, _, _, se) in s["advs"]])
        if len(a_obs) > 2:
            true_var = float(a_obs.var() - se2.mean())
            lines.append(
                f"advantages: observed sd {a_obs.std():.4f}, mean continuation-noise se {np.sqrt(se2.mean()):.4f}, "
                f"implied true-effect sd {math.sqrt(max(true_var, 0)):.4f} (var {true_var:+.2e})"
            )
            rep["signal"] = {"observed_sd": float(a_obs.std()), "noise_se": float(np.sqrt(se2.mean())), "true_var": true_var}
        # Split-half reliability: even-half vs odd-half advantage across (job, alternative).
        ev_, od_ = [], []
        for job, r, s in rows:
            for alt, _, _, diff in ph.paired_advantages(r, key):
                if len(diff) >= 4:
                    ev_.append(diff[0::2].mean())
                    od_.append(diff[1::2].mean())
        if len(ev_) > 10:
            corr = float(np.corrcoef(ev_, od_)[0, 1])
            lines.append(f"split-half reliability of per-alternative advantages: r = {corr:+.3f} over {len(ev_)} pairs")
            rep["split_half_r"] = corr
        # Horizon values vs final outcome.
        hz = defaultdict(lambda: ([], []))
        for job, r, s in rows:
            final = {k: a for k, (alt, a, _, _) in enumerate(ph.paired_advantages(r, key), start=1)}
            for h in (0, 10, 30, 60, 100):
                ha = ph.horizon_advantages(r, h)
                for k, v in ha.items():
                    if k in final:
                        hz[h][0].append(v)
                        hz[h][1].append(final[k])
        for h, (xs, ys) in sorted(hz.items()):
            if len(xs) > 10:
                c = float(np.corrcoef(xs, ys)[0, 1])
                lines.append(f"value-at-horizon {h:3d} advantage vs final advantage: r = {c:+.3f} (n={len(xs)}), "
                             f"mean horizon adv {np.mean(xs):+.4f}, sd {np.std(xs):.4f}")
                rep[f"horizon_{h}_r"] = c
        # Open loop.
        ol = Counter()
        for job, r, s in rows:
            if not job.get("event"):
                continue
            alts = r["alts"]
            base = alts[0].get("open_loop", {})
            if not base:
                continue
            ol["events"] += 1
            if job["event"]["kind"] == "unaffordable":
                ol["unaffordable"] += 1
                if any(a.get("open_loop", {}).get("wanted_legal") for a in alts[1:]):
                    ol["wanted_made_legal"] += 1
            if any(a.get("open_loop", {}).get("cash", -1) > base.get("cash", 0) for a in alts[1:]):
                ol["more_cash_at_event"] += 1
            if np.mean([a.get("open_loop", {}).get("skipped", 0) for a in alts[1:]] or [0]) > 0:
                ol["replay_skips"] += 1
        lines.append(f"open loop: {dict(ol)}")
        report[key] = rep

    text = "\n".join(lines)
    print(text)
    (data / "report.txt").write_text(text)
    (data / "report.json").write_text(json.dumps(report, indent=2, default=float))


# ---------------------------------------------------------------------------
# finetune
# ---------------------------------------------------------------------------


def _trunk_features(model, rows: list, device, batch_size: int = 128):
    """The price head's input (the trunk) at each row's decision state, frozen."""
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _lay_tile_range  # noqa: F401

    feats = []
    store = {}

    def hook(_module, inputs, _output):
        store["x"] = inputs[0].detach()

    handle = model.price_head.register_forward_hook(hook)
    try:
        for k in range(0, len(rows), batch_size):
            chunk = rows[k : k + batch_size]
            gs = torch.from_numpy(np.stack([r["state"][0] for r in chunk])).float().to(device)
            nf = torch.from_numpy(np.stack([r["state"][1] for r in chunk])).float().to(device)
            with torch.no_grad():
                model(gs, nf)
            feats.append(store["x"].float().cpu())
    finally:
        handle.remove()
    return torch.cat(feats) if feats else torch.zeros(0)


def finetune_rows(jobs: dict, results: list, key: str, beta: float, z: float) -> list:
    """Price-head rows: the decision's state, slot, live cells, the old head's cell
    probabilities (from the branch run), the cell advantages and the target."""
    from rl18xx.agent.alphazero import price_pmf

    rows = []
    for r in results:
        job = jobs.get(r["job"])
        if job is None or r.get("error") or job["decision"]["kind"] != "price" or "cell_probs" not in r:
            continue
        cell_adv = ph.cell_advantages(r, key)
        if len(cell_adv) < 2:
            continue
        d = job["decision"]
        probs = np.asarray(r["cell_probs"], dtype=np.float64)
        target = ph.price_cell_target_from_advantages(probs, cell_adv, beta, z)
        cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[d["slot"]], d["lo"], d["hi"])
        rows.append({
            "job": job, "result": r, "slot": d["slot"], "mask": cells.nonempty, "probs": probs,
            "target": target, "cell_adv": cell_adv,
            "cell_q": {c: a for c, (a, _, _) in cell_adv.items()},
        })
    return rows


def _encode_rows(rows: list) -> None:
    from rl18xx.agent.alphazero.mcts import _rust_encode

    cache = {}
    for row in rows:
        d = row["job"]["decision"]
        path = row["job"]["path"]
        if path not in cache:
            cache = {path: load_actions(path)}
        g = ph.replay(cache[path], d["i"])
        enc = _rust_encode(g)
        row["state"] = (np.asarray(enc[0]).reshape(-1), np.asarray(enc[1]))


def cmd_finetune(args) -> None:
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _load, _save, use_tf32

    use_tf32()
    data = Path(args.data)
    jobs = {j["job_id"]: j for j in read_jsonl(Path(args.jobs) if args.jobs else data / "jobs.jsonl")}
    results = read_jsonl(Path(args.branch) if args.branch else data / "branch.jsonl")
    rows = finetune_rows(jobs, results, args.key, args.beta, args.z)
    train = [r for r in rows if not r["job"]["heldout"]]
    test = [r for r in rows if r["job"]["heldout"]]
    LOGGER.info(f"{len(rows)} price rows: {len(train)} train, {len(test)} held out")
    _encode_rows(rows)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = _load(args.init).to(device)
    model.eval()
    feats = _trunk_features(model, rows, device).to(device)
    for k, row in enumerate(rows):
        row["feat"] = k
    head = model.price_head
    old_head = __import__("copy").deepcopy(head).eval()
    for p in old_head.parameters():
        p.requires_grad_(False)
    # Check the cached features reproduce the branch run's cell probabilities.
    with torch.no_grad():
        logits = head(feats)
    gap = []
    for row in rows:
        lg = logits[row["feat"], row["slot"]].float().cpu().numpy()
        m = row["mask"]
        q = np.where(m, np.exp(lg - lg[m].max()), 0.0)
        q /= q.sum()
        gap.append(float(np.abs(q - row["probs"]).max()))
    LOGGER.info(f"cell probabilities vs the branch run's: max |diff| {max(gap) if gap else 0:.2e}")

    def cell_log_probs(h, idx, slots, masks):
        lg = h(feats[idx])[torch.arange(len(idx), device=device), slots]
        lg = lg.float().masked_fill(~masks, float("-inf"))
        return torch.log_softmax(lg, dim=-1).masked_fill(~masks, 0.0)

    def tensors(rs):
        idx = torch.tensor([r["feat"] for r in rs], device=device)
        slots = torch.tensor([r["slot"] for r in rs], device=device)
        masks = torch.from_numpy(np.stack([r["mask"] for r in rs])).to(device)
        target = torch.from_numpy(np.stack([r["target"] for r in rs])).float().to(device)
        return idx, slots, masks, target

    def evaluate(rs, h) -> dict:
        if not rs:
            return {}
        idx, slots, masks, target = tensors(rs)
        with torch.no_grad():
            lp = cell_log_probs(h, idx, slots, masks)
            old = cell_log_probs(old_head, idx, slots, masks)
        p_new, p_old = lp.exp().cpu().numpy(), old.exp().cpu().numpy()
        gains, kls = [], []
        for k, r in enumerate(rs):
            e_new = ph.expected_under(p_new[k], r["cell_q"])
            e_old = ph.expected_under(p_old[k], r["cell_q"])
            if e_new is not None and e_old is not None:
                gains.append(e_new - e_old)
            m = r["mask"]
            kls.append(float((p_old[k][m] * (np.log(np.clip(p_old[k][m], 1e-12, None)) - np.log(np.clip(p_new[k][m], 1e-12, None)))).sum()))
        g, se, n = _mean_se(gains)
        ce = float(-(target * lp).sum(dim=1).mean())
        return {"gain": g, "gain_se": se, "n": n, "kl_old_new": float(np.mean(kls)), "target_ce": ce}

    LOGGER.info(f"before: train {evaluate(train, head)}, held out {evaluate(test, head)}")
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    history = []
    for step in range(args.steps):
        batch = [train[k] for k in rng.choice(len(train), size=min(args.batch, len(train)), replace=False)]
        idx, slots, masks, target = tensors(batch)
        lp = cell_log_probs(head, idx, slots, masks)
        with torch.no_grad():
            old = cell_log_probs(old_head, idx, slots, masks)
        ce = -(target * lp).sum(dim=1).mean()
        kl = (lp.exp() * (lp - old)).sum(dim=1).mean()
        loss = ce + args.kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "train": evaluate(train, head), "heldout": evaluate(test, head)}
            history.append(row)
            LOGGER.info(json.dumps(row))
    out = Path(args.out)
    ckpt = out / "learner" / "1.pth"
    _save(model, ckpt)
    src_config = Path(args.init).parent.parent / "config.json"
    if src_config.exists():
        (out / "config.json").write_text(src_config.read_text())  # learner_temperature for run_temperature
    (out / "finetune.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2, default=float))
    LOGGER.info(f"saved {ckpt}")


# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect")
    p.add_argument("--collection", action="append", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=0, help="Per collection (0: all)")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--games-per-task", type=int, default=4)
    p.add_argument("--probe-cash", type=int, default=100)
    p.add_argument("--p-wanted", type=float, default=0.25)
    p.add_argument("--bid-margin", type=int, default=10)

    p = sub.add_parser("jobs")
    p.add_argument("--data", required=True)
    p.add_argument("--out")
    p.add_argument("--window", type=int, default=40, help="Decisions between a price decision and its event")
    p.add_argument("--event-kinds", default="")
    p.add_argument("--max-shortfall", type=int, default=None)
    p.add_argument("--control-rate", type=float, default=0.1)
    p.add_argument("--money-jobs", type=int, default=0, help="Par and dividend decisions to branch (each kind)")
    p.add_argument("--per-stratum", type=int, default=0, help="Jobs per (group, decision type, event kind); 0: all")
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--heldout-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("branch")
    p.add_argument("--data", required=True)
    p.add_argument("--jobs")
    p.add_argument("--out")
    p.add_argument("--groups", default="")
    p.add_argument("--policy", required=True)
    p.add_argument("--value", default="")
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-decisions", type=int, default=700)
    p.add_argument("--horizons", default="0,10,30,60,100")
    p.add_argument("--deltas", default="-20,-10,-5,-1,1,5,10,20")
    p.add_argument("--no-extremes", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--jobs-per-task", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("report")
    p.add_argument("--data", required=True)
    p.add_argument("--jobs")
    p.add_argument("--branch")
    p.add_argument("--keys", default="win,place,worth")

    p = sub.add_parser("finetune")
    p.add_argument("--data", required=True)
    p.add_argument("--jobs")
    p.add_argument("--branch")
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--key", default="win")
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-weight", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    {"detect": cmd_detect, "jobs": cmd_jobs, "branch": cmd_branch, "report": cmd_report, "finetune": cmd_finetune}[args.cmd](args)


if __name__ == "__main__":
    main()
