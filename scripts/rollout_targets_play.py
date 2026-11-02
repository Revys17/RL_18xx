"""Play test for scripts/rollout_targets.py arms, with /home/revys/longrun/ops/azf_eval.sh's settings: the 11g
deployment's eval_head_to_head.py, 200 readouts, 2v2, both seatings listed first, --games each, holding the GPU eval
lock (/home/revys/longrun/ops/gpu_eval.lock) for the whole test so play tests run one at a time.

    rollout_targets_play.py --out DIR --name NAME --a SYSTEM_A --b SYSTEM_B [--games 240]
"""

import argparse
import fcntl
import logging
import os
import resource
import subprocess
from pathlib import Path

LOGGER = logging.getLogger("rollout_targets_play")
DEPLOY = Path("/home/revys/longrun/deploy/longrun-2026-11g")
WORKDIR = "/home/revys/longrun/work/lr1"
SEARCH = "@200/0.05:mean,dedupe,par=8,ptemp=1,budget=total"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--a", required=True, help="system A: a checkpoint or <policy>+<value>")
    p.add_argument("--b", required=True)
    p.add_argument("--games", type=int, default=240, help="games per seating")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--lock", default="/home/revys/longrun/ops/gpu_eval.lock")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    env = dict(os.environ)
    env.update(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="2")
    env.pop("PYTHONPATH", None)  # the deployment's own code
    with open(args.lock, "a") as lock:
        LOGGER.info("waiting for %s", args.lock)
        fcntl.flock(lock, fcntl.LOCK_EX)
        LOGGER.info("lock held")
        for side, first, second in (("Afirst", args.a, args.b), ("Bfirst", args.b, args.a)):
            name = args.name + "_" + side
            cmd = [str(DEPLOY / ".venv/bin/python"), str(DEPLOY / "scripts/eval_head_to_head.py"),
                   "--match", first + SEARCH, second + SEARCH, "--games", str(args.games), "--a-seats", "2",
                   "--seed", "0", "--start-positions", "human_games/start_positions_1830_4p_v2.jsonl",
                   "--random-start-fraction", "0.2", "--softpick", "0", "--max-length", "1000", "--stall", "400",
                   "--workers", str(args.workers), "--games-per-task", "8", "--dedupe-maps", "--server-replicas", "2",
                   "--out", str(out / name)]
            with open(out / (name + ".out"), "w") as log:
                subprocess.run(cmd, cwd=WORKDIR, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            LOGGER.info("done %s", name)


if __name__ == "__main__":
    main()
