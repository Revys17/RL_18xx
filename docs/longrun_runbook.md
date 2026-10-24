# Long-run runbook (months-long training on the workstation)

How to run one training job for months on this machine (Threadripper PRO 9975WX, 64 threads, 251 GB RAM,
one RTX PRO 6000 Blackwell 96 GB, shared with another project and with `claude-rc.service`) so that the
failures of October 2026 can't recur. The harness is `scripts/longrun/`; nothing in it touches the training
code. The 2026-11 run (docs/long_run_recipe.md: a generator, a value trainer and an eval ladder, plus daily
jobs) is set up as in [The long run (2026-11)](#the-long-run-2026-11); the sections before it describe one
instance, which is how each of its three long-lived processes runs.

| Failure (2026-10) | What prevents it now |
|---|---|
| az1 died when a merge put conflict markers into the worktree it imported from (10-18) | the run imports from a **frozen deployment**: its own read-only clone at a tag, with its own interpreter and venv; the launcher refuses to start a deployment whose files changed |
| an OOM from two parallel eval queues killed the user's systemd and Remote Control | the run is a **systemd user service** with `MemoryHigh` / `MemoryMax` / no swap: it is throttled, then OOM-killed inside its own cgroup and restarted; the global OOM killer never gets that far |
| killing the main process orphaned the inference servers and pool workers (and their GPU memory) | `KillMode=control-group`: stopping the unit, or its main process dying, kills every process in its cgroup before a restart |
| a run crashed and stayed down | `Restart=on-failure` from the last snapshot (`--resume`), with a start limit for crash loops |
| a resume silently reset settings (fixed on `claude/search-path-fixes`, 5634a72) | the launcher passes the same `TRAIN_ARGS` on every start, so either version resumes with the same settings |
| evals hung when an inference server died | the training already fails on a dead server (`wait_checking_servers`); the watchdog alerts on any stall |
| a driver/library mismatch after an unattended NVIDIA upgrade needed a mid-run reboot | hold the driver and kernel packages for the run (`host_hold.sh`); the watchdog alerts on a mismatch before it bites |

## The pieces

```
<deployment>/                 frozen code: a git clone detached at a tag, read-only
  .python/  .venv/              its own copy of the interpreter; uv sync --locked (engine-rs compiled)
  DEPLOYMENT.json               commit, tag, kernel, NVIDIA module + packages, torch / CUDA, uv, rustc, hashes
  deployment-packages.txt       every installed package==version     deployment-sha256.txt: engine_rs, locks
  CLAUDE.local.md               "frozen deployment, never edit" for agents that open it
<workdir>/                    the run's working directory (data; relative paths resolve here)
  human_games -> ...            start / mid-game positions (symlink)
  model_checkpoints_pg -> ...   the starting policy and critic (symlink, read-only use)
  model_checkpoints_search/<run>/   the run: learner/, critic/, history.jsonl, state.json, optimizer.pth
  runs/                         TensorBoard (SL/*)
  logs/longrun_<instance>.log   main.py's output and the launcher's decisions
  longrun/<instance>/           starts.jsonl, exits.jsonl (launcher), alerts.log, watchdog_state.json
~/.config/rl18xx-longrun/<instance>.env                      the instance's settings (install.sh)
~/.config/systemd/user/rl18xx-longrun@.service               the run (template; instance = <instance>)
~/.config/systemd/user/rl18xx-longrun-watchdog@.service      its watchdog
~/.config/systemd/user/rl18xx-longrun@<instance>.service.d/10-limits.conf   memory, CPUs, restart delay
```

- `deploy.sh create|verify|in-use|update` -- the frozen deployment.
- `launch.sh start|stop` -- the unit's `ExecStart` / `ExecStop` (runs from the deployment). `KIND` says what it
  runs: `generator` (search-learning, the default: fresh, then `--resume`), `value` (`main.py value-train`) or
  `ladder` (`scripts/eval_ladder.py run`); the last two get the same arguments on every start and resume themselves.
- `install.sh` -- writes the instance's `.env`, the unit templates and the limits drop-in; starts nothing
  (`--kind generator|value|ladder|daily`).
- `watchdog.py` -- the watchdog service, and `status` for a read-only look at a run; with `HOST_CHECKS=1` (one
  instance) also the host samples the daily report reads and the Xid scan.
- `snapshot_check.py` -- the launcher's check that a snapshot is resumable (a `--value-follow` run has no critic).
- `host_hold.sh` -- apt holds for the NVIDIA driver and the kernel (`check`: are they in place, for M7).
- `dryrun.sh`, `make_tiny_start.py` -- the end-to-end check on a tiny CPU run.
- Daily jobs (`daily.sh` under `rl18xx-longrun-daily@.service` + `.timer`): `retention.py`, `style_stats.py`,
  `integrity_check.py`, `status_report.py`; `maintenance.py` holds the M1-M9 metrics (stdlib, shared with the
  watchdog).

## Before the run

1. **Host.** Bring the machine up to date and reboot *before* the run, then hold (see [Host stability](#host-stability)):
   `bash scripts/longrun/host_hold.sh status` shows a pending reboot, unheld packages and pending upgrades.
2. **Code.** Merge everything the run needs into master (at least `claude/search-path-fixes` for 5634a72,
   and this harness), tag it, and deploy it outside `.claude/worktrees` (the desktop app cleans up worktrees
   there; a deployment is ~5.3 GB):

   ```bash
   git tag longrun-2026-11 <commit>
   bash scripts/longrun/deploy.sh create longrun-2026-11 ~/longrun/deploy/longrun-2026-11 --cpus 24-31,56-63
   bash scripts/longrun/deploy.sh verify ~/longrun/deploy/longrun-2026-11
   ```
   About 75 s on 16 pinned cores. `--cpus` pins the build (taskset, nice 19) while other jobs run. The
   interpreter is copied from uv's managed 3.11 (so `uv python upgrade` can't move it), packages are copied out
   of uv's cache (not hardlinked), and the Rust engine is built with `--locked`.
3. **Workdir.** One directory per run, with symlinks to the shared inputs:

   ```bash
   W=~/longrun/work/lr1; mkdir -p $W/logs $W/model_checkpoints_search
   ln -s /home/revys/workspace/RL_18xx/human_games $W/human_games
   ln -s <checkpoints dir>/model_checkpoints_pg $W/model_checkpoints_pg      # the starting policy + critic
   ```
   The run writes only under `model_checkpoints_search/`, `runs/`, `logs/` and `longrun/` of the workdir.
4. **Memory budget.** azl at 40 workers x 32 games held ~80 GB RSS (~70 GB PSS) on 2026-10-22. Set
   `MemoryMax` to the measured peak plus margin, and keep `MemoryMax` + everything else on the machine
   (the other project, evals, the desktop) under ~230 GB. `MemoryHigh` (~10% below `MemoryMax`) throttles the
   run first and the watchdog alerts at it.
5. **Install** the instance (writes config only; nothing starts):

   ```bash
   D=~/longrun/deploy/longrun-2026-11
   bash $D/scripts/longrun/install.sh lr1 --deploy $D --workdir $W \
     --run-dir model_checkpoints_search/lr1_20261101 \
     --start-args "--policy model_checkpoints_pg/<run>/learner/<n>.pth --value model_checkpoints_pg/<run>/critic/<n>.pth" \
     --train-args "--max-updates 100000 --microbatch 1024 --workers 40 --games-per-worker 32 \
       --midgame-positions human_games/midgame_positions_1830_4p_v2.jsonl --midgame-fraction 0.3 \
       --auction-fraction 0.15 --dedupe-maps --server-replicas 2 --full-move-temperature 1 --explore-decisions 120 ..." \
     --memory-high 150G --memory-max 170G --restart-sec 120 \
     --set STALL_MINUTES=90 --set DISK_MIN_GB=100
   ```
   - `--start-args` go to the first start only (the checkpoints); `--train-args` to **every** start, and must
     name every setting the run should keep: a resume on code before 5634a72 resets an omitted resumable
     setting (`--midgame-fraction`, `--explore-decisions`, `--workers`, ...) to its default. After it, a
     resume changes only the settings passed, and warns about passed settings it ignores -- so the same
     `TRAIN_ARGS` is right for both.
   - `--cpus "0-63"` (CPUAffinity), `--nice N`: optional. `--restart-sec`: the pause before a restart (give
     the GPU memory time to go). The start limit is 5 failures in 2 h (in the unit template).
   - `--set KEY=VALUE`: anything else in the `.env` -- watchdog thresholds (below), `ALERT_COMMAND`,
     `CUDA_VISIBLE_DEVICES`, `LOG`, `STATE_DIR`, `COMMAND`.
   - `STALL_MINUTES`: above the time a resumed run needs to refill its replay window (`--min-window-rows`;
     the window is not in the snapshot) plus one update, or every restart ends in a stall alert.
6. **Dry run** the deployment once (`dryrun.sh`, below): ~5 minutes, CPU only.

## Operating it

```bash
systemctl --user start rl18xx-longrun@lr1           # first start (fresh); also starts the watchdog
python3 $D/scripts/longrun/watchdog.py --instance lr1 status   # unit, last update, snapshot, active alerts
journalctl --user -u rl18xx-longrun@lr1 -f          # the launcher's decisions; main.py writes to the log:
tail -f $W/logs/longrun_lr1.log
cat $W/longrun/lr1/alerts.log                        # every alert the watchdog raised
systemctl --user stop rl18xx-longrun@lr1             # stop (waits out a snapshot write, then kills all of it)
systemctl --user restart rl18xx-longrun@lr1          # = stop + start: resumes from the last snapshot
touch $W/model_checkpoints_search/lr1_20261101/STOP  # finish at the next update, exit 0, no restart
systemctl --user reset-failed rl18xx-longrun@lr1     # after a start-limit or a 75/76/77 stop, before starting again
systemctl --user enable rl18xx-longrun@lr1           # optional: start at boot (the user manager lingers)
```

Every start goes through `launch.sh start`:

1. `deploy.sh verify`: HEAD is the deployed commit, no tracked file differs, `.venv/bin/python` is the
   deployment's own interpreter, the installed packages and the engine's `.so` match the record. Otherwise:
   exit **76**, not restarted.
2. A shared lock on `<deployment>/.longrun.lock` for the run's lifetime (`deploy.sh update` refuses while it
   is held, and also while any process maps, runs from or names the deployment).
3. A stale `STOP` file is removed (it would end the run at once).
4. **Resume or fresh.** With `state.json`: `snapshot_check.py` (state.json parses; the snapshot's learner and
   critic, the KL anchor and `optimizer.pth` load), then `--resume <run dir> $TRAIN_ARGS`; an unreadable
   snapshot is exit **77**, not restarted. Without `state.json`: a run directory that never reached its first
   snapshot is moved aside (`<run>.nosnapshot-<time>`) and the run starts fresh with
   `--out-dir/--run-name $START_ARGS $TRAIN_ARGS`; one with `snapshot_every` or more updates of history but no
   `state.json` is exit 77 (someone deleted it: look first).
5. `main.py` runs with `-s` and no `PYTHONPATH` (only the deployment's code), `OMP_NUM_THREADS=2`,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, its output appended to the log.
6. On exit: 0 = finished (max updates or STOP). A `RuntimeError` naming a non-finite loss / gradient is exit
   **75** and is **not** restarted: the run needs a decision (lower the LR, resume anyway, stop). Anything
   else (a crash, a kill, a dead server, an OOM kill) is restarted after `RestartSec`.

What is lost on a restart: the updates since the last snapshot (`--snapshot-every`, default 10) and the
replay window. `history.jsonl` then repeats those update numbers (the resumed run appends them again), and so
does TensorBoard: take the last line for an update.

### Changing the run mid-way

- **Settings a resume may change** (`max_updates`, `workers`, `games_per_worker`, `microbatch`, the
  exploration settings, mid-game / auction fractions, the inference speed-ups, the learning rates after
  e96d852): edit `TRAIN_ARGS` in `~/.config/rl18xx-longrun/<instance>.env` (or re-run `install.sh`), then
  `systemctl --user restart`. Anything else is fixed by the run's `state.json`.
- **New code**: never edit the deployment. Tag the new commit, `deploy.sh create` a second deployment,
  dry-run it, check that its code still reads the run's config
  (`cd $W && <new>/.venv/bin/python -s -c 'import json, sys; sys.path.insert(0, "<new>"); from
  rl18xx.agent.alphazero.search_learning import SearchLearningConfig as C; C(**json.load(open("<run dir>/state.json"))["config"])'`
  -- a removed or renamed config field fails here instead of in a restart loop), point the instance's
  `DEPLOY` / `LAUNCHER` / `WATCHDOG` at it (re-run `install.sh` with `--deploy <new>`), restart both units.
  `deploy.sh update` rebuilds a deployment in place instead, only while nothing uses it -- the watchdog
  counts, since it runs from the deployment.
- **Evals** run beside the training, not inside its unit. Give each queue its own memory cap so two queues
  can't take the machine down again, e.g.
  `systemd-run --user --unit eval-azl-u3000 -p MemoryMax=40G -p MemorySwapMax=0 --working-directory=$W
  <deployment>/.venv/bin/python -s <deployment>/scripts/eval_head_to_head.py ...` (a transient unit: gone when it
  ends, stoppable with `systemctl --user stop eval-azl-u3000`, never orphaned).

### Recovering a snapshot

`optimizer.pth` and `state.json` are written in place (the checkpoints are atomic), so a kill in the middle of
a snapshot -- an OOM kill, a power cut, or the user manager's 5-second stop timeout at shutdown
(`/usr/lib/systemd/system/user@.service.d/timeout.conf`; `systemctl stop` itself waits the write out) --
can leave one unreadable. The launcher then stops with 77 (not restarted, a `refused` alert) and the log
names the file and the repair:

```bash
<deployment>/.venv/bin/python -s <deployment>/scripts/longrun/snapshot_check.py <workdir>/<run dir> --repair
systemctl --user reset-failed rl18xx-longrun@<i> && systemctl --user start rl18xx-longrun@<i>
```

`--repair` keeps every file it replaces (`<name>.bad-<time>`). An unreadable `optimizer.pth` is moved aside
(the resume starts AdamW afresh; its moments rebuild within a few updates). An unreadable `state.json` is
rebuilt at the newest snapshot that has its learner / critic / anchor checkpoints and a `history.jsonl` line,
from `config.json` (the run's merged config, rewritten at every start) and that line's `kl_coef` and
`anchor_update`. A kill during the write of snapshot `S`'s `state.json` happens before `S`'s history line, so
the rebuild lands on `S - snapshot_every`, with `optimizer.pth` from `S` (kept: closer than starting afresh).

## The watchdog

`rl18xx-longrun-watchdog@<instance>.service` (pulled in by the run's `Wants=`; keeps running after the run
stops, so it can report why; stdlib only, system `python3`). Every `WATCH_INTERVAL_SEC` it checks:

| Alert | Level | When | Setting (default) |
|---|---|---|---|
| `crash` | WARN | main.py exited non-zero (status, last error line); systemd restarts it | |
| `restart` | WARN | systemd restarted the unit | |
| `crash_loop` | CRIT | that many automatic restarts (systemd's `NRestarts`) within the window | `CRASH_LOOP_RESTARTS` (3), `CRASH_LOOP_WINDOW_MIN` (60) |
| `failed` | CRIT | the unit is down: exit 75 / 76 / 77 with its meaning, or systemd's start limit stopped the restarts (the unit's `Result` then keeps the last failure's kind, e.g. `oom-kill`; `start-limit-hit` only appears for a refused manual start) | |
| `nonfinite` | CRIT | the training stopped on a non-finite loss (not restarted) | |
| `refused` | CRIT | the launcher refused to start (76 deployment modified, 77 snapshot) | |
| `stall` | WARN | no progress for that long while the unit is up (counted from the later of the last progress and the unit's start): a new `history.jsonl` line (generator), `status.json` rewritten (value trainer: every poll and evaluation), `heartbeat` touched (ladder: every poll and every finished game) | `STALL_MINUTES` (60), `PROGRESS_FILE` |
| `disk:<path>` | WARN, CRIT < 1/4 | free space on the run's / workdir's filesystem | `DISK_MIN_GB` (100) |
| `mem_available` | WARN | the machine's MemAvailable | `MEM_AVAILABLE_MIN_GB` (16) |
| `memory_high` / `oom_kill` | WARN / CRIT | the run's cgroup above 95% of its MemoryHigh / an OOM kill in it | |
| `gpu<i>_temp` / `gpu<i>_tlimit` | WARN | GPU temperature, or headroom to its slowdown temperature (`temperature.gpu.tlimit`; 26-27 C under azl's load) | `GPU_TEMP_MAX_C` (87), `GPU_TLIMIT_MIN_C` (5) |
| `gpu<i>_throttle` | WARN | thermal / hardware slowdown or power brake two checks running (the power cap, normal under load, is not alerted) | `GPU_CHECK` (1) |
| `gpu` | CRIT | `nvidia-smi` fails (e.g. "Driver/library version mismatch") | |
| `nvidia_mismatch` | CRIT | installed libnvidia-compute differs from the loaded module: the next CUDA start fails until a reboot | |
| `xid` | CRIT | an NVIDIA Xid error in the kernel log (`journalctl -k`; `xid_unreadable` INFO if it can't be read) | `HOST_CHECKS` (0), `XID_CHECK_MIN` (5) |
| `start` / `stop` / `exit` / `inactive` / `reboot_required` | INFO | log only | |

Each alert goes to `<workdir>/longrun/<instance>/alerts.log` and the journal; WARN and CRIT also to a
desktop notification when a notification server runs (`notify-send`, else D-Bus `Notify` -- nothing is shown
while nobody is logged into the desktop, as on 2026-10-22) and to `ALERT_COMMAND` when set: a shell command
with `ALERT_LEVEL`, `ALERT_KEY`, `ALERT_MESSAGE`, `ALERT_INSTANCE` in its environment, e.g. a phone push
through a service you choose (`--set 'ALERT_COMMAND=curl -s -d "$ALERT_MESSAGE" https://ntfy.sh/<topic>'` --
that sends the alert text to that service). A condition alerts when it starts, again every `ALERT_REPEAT_MIN`
(240) while it lasts, and logs `cleared`. `watchdog.py --instance <i> status` prints the same checks without
alerting. The daily jobs alert through the same path (`watchdog.send_alert`), into their own state dir.

Samples for the daily report: every `RSS_EVERY_MIN` (60) the RSS of each process of the unit, to
`<state dir>/rss.jsonl` (M4's leak check); with `HOST_CHECKS=1`, every `USAGE_EVERY_MIN` (360) a usage sample to
`USAGE_LOG` -- free disk, sizes by category (`USAGE_PATHS`), the worktrees (`WORKTREES_DIR`), the lowest
MemAvailable since the last sample, swap, GPU throttling per clock hour (M1-M5).

## Host stability

**Why.** `unattended-upgrades` is on (`APT::Periodic::Unattended-Upgrade "1"`) and Ubuntu ships the NVIDIA
driver (`nvidia-driver-595-open`, `libnvidia-compute-595`, ...) through the security pocket it installs from.
It upgraded the driver on 2026-05-22, 2026-07-24 and 2026-10-10 (`/var/log/unattended-upgrades/`). An
upgrade replaces the userspace libraries but not the loaded kernel module, so every CUDA process started
after it fails ("Failed to initialize NVML: Driver/library version mismatch") until a reboot: a run that
restarts after an ordinary crash loses the GPU. Kernel upgrades (also automatic) only bite at the next
reboot, but a power cut would then boot a kernel the run was never tested on, with a freshly built DKMS
module.

**Hold both for the run** (needs sudo; run by the user):

```bash
bash scripts/longrun/host_hold.sh status      # module vs libraries, what is held, pending upgrades / reboot
sudo bash scripts/longrun/host_hold.sh hold   # apt-mark hold the driver-series packages + kernel metapackages
sudo bash scripts/longrun/host_hold.sh unhold # after the run; then upgrade and reboot in one go
```

`hold` covers every installed `*nvidia*-595*` package (driver, libraries, DKMS, firmware, utils, the X
driver) and the kernel metapackages (`linux-generic`, `linux-image-generic`, `linux-headers-generic` and
their `-hwe-24.04` versions, so no new kernel is installed). `unattended-upgrades` and `apt upgrade` both skip
held packages; everything else keeps getting security updates. For a second line, the same patterns can go in
`Unattended-Upgrade::Package-Blacklist` (`/etc/apt/apt.conf.d/51rl18xx-hold`: `"nvidia-"; "libnvidia-";
"xserver-xorg-video-nvidia"; "linux-";`). The NVIDIA container toolkit comes from NVIDIA's repository, which
`unattended-upgrades` doesn't use, and doesn't touch the driver. `needrestart` never restarts `user@.service`
(`/etc/needrestart/needrestart.conf`), so a library upgrade can't restart the run's user manager.

**State on 2026-10-22**: kernel 7.0.0-38-generic, module and libraries 595.99.02 (matched), nothing held,
`/var/run/reboot-required` set (linux-image-6.8.0-147, linux-base): reboot before the run starts, not during.

**Planned reboots during the run**: `systemctl --user stop rl18xx-longrun@<i>` first (it waits out a snapshot
write; at shutdown the user manager gives its services only 5 s), reboot, `host_hold.sh status`, start. The
user manager lingers (`Linger=yes`), so an *enabled* run starts at boot by itself; one not enabled waits for
`systemctl --user start`.

## Dry run

```bash
bash <deployment>/scripts/longrun/dryrun.sh <deployment> <scratch dir>   # CPUS=24-31,56-63 by default
```

A tiny CPU search-learning run (random d=32 transformers from `make_tiny_start.py`, 2 workers x 2 games,
8 readouts, a snapshot every 5 updates of 16 rows; an update every 1-3 s) under the real units, with
`CUDA_VISIBLE_DEVICES` empty, pinned and at nice 19, the watchdog every 5 s and `STALL_MINUTES=1`. It checks
each claim and prints PASS / FAIL, then uninstalls its instances (`lr-dryrun`, `lr-dryrun-nan`,
`lr-dryrun-oom`; their files stay in the scratch dir).

**2026-10-22, commit a2f89a4** (deployed once with `create`, then rebuilt in place with `update` four times
-- each build ~75 s on 16 pinned cores): **38 of 38 checks passed in 4 min 22 s**; the log is
`docs/longrun_dryrun_2026-10-22.txt`. The tiny run's unit held 7 processes (launch.sh, main.py, its
resource tracker, 2 inference servers, 2 pool workers) and peaked at 2.7 GB.

| # | What was done | What happened |
|---|---|---|
| 1 | first start | `mode: fresh`; snapshot 5 about 25 s after the start; the watchdog came up with it |
| 2 | `kill -9` main.py at update 7 (snapshot 5) | all 7 processes gone before the restart; systemd restarted it after `RestartSec`; `resume` from snapshot 5; the next history line is update 6 |
| 3 | `kill -9` an inference server at update 10 (snapshot 10) | main.py stopped on `RuntimeError: inference server died`; all 7 gone; restarted; resumed from 10; next line update 11 |
| 4 | `SIGSTOP` main.py for 100 s | `stall` WARN after 1 min, `cleared` within 5 s of `SIGCONT` |
| 5 | `deploy.sh update` while the run is up | refused, listing the run's processes |
| 6 | `systemctl --user stop` | inactive; none of the 7 processes left; no process of the deployment's venv anywhere; the stop recorded |
| 7 | one line appended to the deployment's `search_learning.py` | start refused with 76, not restarted; CRIT `refused` and `failed` |
| 7b | `state.json` cut to 100 bytes, `optimizer.pth` to 1000 | refused with 77; `snapshot_check.py --repair` rebuilt `state.json` at snapshot 15 and moved `optimizer.pth` aside; the run resumed from 15 and trained |
| 8 | a fresh run at `--lr 1e30` | `RuntimeError: search learning diverged: non-finite loss`: exit 75, not restarted; CRIT `nonfinite` |
| 9 | `MemoryMax=1500M` (below the 2.7 GB the run needs) | every kernel OOM kill `constraint=CONSTRAINT_MEMCG, oom_memcg=.../rl18xx-longrun@lr-dryrun-oom.service`, at `oom_score_adj=500`, none global; 5 restarts, then the start limit; the user manager and claude-rc kept their PIDs; CRIT `crash_loop` at the 3rd restart, CRIT `failed` ("start limit ... last failure: an OOM kill") |
| 10 | `DISK_MIN_GB=1000000`, `GPU_TEMP_MAX_C=1` | `disk:` and `gpu0_temp` alerts; `ALERT_COMMAND` received them (and the instances' alerts) |

Harness bugs the dry runs found, all fixed before the last run: `install.sh` wrote a rejected value as an
empty one (its check ran inside `$(...)`); when the start limit stops automatic restarts systemd leaves
`Result` at the last failure (`oom-kill`), not `start-limit-hit` (that is only for a refused manual start),
so the watchdog now reads a failed unit by its exit status; `OOMPolicy=stop` SIGTERMs the launcher while it
writes its exit record, so the crash loop counts systemd's `NRestarts` and the launcher ignores SIGTERM once
main.py has exited; `--repair` resolved the run's relative starting-policy path against the caller's
directory.

Not covered: the GPU (CUDA was hidden; GPU memory is freed when the processes exit, which checks 2, 3 and 6
show); `enable` and a reboot; a desktop notification on screen (no notification server was running: the
D-Bus call is made and fails quietly); `host_hold.sh hold` (needs sudo; `status` was run).

## The long run (2026-11)

docs/long_run_recipe.md's run: three long-lived instances of `rl18xx-longrun@.service` and the daily jobs, all
from one frozen deployment `D` in one workdir `W`. The settings live in `scripts/launch/2026-11/`
(`install_lr1.sh` -- the install commands; `env-examples/` -- the `.env` files it writes; `eval_ladder_longrun.json`
-- the ladder's panel; `league_longrun.json` -- the league's fixed members; `retention.json`). The names
(`lr1`, `saved_games/lr1`, `model_checkpoints_value/lr1`, `logs/eval/ladder_lr1`) appear in all of them: change
them together.

| instance | kind | runs | resumes | progress (stall check) | memory (High / Max) |
|---|---|---|---|---|---|
| `lr1-gen` | generator | `main.py search-learning`: checkpoint 10 frozen, `--value-follow` the value run, league seats (`--population league_longrun.json`, 30%), every game saved (`--save-data saved_games/lr1`) | `--resume` from its snapshot | `history.jsonl`, 90 min | 120G / 135G |
| `lr1-value` | value | `main.py value-train --run-dir model_checkpoints_value/lr1 --data saved_games/lr1 --init <v0>` | its own `state.json` | `status.json`, 120 min | 36G / 44G |
| `lr1-ladder` | ladder | `eval_ladder.py run --panel eval_ladder_longrun.json` (12 workers) | its own `state.json` | `logs/eval/ladder_lr1/heartbeat`, 90 min | 26G / 32G |
| `lr1-daily` | daily (timer, 06:00) | retention, style statistics, integrity check, status report | -- | -- | 12G / 16G |

Memory: the recipe's budget table says ~60 / 30 / 20 GB, but azl alone held ~80 GB at 40 workers x 32 games
(2026-10-22) and the generator now also serves up to 12 league servers; the caps are those numbers plus margin, 211
GB together for the three. Measure the peaks in the G0 pilot (`systemctl --user status`, the watchdog's
`rss.jsonl`) and re-run the installer with tighter caps. Only `lr1-gen`'s watchdog does the host checks
(`HOST_CHECKS=1`: usage samples, Xid scan, GPU) -- three watchdogs would triple every GPU alert.

### Setting it up

1. **Deployment** as above (`deploy.sh create longrun-2026-11 ~/longrun/deploy/longrun-2026-11`), dry-run it.
2. **Workdir.** Real directories for everything the run writes (never a symlink into `.claude/worktrees`: the
   desktop app cleans those up, and retention refuses to delete through symlinks), and real copies of the panel's
   and the league's checkpoints and of the value gate's files, at the paths the panel / league / install files
   name (they live in session worktrees today):

   ```bash
   W=~/longrun/work/lr1; G=/home/revys/workspace/RL_18xx/.claude/worktrees/gpu-experiments-training-93d287
   V=/home/revys/workspace/RL_18xx/.claude/worktrees/1830-rl-analysis-ea948b   # the value gate's outputs
   mkdir -p $W/logs $W/longrun $W/saved_games $W/model_checkpoints_value $W/model_checkpoints_search
   ln -s /home/revys/workspace/RL_18xx/human_games $W/human_games
   S=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562
   cd $G && cp --parents -a $S/10.pth $S/config.json \
       model_checkpoints_pg/rab_place_20261014/{learner,critic}/1800.pth model_checkpoints_pg/rab_place_20261014/config.json \
       model_checkpoints_pg/gx2_{bankrupt,rust}_20261013/learner/300.pth model_checkpoints_pg/gx2_{bankrupt,rust}_20261013/config.json \
       model_checkpoints_search/az8_20261019/learner/170.pth model_checkpoints_search/azl_20261019/learner/2100.pth \
       model_checkpoints_search/{az8,azl}_20261019/config.json $W/
   cd $V && cp --parents -a logs/value_gate/data logs/value_gate/runs/base_36989/{learner/0.pth,config.json} $W/
   cd $W && $D/.venv/bin/python -s $D/scripts/eval_ladder.py check --panel $D/scripts/launch/2026-11/eval_ladder_longrun.json
   ```
   (`check` lists every checkpoint the panel names; `model_checkpoints_value/lr1/promoted/0.pth` shows as missing
   until the value trainer writes version 0. The runs' `config.json` files: loaders read a run's temperatures and
   reward from them.) Every path in `league_longrun.json` must exist too (verified 2026-10-23 against `$G` and `$V`).
3. **Install** (writes the four `.env` files, the unit templates, the timer and the limits; starts nothing):
   `bash $D/scripts/launch/2026-11/install_lr1.sh $D $W` (set `ALERTS` in it, or `--set ALERT_COMMAND=...` per
   instance, for phone alerts).
4. **Start, in this order** -- the value trainer first: it writes `promoted/0.pth` (v0 = the start value, its
   fixed-set CE measured), which the generator follows and the ladder's u0 is:

   ```bash
   systemctl --user start rl18xx-longrun@lr1-value
   systemctl --user start rl18xx-longrun@lr1-gen
   systemctl --user start rl18xx-longrun@lr1-ladder
   systemctl --user enable --now rl18xx-longrun-daily@lr1-daily.timer
   ```

### Day 0: the baselines

- **The ladder** evaluates u0 (checkpoint 10 + v0, update 0) against the whole panel first (`baseline_update`:
  2v2 + 1v3, 240 games per direction, ~3,840 games, ~9 h at 12 workers); the ratings are anchored on u0 (0 Elo).
- **The value**: v0's fixed-set CE is `promoted.json`'s version 0 (`fixed_ce`; ~0.920 expected).
- **Style and everything else**: run the daily jobs once by hand at the end of the first day
  (`systemctl --user start rl18xx-longrun-daily@lr1-daily.service`). The first style-statistics line is the drift
  guard's baseline (`--baseline first`); the status report records each metric's baseline in
  `longrun/lr1-daily/baseline.json` the first time it has a value (P1 / P2 / P7: the first day with data; P3: v0;
  P5: u0 = 0 Elo). To re-record (e.g. after a bad first day): `cd $W && $D/.venv/bin/python -s
  $D/scripts/longrun/status_report.py --set-baseline P1,P6` (with the daily `.env`'s variables exported) or
  `--set-baseline` for all. `RUN_START` in the daily `.env` fixes day 0 for the milestones (default: when the
  baseline file was created).

### The ladder's schedule

`eval_ladder_longrun.json`: candidates are the value trainer's promoted versions (`run` = the value run,
`checkpoint` = `promoted/{update}.pth`, `every` 1: the newest version whenever the ladder is free), searched
under checkpoint 10 at ptemp 1 (`<ckpt10>+{checkpoint}@200/0.05:mean,dedupe,par=8,ptemp=1,budget=total`). A
candidate plays either a **rotation** -- two panel members taken in turn from the panel's order (u0, gen4critic,
az8_u170, azl_u2100, sl10_gen4critic), 2v2 only, 120 games per direction (240 per pairing): the daily P5 -- or,
when no full schedule started in the last 168 h, the **full panel**: every member, 2v2 + 1v3, 240 games per
direction (480 per pairing), plus the weekly **value check** against the previous full candidate (2v2, 480 games;
`purpose: value_check` in the ledger; against u0 in week 1, whose own panel matches are tagged). The schedule of
each candidate is kept in `logs/eval/ladder_lr1/state.json`, and each ledger record says its `schedule` and `time`.
`eval_ladder.py decide --panel ... --full --back 1` compares the last two full candidates (the weekly review);
the ladder's own `report.txt` / `ratings.json` do the same. A `STOP` file in `logs/eval/ladder_lr1/` ends the
ladder before its next match.

### Daily jobs and where everything lands (under `W`)

| output | written by | what |
|---|---|---|
| `long_run_log.md` | status report | the dated daily sections (P1-P8, M1-M9, rules, milestones); the weekly reviews go here too |
| `longrun/lr1-daily/status.jsonl`, `baseline.json`, `alerts.log` | status report | each report as JSON; the baselines; the yellow / red alerts |
| `longrun/lr1-daily/style_stats.jsonl` | style statistics | one line per day: par shares, par prices, first-train timing, withhold rate, bankruptcies, length; main seats and all seats; drift flags |
| `longrun/lr1-daily/integrity.jsonl`, `shard_checksums.txt` | integrity check | 50 random games of the last day rebuilt twice bit for bit; every shard's sha256, 20 re-hashed a night |
| `longrun/lr1-daily/retention.jsonl` | retention | what it deleted or would delete |
| `longrun/lr1-daily/daily.log`, `daily_steps.json` | daily.sh | the jobs' output; their exit statuses |
| `longrun/usage.jsonl` | lr1-gen's watchdog (6-hourly), the status report (daily) | disk, sizes by category, worktrees, memory, swap, throttling |
| `longrun/lr1-{gen,value,ladder}/` | launcher, watchdog | starts / exits, alerts, `rss.jsonl`, watchdog state |
| `logs/longrun_lr1-*.log` | launcher | each process's output |
| `model_checkpoints_search/lr1/` | generator | `history.jsonl` (P1, P2, P7), snapshots, `league.json`, `league/` |
| `model_checkpoints_value/lr1/` | value trainer | `promoted.json`, `promoted/`, `candidates/`, `cycles.jsonl` (P3, P4, M8), `status.json` |
| `logs/eval/ladder_lr1/` | ladder | `ledger.jsonl`, `ratings.json`, `report.txt`, the matches (P5, P8) |
| `saved_games/lr1/` | generator | every game (`.sld` shards) |

**Status report rules** (`status_report.py`; "days" are 24-h windows ending at the report): yellow for each of
the recipe's daily alarms (P1 below its 5-day max by >= 2 SE; two promotions without a fixed-set gain; the
fixed-minus-rolling CE gap widening 3 cycles running; P2 up >= 2 SE in 3 days with P1 flat; P5's 3-day mean below
the previous 3 days' by >= 2 SE; a drift flag; throughput < 80% of the 7-day median or > 3% of games at the length
cap), 3 failed promotions in a row, one weekly value check < 0.47, a missed milestone (on its date), any WARN the
watchdogs hold, a failed daily job, and the maintenance metrics' yellows; red for stall (P1's 5-day slope upper
95% bound < +0.001/day **and** P3's slope over the last 3 promotions >= 0 **and** P5 not up by 2 SE over 6 days),
decline (P1's 5-day mean below its best by >= 3 SE **and** P5's 3-day mean below its best by >= 2 SE), value
divergence, 6 failed promotions, two value checks < 0.47 running, saved-data write errors, any CRIT a watchdog
holds (crash loop, non-finite stop, Xid, disk), and the maintenance reds. A yellow report sends a WARN `status`
alert, a red one CRIT. P5's SEs are the ratings' SEs against u0 (comparisons between candidates count u0's
uncertainty twice: conservative).

**Maintenance thresholds** (`.env` of the daily instance; recipe defaults): `M1_YELLOW_DAYS` 30, `M1_RED_DAYS` 14,
`M1_RED_FREE_GB` 150; `M2_EXPECTED_GB_PER_DAY` (`saved=1.4,checkpoints=0.5,evals=0.06,logs=0.05`), `M2_FACTOR` 2;
`M3_YELLOW_GB` 200, `M3_RED_GB` 300; `M4_LEAK_GB_PER_DAY` 2, `M4_SWAP_MIN_GB` 0.5, `M4_MEM_RED_GB` 30;
`M5_THROTTLE_FRACTION` 0.10; `M7_HOLD_CHECK` 1 (`host_hold.sh check`); `M8_TARGET_HOURS` 1, `M8_FACTOR` 2;
`BACKUP_DIR` / `M9_MAX_DAYS` 8 (M9 is n/a without a backup directory). On 2026-10-23 this host had 1.8 GB of swap
in use and a pending reboot: both are yellows until fixed (or `M4_SWAP_MIN_GB` raised).

**Retention** (`retention.py --config retention.json`, daily): a dry run until `dry_run_until` (set it to launch
day + 7 once the policy is approved), listing what it would delete in `retention.jsonl` and the daily report; then
for real. Generator snapshots thinned (every 10th update for 48 h, every 100th for 2 weeks, every 1000th
forever; the resume snapshot, the anchor, every checkpoint the panel / ledger / ladder state / `league.json` /
population names are kept), league snapshots no longer in `league.json` after 7 days, unpromoted value candidates
after 7 days, optimizer state (latest + one per week), logs rotated at 1 GB (4 kept). Hard guards in the code:
saved games, promoted values, eval collections, the ladder ledger and runs' metadata are never touched, nor
anything outside the workdir or behind a symlink; a config that asks for one is refused and nothing is deleted
that day (WARN `retention` alert). Test changes to the policy on a copy, never on the run's workdir:
`retention.py --config <file> --workdir <copy> --dry-run`.

## Rules for agents (Claude sessions)

- Never edit, check out, merge, rebase, `uv sync` or build in a deployment (its `CLAUDE.local.md` says so;
  its files are read-only). A change is a new deployment.
- Look at a run read-only: `watchdog.py --instance <i> status`, the log, `history.jsonl`, `alerts.log`,
  `systemctl --user status`. Starting, stopping, restarting or reconfiguring the run is the user's call.
- Run evals as their own transient units with a `MemoryMax` (above), pinned away from the run if it is
  pinned, and budget their RAM against the run's `MemoryMax`.

## Known gaps (in the training code, not the harness)

- `search_learning.run` and `policy_gradient.run` write `optimizer.pth` (`torch.save`) and `state.json`
  (`write_text`) in place; `checkpointer._atomic_save_torch` / `_atomic_write_json` would close the torn-snapshot
  case that `snapshot_check.py --repair` exists for.
- The replay window is not in the snapshot: every restart refills it (`--min-window-rows`) before the next
  update, and repeats the updates since the last snapshot in `history.jsonl` / TensorBoard.
