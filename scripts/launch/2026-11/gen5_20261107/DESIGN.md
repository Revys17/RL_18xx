# gen5 (queued 2026-11-07, starts when the 2-player loop ends)

The user's plan: policy gradient (our most reliable way to train the policy) from gen4, under the current rules, with
real auction training. Later: a value trained offline on this league's games (lr1-style broad tables), search at play,
one network by distillation at the end.

Departures from gen4's run (rab_place_20261014's config.json), named before launch:
- start: gen4 learner + critic 1800; rules: the 2-OR train rule (on by default since 2026-10-31).
- auction starts 50% of games (gen4 15%), real rules (--no-auction-unlock), --auction-stall-decisions 150 (a game still
  in the auction after 150 decisions ends, scored on net worth as it stands, like the 1000-decision cap).
- league: league_v7 (30 members) + gen4 itself (league_v7_gen4.json); own snapshots every 50 updates as before.
- start files: the _v2 files (current engine).
- 1000 updates, then a check: vs gen4 2v2 + 1v3 policy-only, auction prices vs humans (history auction_price_*).
- Network: today's 8.3M (the size test: half 2.3M loses ~.03 policy-only, quarter collapses, and whole policy-only
  games ran at the same speed for all three sizes -- the engine, not the network, sets the pace).

Reference, human 4-player auction prices (1,945 games, mean): CS 48, DH 87 (median 80), MH 135, CA 208, BO 222.
gen4 in real-rules auctions (12-game smoke test, CPU): CS 68, DH 89, MH 172, CA 282, BO 220.

Files: run.sh (resumes when the run exists), queue.sh (unit rl18xx-gen5-queue; touch HOLD to keep it from starting).
Run dir: /home/revys/longrun/work/lr1/model_checkpoints_pg/gen5_20261107.

## Decision (2026-11-07 ~17:30, the user: "yes")
Launch as queued: credit by the game's outcome (gae_lambda 1, as gen4), placement reward, auction stall limit 150
scored by net-worth placement. Fallback, if the logged auction prices (auction_price_*) haven't moved toward humans'
(CS 48, DH 87, MH 135, CA 208, BO 222; gen4 under real rules: 68 / 89 / 172 / 282 / 220) within 100-200 updates or
auction_stalls is not rare: gae_lambda < 1 for auction decisions only, with a value that judges auction positions
(lr1 v4) as their baseline.

## 2026-11-09: deployment changed
run.sh now uses /home/revys/longrun/deploy/az0-2p-7e21baa (branch az0-2p-20261108 at 7e21baa). gen5-20261107 (ae3438e) is
its ancestor, so gen5's code is unchanged; on top it carries the 2026-11-08 audit fixes that touch policy gradient:
samplers renormalize over the legal moves before tempering, and league games are credited to the member that played
them (games whose slot changed members go unscored). Bankrupt players keep Ruby's scoring (the user, 2026-11-09: going
bankrupt can be a winning move). Old run.sh kept as run.sh.orig-20261109. HOLD is still in place.

## 2026-11-10 07:18: launched (by Claude, after the 2-player stages per the agreed order)
Unit rl18xx-gen5, as queue.sh would have started it, plus the MPS client environment (the user: use MPS). No change to
run.sh. Added evals.sh (unit rl18xx-gen5-evals): every 100 updates, policy-only vs gen4 2v2 and 1v3 from the auction
(real rules, stall 150), 400 games each, lines in results.txt. What the 2-player run showed for this recipe: 2000
updates of steady gains, no auction stalls; and at play, search with the PG critic beat the policy alone 0.82 while a
value refit offline on the policy's games (better CE) searched worse than the critic (0.43 head to head) -- relevant to
the "value trained offline" step planned after this run.

## 2026-11-10 08:40: the auction-price statistic

history.jsonl's `auction_price_*` averages every buyer, and the league's frozen members overpay (u1-150: MH 171-192, CA 275-283, DH 107-113). The learner's own prices, from the saved games (`learner_prices.py`): BO 220, CA 246-258 (it buys only 9-17% of the CAs against a 45% seat share), CS 66-70, DH 91-93, MH 135-141, against humans' 222/208/48/87/135. The fallback test ("prices moved toward humans' within 100-200 updates") reads the learner's split.

## 2026-11-10 15:05: past 1000, until it plateaus (the user)

At u600 (2v2 .746, 1v3 .502 vs gen4, still rising) the user: keep going until it plateaus. run.sh's resume branch now
passes `--max-updates 4000` (a backstop, ~36 h more at ~43 s/update); a resume may raise the cap (RESUME_SETTINGS).
evals2.sh (unit rl18xx-gen5-evals2) replaced evals.sh: every 100 updates vs gen4 (2v2 + 1v3, 400 games, as before)
and vs the checkpoint 200 updates older (2v2 + 1v3, 800 games; backfilled from u300); when the run stops at its
original 1000 cap it starts it again, once. Plateau rule (plateau.py, chosen by Claude -- pg2p's "H vs H-100 < .55"
would fire at once in 4 players, where a hundred is worth ~.52): a hundred is flat when H vs H-200 is below 0.53 at
2v2 AND below 0.28 at 1v3 (fair .50 / .25; about 1.7-1.9 SE at 800 games); two flat hundreds in a row from u1000 on
touch the run's STOP file (earliest stop u1100).

## 2026-11-10 19:07: the restart at 1000 failed; restarted by hand

evals2.sh's `systemctl --user start rl18xx-gen5` at the 1000 cap found no unit: rl18xx-gen5 was a transient
(systemd-run) unit, which is removed once it exits cleanly. evals2 logged the restart anyway, then saw the run inactive
and ended. Claude started both again at 19:07 with the 07:18 commands (gen5 with the MPS env; it resumed from u1000
with max_updates 4000, update 1001 at 19:09); ~6 min of GPU lost. Lesson: to restart a transient unit, re-run
systemd-run, never `systemctl start`.

## 2026-11-11 ~18:55: stopped by the user at ~u2880; final = u2800
The u-200 plateau rule never fired after u1800 (u2700/u2800 beat u2500/u2600 by .667/.665 at 2v2), but an
out-of-family panel (outfamily.sh: one gen5 seat vs three of gen4 / SL ck10@0.5 / gen2 / gx2_rust / gx2_bankrupt,
800 games each) was flat: mean .590 u1600, .595 u2200, .606 u2700, .593 u2800. The rule measured within-family
progress (the league's hardest opponents were gen5's own recent snapshots). The user: start gen6 and put u2800 in the
bots. STOP touched by hand; rl18xx-bot-swap ran `swap_bots.sh 2800` (5004/5002/5014 all healthy on gen5 u2800 at
18:54); gen6's queue was restarted with G5_PIN=2800.
