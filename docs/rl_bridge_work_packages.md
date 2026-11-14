# RL bridge: work packages (2026-11-13)

Briefs for the sessions that build the title-generic RL bridge, scoped by the user's decisions of
2026-11-13 (top of `docs/rl_bridge_design.md`). Start each from branch `gen6-mixed-count` (it has the
latest policy-gradient code: mixed player counts, per-count PFSP, `min_share_players`) in its own
worktree. Packages A and B run in parallel; C starts once A's records and B's model exist.

Common constraints (all packages):
- 1830 must not change: `action_index.rs::build_layout` (pinned by `layout_matches_frozen_1830_constants`),
  `factored.rs` enumeration, `encoder.rs`, `action_mapper.py`, `encoder.py`, `model_transformer.py`.
  After any edit to a shared file (`mcts.rs`, `decode.rs`, `price_pmf.rs`, `lib.rs`, `title/mod.rs`) run the
  frozen-layout tests, the 1830 parity gates (`tests/parity_runner.py --human` on a few hundred games) and pytest.
- Shared machine: cargo builds and heavy tests as `taskset -c 24-31,56-63 nice -19 ...`, cargo `-j 8`,
  modest RAM (two packages build at once). No GPU until the user frees it.
- Never touch `/home/revys/longrun/` (running deployments and the bots), `.claude/worktrees/autoplay`, or
  another session's worktree. Ask before deleting anything. Commit on the package's own branch; never push.
- `lib.rs` registration is shared between A and B: keep edits there small.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## A. Action layout, indexing and import (Rust + Python)

Design steps S1, S2, S4, the action half of T7b, T7c at the index level, T7d.

1. `rl18xx/agent/alphazero/titles.py`: `TitleSpec` (player range 3-6 for new titles, optional rules,
   data paths, eval defaults) and one `new_game(title, ...)` factory; title/layout metadata written into
   game records, data and checkpoints (written, not enforced yet).
2. Rust `layout_for(title)` wherever an index is computed or decoded (`factored_legal_indices`,
   `decode_index`, `price_head_slot_for_index`, MCTS calls); split `alphazero_bridge_ready` so a 1867 call
   through the 1830 path raises "not bridge-ready" instead of mis-indexing.
3. `engine-rs/src/rl/` block registry: action families with axes from title data, a **descriptor** per
   index (family, entities, hex, tile, rotation, parameter), a descriptor table for the frozen 1830
   layout, a layout fingerprint test per title. **Fix the descriptor schema first and document it**
   (`docs/rl_descriptors.md`): package B's policy head is built on it.
4. The 1867 layout from the generator's families (`g1867/legal.rs`: single-item and corporation bids, par
   cells, buy/sell bundles, two-lay tile lays, distance-priced tokens, home tokens, discounted trains and
   trade-ins, merge / convert / remove_token, MajorTrainless choose, ...). Every generated move maps to
   exactly one index + descriptor. Prices go to price heads (design §2.7), never a fixed price menu; price
   specs from a 1867 human price census. The design's step-driven enumerator (S5) is not needed for titles
   that have a generator.
5. Verification (design §2.8): indices injective; decode round-trips to the same move; every generated move
   indexed (fuzz at 3-6 players); index-level corpus completeness on the 1867 human games
   (`human_games/1867/`, plus the 2026-10 scrape that `scripts/g1867_completeness.py` uses).
6. Import (T7d): 1867 human games, 3-6 players, base rules, into training records (state reference, legal
   indices, chosen index, price target, outcome). Keep forced positions (they helped 1830's pretraining).
   Each human action must match exactly one index: skip and count ambiguous or unmatched ones, never guess
   (as `pretraining._matching_choices` does for 1830). Report match rates.

Done: committed with tests green (1830 frozen tests and parity gates unchanged), the descriptor schema
documented, and a short report: fuzz, index-level completeness, import match rates, price census.

## B. Generic encoder and model (Rust + Python)

Design step S7 (§2.6, §2.7, §3, §4.5), without the GPU validation for now.

1. Generic typed-token encoder in Rust (`engine-rs/src/rl/encode.rs`): hex tokens with title-generic tile,
   track, stop and border features (a token in a city slot is a pointer to a corporation token); entity
   tokens for players, corporations of every kind (major, minor, national), companies, train types, phases
   and one global token, each type with a fixed title-independent schema; entity relations (holdings,
   presidency, ownership, trains held) as an attention bias; padding and masks for variable counts;
   per-title vocabularies and normalizers in an `RlSpec`. Batched native encoding, stacked numpy arrays
   (as 1830's packed path does), so inference servers stay fast.
2. `rl18xx/agent/alphazero/model_generic.py`: hex and entity transformers, cross-modal fusion, FiLM trunk
   conditioned on title + round kind + step kind; a **descriptor-conditioned policy head that scores only
   the legal actions**, keeping the P(hex)·P(tile|hex)·P(rot|hex,tile) factorization for tile lays;
   a pointer-keyed price head (bucketed price distribution, as 1830's price head); the per-seat value head
   with `MAX_SEATS` 6 and padded seats masked; the critic as the same architecture with the policy path off.
3. Build against package A's descriptor schema (`docs/rl_descriptors.md`); until it lands, use a stub
   descriptor batch with the design's fields and keep the adapter thin.
4. CPU tests: encoder on 1830 and 1867 positions (random walks via `g1867_legal_moves_json` and the 1830
   engine), shapes and masks across 3-6 players, model forward/backward, legal-only softmax, price head,
   bf16 training / fp32 inference numerics as 1830 does.

Done: committed with tests green and a note in the design doc of what the 1830 GPU check (§4.4: a short
supervised run on 1830's human data vs SL ckpt 10) will need once the GPU is free.

## C. Pipeline (after A and B)

Design step S8, trimmed to the policy-gradient path: dataset and supervised training for the generic
model, a sparse-legal inference server path, `policy_gradient.py` per title (mixed 3-6 player counts,
per-count PFSP, placement reward), `eval_policy_only.py` per title, 1867 start positions (human
post-auction states, random auction endings, mid-game cuts) and an out-of-family panel. Then the 1830
check of the generic model and the first 1867 supervised run, once the GPU is available.

Status (2026-11-13): C1 (dataset, supervised training, policy-only player, 1830 GPU check recipe) and C2
(generic inference server, policy-gradient trainer, per-title starts and start-file builder, title-generic
eval, 1830 PG recipe) are built and CPU-tested (`docs/rl_model.md`); nothing has run on the GPU. Open for 1867:
A2's layout and importer, then the start files, supervised run and PG run of `docs/rl_model.md` "1867 outline".
C3 (2026-11-13): mixed-title supervised training (`pretrain-generic --data A --data B --mix ...`, `--init`
fine-tuning) and the CPU weight-sharing probe (`scripts/sharing_probe.py`); result on 0.54M-parameter models: no
sharing signal for 1867 at 100 / 300 / 885 games by the pre-set rule (`docs/rl_model.md` "The sharing probe").
