// Autoplay: the model plays chosen seats of a game by itself -- a hotseat game, or (when the
// extension's options allow it: ``autoplayLive``) a live game this browser's login plays a seat of.
//
// Each round asks the page bridge (bridge.js, in the page's world) who the site's own
// Ruby engine waits for. On a model seat's turn it sends the game and that answer to
// the backend (POST /api/move through the background worker), which searches the
// position and returns the move in the site's format -- or the "pass" Ruby waits for
// where our engine has already moved on (see rl18xx/agent/advisor/autoplay.py) -- and
// hands the move to the bridge, which plays it only if Ruby accepts it and the game
// hasn't moved meanwhile (a live move: posted to the site, whose server checks it again).
// On a human seat's turn it looks again every POLL_MS (hotseat: a localStorage read, no
// network) or LIVE_POLL_MS (live: a read of the game from the site; a new line in the page's
// log looks at once). With ``dryRun`` a live move is only checked and shown, never posted.
// It stops on anything unexpected and says why; nothing is retried blindly.
(function (root) {
  'use strict';

  const R = root.RL18XX && root.RL18XX.escapeHtml ? root.RL18XX : require('./common.js');

  const POLL_MS = 1500; // a human seat's turn: look again this often
  const LIVE_POLL_MS = 30000; // ... in a live game (a read from the site; the page's log changing pokes sooner)
  // A live read may reuse one this recent: after a poke (a new line in the page's log, which comes
  // over the page's own connection at least a debounce earlier) only a fresh one; otherwise any recent.
  const LIVE_FRESH_MS = 1000;
  const LIVE_RECENT_MS = 15000;
  const JOB_POLL_MS = 400; // a running search: ask for its progress this often
  const BUSY_RETRY_MS = 1000; // the backend runs another search ("Think harder")
  const RETRY_MS = 4000; // the page or the backend isn't ready
  const MAX_PASSES_IN_A_ROW = 8; // implicit passes without another move: something is off
  const KEEP_MOVES = 12;
  // Ruby steps where our engine moves on by itself while Ruby still offers "pass"
  // (scripts/ruby_replay_dump.rb's wait reasons: purchases from other players).
  const IMPLICIT_PASS_STEPS = ['BuyTrain', 'BuyCompany'];

  function lastActionId(actions) {
    for (let i = (actions || []).length - 1; i >= 0; i -= 1) {
      if (actions[i] && typeof actions[i].id === 'number') return actions[i].id;
    }
    return 0;
  }

  /** ``env``: ``bridge(op, args)`` (the page bridge), ``send(message)`` (the background
   * worker), ``now()``, ``setTimeout`` / ``clearTimeout``, ``onChange()`` (redraw); optional
   * ``onPlayed(move)`` and ``onStopped(gameId)``. */
  function createAutoplay(env) {
    const state = {
      gameId: null,
      enabled: false,
      paused: false, // the site's history view is open
      seats: [],
      readouts: 200,
      phase: 'off', // off / waiting / thinking / playing / over / error
      text: '',
      progress: null,
      moves: [], // the model's latest moves, newest first
      players: [],
      dryRun: false, // live games: check and show the model's move, never post it
      dryRunAt: null, // the last action id a dry-run move was shown for
      running: false,
      again: false,
      poked: false, // something changed since the last look: read the game fresh
      timer: null,
      passes: 0,
    };

    function changed() {
      if (env.onChange) env.onChange();
    }

    function setPhase(phase, text, progress = null) {
      state.phase = phase;
      state.text = text || '';
      state.progress = progress;
      changed();
    }

    function live() {
      return state.gameId !== null && !R.isHotseat(state.gameId);
    }

    function idleMs() {
      return live() ? LIVE_POLL_MS : POLL_MS;
    }

    function active(id = state.gameId) {
      return id !== null && id === state.gameId && state.enabled && !state.paused && state.seats.length > 0;
    }

    function clearTimer() {
      if (state.timer) env.clearTimeout(state.timer);
      state.timer = null;
    }

    function schedule(ms) {
      clearTimer();
      state.timer = env.setTimeout(() => {
        state.timer = null;
        tick();
      }, ms);
    }

    function sleep(ms) {
      return new Promise((resolve) => env.setTimeout(resolve, ms));
    }

    // ------------------------------------------------------------ settings
    /** Follow hotseat game ``id`` (null: none) with its saved ``settings``. */
    function setGame(id, settings = {}) {
      clearTimer();
      Object.assign(state, {
        gameId: id, enabled: false, seats: [], moves: [], players: [], passes: 0, again: false, dryRunAt: null,
      });  // prettier-ignore
      configure(settings);
    }

    /** ``{enabled, seats, readouts, dryRun}`` (any subset). */
    function configure(settings = {}) {
      if (typeof settings.dryRun === 'boolean') state.dryRun = settings.dryRun;
      if (Array.isArray(settings.seats)) state.seats = settings.seats.map(Number).filter((s) => s >= 1).sort();
      if (Number.isFinite(Number(settings.readouts)) && Number(settings.readouts) > 0) {
        state.readouts = Number(settings.readouts);
      }
      if (typeof settings.enabled === 'boolean') state.enabled = settings.enabled;
      if (!state.enabled || state.gameId === null) {
        clearTimer();
        setPhase('off', state.seats.length ? '' : 'Choose the seats the model plays.');
        return;
      }
      if (state.phase === 'off' || state.phase === 'error' || state.phase === 'over') setPhase('waiting', 'Starting…');
      poke();
    }

    function setPaused(paused) {
      if (paused === state.paused) return;
      state.paused = paused;
      if (paused) {
        clearTimer();
        if (state.enabled) setPhase('waiting', 'Paused while the history view is open.');
      } else poke();
    }

    /** Look at the game soon (a move landed, or the settings changed). */
    function poke() {
      state.poked = true;
      if (active()) schedule(0);
    }

    // ---------------------------------------------------------------- loop
    async function tick() {
      if (!active()) return;
      if (state.running) {
        state.again = true;
        return;
      }
      state.running = true;
      clearTimer();
      const id = state.gameId;
      let next = null;
      try {
        next = await step(id);
      } catch (e) {
        next = fail(`Autoplay stopped: ${(e && e.message) || e}`);
      } finally {
        state.running = false;
      }
      if (!active(id)) return;
      if (state.again) {
        state.again = false;
        next = 0;
      }
      if (next !== null) schedule(next);
    }

    /** Stop playing and say why (the user starts again from the panel). */
    function fail(text) {
      state.enabled = false;
      setPhase('error', text);
      if (env.onStopped) env.onStopped(state.gameId);
      return null;
    }

    function seatOf(site, playerId) {
      const players = site.players || [];
      const i = players.findIndex((p) => p.id === playerId);
      return i < 0 ? null : i + 1;
    }

    function who(site, seat) {
      const name = (site.players[seat - 1] || {}).name || `Seat ${seat}`;
      const entity = site.entity || {};
      return entity.type === 'player' ? name : `${entity.id} (${name})`;
    }

    /** One look at the game; the delay until the next (null: stop). */
    async function step(id) {
      const maxAge = state.poked ? LIVE_FRESH_MS : LIVE_RECENT_MS;
      state.poked = false;
      const site = await env.bridge('inspect', { game: id, max_age_ms: maxAge });
      if (!active(id)) return null;
      if (!site || !site.ok) {
        setPhase('waiting', `Waiting for the page: ${(site && site.error) || 'no answer'}`);
        return RETRY_MS;
      }
      state.players = site.players || [];
      if (site.finished) {
        state.enabled = false;
        setPhase('over', 'The game is over.');
        return null;
      }
      const seat = site.entity ? seatOf(site, site.entity.player) : null;
      if (seat === null || !state.seats.includes(seat)) {
        const text =
          seat === null ? 'Waiting for the next move.' : `${who(site, seat)} to move${live() ? '.' : ' — your turn.'}`;
        if (state.phase !== 'waiting' || state.text !== text) setPhase('waiting', text);
        return idleMs();
      }
      if (state.dryRun && live() && state.dryRunAt === site.last_id) return idleMs(); // shown; wait for a change
      const game = site.data;
      if (!game || lastActionId(game.actions) !== site.last_id) return 0; // it moved meanwhile: look again
      const label = who(site, seat);
      setPhase('thinking', `${label} is thinking…`, 0);
      const response = await env.send({
        type: 'move',
        game: R.stripGame(game),
        seats: state.seats,
        readouts: state.readouts,
        site: {
          last_id: site.last_id, entity: site.entity, step: site.step, actions: site.actions, finished: site.finished,
          trains: site.trains, depot: site.depot,
        },  // prettier-ignore
      });
      if (!active(id)) return null;
      if (!response || !response.ok) {
        const data = (response && response.data) || {};
        if (data.busy) {
          setPhase('thinking', `${label}: waiting for another search to finish…`, 0);
          return BUSY_RETRY_MS;
        }
        if (response && response.unreachable) {
          setPhase('waiting', `Backend unreachable: ${response.error}`);
          return Math.max(RETRY_MS, idleMs());
        }
        return fail((response && response.error) || 'No answer from the background worker');
      }
      const job = await waitForJob(response.data, id, label);
      if (!job) return null;
      if (job.status !== 'done' || !job.result) return fail(`The model's move failed: ${job.error || job.status}`);
      const result = job.result;
      setPhase('playing', `${result.actor}: ${result.description}`);
      const dryRun = state.dryRun && live();
      let applied = await env.bridge('apply', {
        game: id, action: result.action, last_id: site.last_id, dry_run: dryRun,
      });  // prettier-ignore
      if (!active(id)) return null;
      if (applied && applied.stale) return 0;
      if (applied && applied.ok) return played(result, applied.action);
      if (applied && applied.dry_run) {
        state.dryRunAt = site.last_id;
        setPhase('waiting', `Dry run — would play ${result.actor}: ${result.description}`);
        return idleMs();
      }
      // Ruby refused it. Where our engine skipped a purchase step Ruby still waits in
      // (BuyTrain / BuyCompany, offering "pass"), pass that step first and look again.
      const refusal = (applied && applied.error) || 'no answer from the page';
      if (result.action.type !== 'pass' && site.actions.includes('pass') && IMPLICIT_PASS_STEPS.includes(site.step)) {
        const pass = { type: 'pass', entity: site.entity.id, entity_type: site.entity.type };
        applied = await env.bridge('apply', { game: id, action: pass, last_id: site.last_id });
        if (applied && applied.ok) {
          return played(
            { kind: 'implicit_pass', actor: label, description: `Pass (${site.step})`, seat },
            applied.action,
          );
        }
        if (applied && applied.dry_run) {
          state.dryRunAt = site.last_id;
          setPhase('waiting', `Dry run — would pass for ${label} (${site.step})`);
          return idleMs();
        }
      }
      return fail(`The site refused the model's move "${result.actor}: ${result.description}": ${refusal}`);
    }

    async function waitForJob(job, id, label) {
      let current = job;
      while (current && current.status === 'running') {
        setPhase('thinking', `${label} is thinking…`, current.progress || 0);
        await sleep(JOB_POLL_MS);
        if (!active(id)) return null;
        const response = await env.send({ type: 'move_status', job: current.job });
        if (!active(id)) return null;
        if (!response || !response.ok) return { status: 'error', error: (response && response.error) || 'lost the search' };
        current = response.data;
      }
      return current;
    }

    function played(result, stored) {
      state.passes = result.kind === 'implicit_pass' ? state.passes + 1 : 0;
      state.moves.unshift({
        id: stored && stored.id,
        kind: result.kind,
        actor: result.actor,
        description: result.description,
        visits: result.visits || 0,
        share: result.share,
        seat: result.seat,
      });
      state.moves.length = Math.min(state.moves.length, KEEP_MOVES);
      if (state.passes > MAX_PASSES_IN_A_ROW) {
        return fail(`Autoplay stopped: ${state.passes} passes in a row the engine didn't ask for — the engines may disagree.`);
      }
      setPhase('playing', `${result.actor}: ${result.description}`);
      if (env.onPlayed) env.onPlayed(result);
      return 0;
    }

    /** What the panel shows. */
    function view() {
      return {
        enabled: state.enabled,
        paused: state.paused,
        live: live(),
        dryRun: state.dryRun,
        seats: state.seats.slice(),
        readouts: state.readouts,
        phase: state.phase,
        text: state.text,
        progress: state.progress,
        moves: state.moves.slice(),
        players: state.players.slice(),
      };
    }

    function stop() {
      clearTimer();
      state.gameId = null;
      state.enabled = false;
    }

    return { state, setGame, configure, setPaused, poke, tick, view, stop };
  }

  const exported = { createAutoplay, lastActionId, IMPLICIT_PASS_STEPS };
  root.RL18XX = Object.assign(root.RL18XX || {}, exported);
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof globalThis !== 'undefined' ? globalThis : this);
