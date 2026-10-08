// Content script for 18xx.games game pages (/game/<id>).
//
// It reads the open game from the site itself -- GET /api/game/<id> on the same
// origin, without cookies, the request the site's own page makes -- and never
// more than once every 10 s: when the game opens, when the site's game log shows
// a new line (seen by a debounced MutationObserver, no network), and when the
// panel's refresh button is pressed; a hidden tab waits until it is shown. The
// game goes to the background worker, which asks the local advisor backend; the
// answer is drawn in the panel (panel.js), and the recommended hex is outlined
// on the site's map when it can be found. The site is a single-page app, so the
// page's path is watched locally (no network) for a change of game.
(function (root) {
  'use strict';

  const R =
    root.RL18XX && root.RL18XX.Panel
      ? root.RL18XX
      : Object.assign({}, require('./common.js'), require('./panel.js'));
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const THINK_POLL_MS = 1000;

  /** The page logic, with its browser surroundings passed in (``env``) so the unit
   * tests can drive it: ``location``, ``document``, ``fetchGame(id)``,
   * ``send(message)`` (to the background worker), ``makePanel(callbacks, placement)``,
   * ``now()``, ``setTimeout`` / ``clearTimeout``, ``options`` and ``saveOptions(partial)``. */
  function createController(env) {
    const gate = new R.FetchGate(R.MIN_FETCH_INTERVAL_MS, env.now);
    const state = {
      options: env.options,
      gameId: null,
      panel: null,
      advice: null,
      statuses: [],
      think: { state: 'idle' },
      signature: null,
      fetching: false,
      again: false, // fetch once more when the current fetch ends
      waitingForVisible: false,
      timer: null,
      mutationTimer: null,
      thinkTimer: null,
      highlight: null,
    };

    function render() {
      if (!state.panel) return;
      state.panel.update({
        statuses: state.statuses,
        advice: state.advice,
        think: state.think,
        readouts: state.options.readouts,
        busy: state.fetching,
      });
    }

    function setStatus(kind, text) {
      state.statuses = text ? [{ kind, text }] : [];
      render();
    }

    // ------------------------------------------------------------- the game
    function teardown() {
      for (const timer of [state.timer, state.mutationTimer, state.thinkTimer]) if (timer) env.clearTimeout(timer);
      state.timer = state.mutationTimer = state.thinkTimer = null;
      clearHighlight();
      if (state.panel) state.panel.destroy();
      Object.assign(state, {
        panel: null,
        advice: null,
        statuses: [],
        think: { state: 'idle' },
        signature: null,
        again: false,
        waitingForVisible: false,
      });
    }

    /** Follow the game in the page's path (call when the path may have changed). */
    function checkLocation() {
      const id = state.options.enabled ? R.parseGameId(env.location.pathname) : null;
      if (id === state.gameId) return;
      teardown();
      state.gameId = id;
      if (id === null) return;
      state.panel = env.makePanel(
        {
          onRefresh: refresh,
          onThink: think,
          onHoverMove: (move) => highlightMove(move || topMapMove()),
          onPlace: (place) => savePanel(place),
          onCollapse: (collapsed) => savePanel({ collapsed }),
        },
        state.options.panel,
      );
      setStatus('info', 'Reading the game…');
      state.signature = R.logSignature(env.document);
      requestFetch();
    }

    function savePanel(partial) {
      state.options.panel = Object.assign({}, state.options.panel, partial);
      if (env.saveOptions) env.saveOptions({ panel: state.options.panel });
    }

    /** Fetch the game now, or as soon as the 10 s spacing (and a hidden tab) allows. */
    function requestFetch() {
      if (state.gameId === null) return;
      if (env.document.hidden) {
        state.waitingForVisible = true;
        return;
      }
      if (state.fetching) {
        state.again = true;
        return;
      }
      const wait = gate.waitMs();
      if (wait > 0) {
        if (!state.timer) {
          state.timer = env.setTimeout(() => {
            state.timer = null;
            requestFetch();
          }, wait);
        }
        return;
      }
      fetchAndAdvise();
    }

    async function fetchAndAdvise() {
      const id = state.gameId;
      gate.mark();
      state.fetching = true;
      state.signature = R.logSignature(env.document);
      render();
      try {
        let game;
        try {
          game = await env.fetchGame(id);
        } catch (e) {
          if (id === state.gameId) setStatus('error', `Couldn't read the game from the site: ${e.message || e}`);
          return;
        }
        if (id !== state.gameId) return;
        const response = await env.send({ type: 'advise', game: R.stripGame(game) });
        if (id !== state.gameId) return;
        showAdvice(response);
      } catch (e) {
        if (id === state.gameId) setStatus('error', `The advisor failed: ${e.message || e}`);
      } finally {
        state.fetching = false;
        render();
        if (state.again) {
          // Asked again meanwhile (for this game or, after a switch, the new one).
          state.again = false;
          requestFetch();
        }
      }
    }

    function showAdvice(response) {
      if (!response || !response.ok) {
        const text = (response && response.error) || 'No answer from the background worker';
        state.statuses = [{ kind: 'error', text: response && response.unreachable ? `Backend unreachable: ${text}` : text }];
        render();
        return;
      }
      const advice = response.data;
      state.advice = advice;
      if (!advice.supported) state.statuses = [{ kind: 'warn', text: `Unsupported game: ${advice.reason}` }];
      else if (advice.started === false) state.statuses = [{ kind: 'info', text: advice.reason || 'Not started yet.' }];
      else state.statuses = [];
      render();
      highlightMove(topMapMove());
    }

    function refresh() {
      const wait = gate.waitMs();
      if (wait > 0 && !state.fetching) {
        state.statuses = [{ kind: 'info', text: `Refreshing in ${Math.ceil(wait / 1000)} s (at most once every 10 s).` }];
        render();
      }
      requestFetch();
    }

    /** The page changed: look for a new line in the game log once it settles. */
    function onMutations() {
      if (state.gameId === null) return;
      if (state.mutationTimer) env.clearTimeout(state.mutationTimer);
      state.mutationTimer = env.setTimeout(() => {
        state.mutationTimer = null;
        const signature = R.logSignature(env.document);
        if (signature !== null && signature !== state.signature) {
          state.signature = signature;
          requestFetch();
        }
        ensureHighlight();
      }, R.MUTATION_DEBOUNCE_MS);
    }

    function onVisibility() {
      if (!env.document.hidden && state.waitingForVisible) {
        state.waitingForVisible = false;
        requestFetch();
      }
    }

    function setOptions(options) {
      const wasEnabled = state.options.enabled;
      state.options = options;
      if (wasEnabled !== options.enabled) checkLocation();
      else if (state.panel) {
        state.panel.setPlacement(options.panel);
        render();
      }
    }

    // ----------------------------------------------------------------- think
    async function think() {
      const advice = state.advice;
      if (!advice || !advice.supported || state.think.state === 'running') return;
      const id = state.gameId;
      state.think = { state: 'running', progress: 0, position: advice.position };
      render();
      let response;
      try {
        response = await env.send({
          type: 'think',
          game_id: advice.game_id,
          position: advice.position,
          readouts: state.options.readouts,
        });
      } catch (e) {
        response = { ok: false, error: `The advisor failed: ${e.message || e}` };
      }
      if (id !== state.gameId) return;
      const busyJob = response && !response.ok && response.data && response.data.job;
      if (response && response.ok) pollThink(response.data.job, id);
      else if (busyJob) pollThink(busyJob.job, id);
      else {
        state.think = { state: 'error', error: (response && response.error) || 'The search did not start' };
        render();
      }
    }

    function pollThink(job, id) {
      state.thinkTimer = env.setTimeout(async () => {
        state.thinkTimer = null;
        let response;
        try {
          response = await env.send({ type: 'think_status', job });
        } catch (e) {
          response = { ok: false, error: `The advisor failed: ${e.message || e}` };
        }
        if (id !== state.gameId) return;
        if (!response || !response.ok) {
          state.think = { state: 'error', error: (response && response.error) || 'Lost the search' };
        } else if (response.data.status === 'running') {
          state.think = { state: 'running', progress: response.data.progress, position: response.data.position };
          pollThink(job, id);
        } else if (response.data.status === 'done') {
          state.think = { state: 'done', result: response.data.result, position: response.data.position };
        } else {
          state.think = { state: 'error', error: response.data.error || 'The search failed' };
        }
        render();
      }, THINK_POLL_MS);
    }

    // ------------------------------------------------------------- highlight
    function topMapMove() {
      const moves = (state.advice && state.advice.moves) || [];
      return moves.length && moves[0].map ? moves[0] : null;
    }

    function clearHighlight() {
      if (state.highlight && state.highlight.element && state.highlight.element.remove) state.highlight.element.remove();
      state.highlight = null;
    }

    /** Outline ``move``'s hex on the site's map; silently nothing when it isn't there. */
    function highlightMove(move) {
      clearHighlight();
      if (!move || !move.map) return;
      try {
        const group = R.findHexGroup(env.document, move.map);
        if (!group) {
          state.highlight = { move, element: null };
          return;
        }
        const outline = env.document.createElementNS(SVG_NS, 'polygon');
        outline.setAttribute('points', R.HEX_POINTS);
        outline.setAttribute('fill', 'none');
        outline.setAttribute('stroke', '#e11d48');
        outline.setAttribute('stroke-width', '14');
        outline.setAttribute('stroke-dasharray', '24 12');
        outline.setAttribute('pointer-events', 'none');
        outline.setAttribute('class', 'rl18xx-highlight');
        group.appendChild(outline);
        state.highlight = { move, element: outline };
      } catch (e) {
        state.highlight = null; // the map isn't what we expect: no highlight
      }
    }

    /** The site redraws its map: put the outline back if it was dropped. */
    function ensureHighlight() {
      const current = state.highlight;
      if (current && (!current.element || current.element.isConnected === false)) highlightMove(current.move);
    }

    return { state, checkLocation, requestFetch, refresh, onMutations, onVisibility, setOptions, think, highlightMove };
  }

  function mergeOptions(stored) {
    const defaults = R.DEFAULTS;
    return Object.assign({}, defaults, stored, { panel: Object.assign({}, defaults.panel, (stored || {}).panel) });
  }

  async function bootstrap() {
    const api = root.browser || root.chrome;
    const options = mergeOptions(await api.storage.local.get(null));
    const controller = createController({
      location: root.location,
      document: root.document,
      fetchGame: async (id) => {
        const url = new URL(`/api/game/${encodeURIComponent(id)}`, root.location.origin).href;
        const response = await fetch(url, { credentials: 'omit', headers: { Accept: 'application/json' } });
        if (!response.ok) throw new Error(`the site answered HTTP ${response.status}`);
        return response.json();
      },
      send: (message) => api.runtime.sendMessage(message),
      makePanel: (callbacks, placement) => new R.Panel(root.document, callbacks, placement),
      now: () => Date.now(),
      setTimeout: (fn, ms) => root.setTimeout(fn, ms),
      clearTimeout: (timer) => root.clearTimeout(timer),
      options,
      saveOptions: (partial) => api.storage.local.set(partial),
    });
    new root.MutationObserver(() => controller.onMutations()).observe(root.document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    root.setInterval(() => controller.checkLocation(), 1000);
    root.addEventListener('popstate', () => controller.checkLocation());
    root.document.addEventListener('visibilitychange', () => controller.onVisibility());
    api.storage.onChanged.addListener((changes, area) => {
      if (area !== 'local') return;
      api.storage.local.get(null).then((stored) => controller.setOptions(mergeOptions(stored)));
    });
    controller.checkLocation();
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { createController, mergeOptions };
  else bootstrap();
})(typeof globalThis !== 'undefined' ? globalThis : this);
