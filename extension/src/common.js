// Shared helpers of the RL18xx 1830 advisor extension: the content script, the
// background worker, the options page and the backend's debug page all load
// this file. A classic script: it defines globalThis.RL18XX (and, under node,
// module.exports for the unit tests in extension/test/).
(function (root) {
  'use strict';

  const DEFAULTS = {
    enabled: true,
    backendUrl: 'http://127.0.0.1:5002',
    readouts: 200,
    // Panel placement: a corner, or a dragged-to position (left/top in px).
    panel: { corner: 'top-right', left: null, top: null, collapsed: false },
  };
  // Never fetch a game from the site more often than this.
  const MIN_FETCH_INTERVAL_MS = 10000;
  // Wait for the page to settle after DOM changes before looking for a new action.
  const MUTATION_DEBOUNCE_MS = 1500;
  // The hex outline of the 18xx.games map (upstream assets/app/lib/hex.rb POINTS).
  const HEX_POINTS = '100,0 50,87 -50,87 -100,0 -50,-87 50,-87';

  /** The game id of an 18xx.games game page path (/game/<id>), or null. */
  function parseGameId(pathname) {
    const match = /^\/game\/(\d+)(?:[/?#]|$)/.exec(pathname || '');
    return match ? Number(match[1]) : null;
  }

  /** The game as the backend needs it: chat and per-user settings stay in the page
   * (the backend's importer skips chat messages anyway). */
  function stripGame(game) {
    const out = {};
    for (const [key, value] of Object.entries(game || {})) {
      if (key !== 'user_settings') out[key] = value;
    }
    out.actions = (Array.isArray(game && game.actions) ? game.actions : []).filter(
      (action) => action && action.type !== 'message',
    );
    return out;
  }

  /** ``http://127.0.0.1:<port>`` or ``http://localhost:<port>`` (no path), or null:
   * the extension only talks to a backend on this machine. */
  function normalizeBackendUrl(url) {
    let parsed;
    try {
      parsed = new URL(String(url || '').trim());
    } catch (e) {
      return null;
    }
    if (parsed.protocol !== 'http:') return null;
    if (!['127.0.0.1', 'localhost'].includes(parsed.hostname)) return null;
    if ((parsed.pathname && parsed.pathname !== '/') || parsed.search || parsed.hash) return null;
    return `http://${parsed.host}`;
  }

  /** Spaces fetches at least ``minInterval`` ms apart. */
  class FetchGate {
    constructor(minInterval = MIN_FETCH_INTERVAL_MS, now = () => Date.now()) {
      this.minInterval = minInterval;
      this.now = now;
      this.last = -Infinity;
    }

    /** Milliseconds until the next fetch is allowed (0: now). */
    waitMs() {
      return Math.max(0, this.last + this.minInterval - this.now());
    }

    mark() {
      this.last = this.now();
    }
  }

  /** A cheap fingerprint of the site's game log (``#chatlog``'s ``.chatline``s: their
   * count and the last one), which changes when an action (or a chat line) lands;
   * null when the log isn't on screen. */
  function logSignature(doc) {
    const log = doc && doc.querySelector ? doc.querySelector('#chatlog') : null;
    if (!log) return null;
    const lines = log.querySelectorAll('.chatline');
    const last = lines.length ? lines[lines.length - 1].textContent || '' : '';
    return `${lines.length}|${last}`;
  }

  /** ``[x, y]`` of an SVG ``translate(x, y)`` / ``translate(x y)`` transform. */
  function parseTranslate(transform) {
    const match = /translate\(\s*(-?[\d.]+)(?:\s*,\s*|\s+)(-?[\d.]+)\s*\)/.exec(transform || '');
    return match ? [parseFloat(match[1]), parseFloat(match[2])] : null;
  }

  /** The ``<g>`` of the hex the map draws at ``xy`` (the backend's ``map`` position:
   * each hex is a ``<g transform="translate(x, y) ...">`` under ``#map-hexes``), or null. */
  function findHexGroup(doc, xy, tolerance = 1) {
    if (!doc || !xy) return null;
    const layer = doc.querySelector('#map-hexes');
    if (!layer) return null;
    for (const group of Array.from(layer.children || [])) {
      if (String(group.tagName || '').toLowerCase() !== 'g') continue;
      const at = parseTranslate(group.getAttribute('transform'));
      if (at && Math.abs(at[0] - xy[0]) <= tolerance && Math.abs(at[1] - xy[1]) <= tolerance) return group;
    }
    return null;
  }

  function percent(p) {
    if (p === null || p === undefined || Number.isNaN(p)) return '–';
    const value = 100 * p;
    return value >= 9.95 ? `${Math.round(value)}%` : `${value.toFixed(1)}%`;
  }

  function escapeHtml(text) {
    return String(text === null || text === undefined ? '' : text).replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  }

  const RL18XX = {
    DEFAULTS,
    MIN_FETCH_INTERVAL_MS,
    MUTATION_DEBOUNCE_MS,
    HEX_POINTS,
    parseGameId,
    stripGame,
    normalizeBackendUrl,
    FetchGate,
    logSignature,
    parseTranslate,
    findHexGroup,
    percent,
    escapeHtml,
  };
  root.RL18XX = Object.assign(root.RL18XX || {}, RL18XX);
  if (typeof module !== 'undefined' && module.exports) module.exports = RL18XX;
})(typeof globalThis !== 'undefined' ? globalThis : this);
