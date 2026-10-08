// Background worker (Chrome service worker / Firefox background script): the
// only part of the extension that talks to the local advisor backend. Content
// scripts on an https page can't reliably reach http://127.0.0.1 (mixed content,
// CORS, Private Network Access); this worker can, with its host permission. It
// takes messages from the extension's own content script and options page:
//   {type: 'advise', game}                      -> POST /api/advise
//   {type: 'think', game_id, position, readouts} -> POST /api/think
//   {type: 'think_status', job}                 -> POST /api/think/<job>
//   {type: 'health'}                            -> POST /api/health
// and answers {ok: true, data} or {ok: false, error, status?, unreachable?, data?}.
(function (root) {
  'use strict';

  if (!root.RL18XX && typeof importScripts === 'function') importScripts('common.js'); // Chrome's service worker
  const R = root.RL18XX || require('./common.js');

  async function backendUrl(storage) {
    const stored = storage ? await storage.get({ backendUrl: R.DEFAULTS.backendUrl }) : {};
    return R.normalizeBackendUrl(stored.backendUrl) || R.DEFAULTS.backendUrl;
  }

  /** POST ``body`` to the backend's ``path``. ``deps``: ``fetch`` and ``storage``. */
  async function call(path, body, deps) {
    const base = await backendUrl(deps.storage);
    let response;
    try {
      response = await deps.fetch(base + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
        credentials: 'omit',
      });
    } catch (e) {
      return {
        ok: false,
        unreachable: true,
        error: `can't reach the advisor backend at ${base} -- is "python main.py advisor" running?`,
      };
    }
    let data = null;
    try {
      data = await response.json();
    } catch (e) {
      data = null;
    }
    if (!response.ok) {
      return { ok: false, status: response.status, error: (data && data.error) || `HTTP ${response.status}`, data };
    }
    return { ok: true, data };
  }

  async function handleMessage(message, deps) {
    switch (message && message.type) {
      case 'advise':
        return call('/api/advise', message.game, deps);
      case 'think':
        return call(
          '/api/think',
          { game_id: message.game_id, position: message.position, readouts: message.readouts },
          deps,
        );
      case 'think_status':
        return call(`/api/think/${encodeURIComponent(message.job)}`, {}, deps);
      case 'health':
        return call('/api/health', {}, deps);
      default:
        return { ok: false, error: `unknown message ${message && message.type}` };
    }
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { handleMessage, call };
    return;
  }
  const api = root.browser || root.chrome;
  const deps = { fetch: (...args) => root.fetch(...args), storage: api.storage.local };
  api.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (sender && sender.id && sender.id !== api.runtime.id) return false; // only our own pages and scripts
    handleMessage(message, deps).then(sendResponse, (e) => sendResponse({ ok: false, error: String(e) }));
    return true; // the answer comes asynchronously
  });
})(typeof globalThis !== 'undefined' ? globalThis : this);
