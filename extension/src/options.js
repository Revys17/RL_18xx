// Options page: backend URL, on/off, "think harder" readouts and the panel's corner.
(function (root) {
  'use strict';

  const R = root.RL18XX;
  const api = root.browser || root.chrome;
  const $ = (id) => root.document.getElementById(id);

  function show(text, ok) {
    $('status').textContent = text;
    $('status').className = ok ? 'ok' : 'bad';
  }

  async function load() {
    const stored = await api.storage.local.get(null);
    const options = Object.assign({}, R.DEFAULTS, stored);
    const panel = Object.assign({}, R.DEFAULTS.panel, stored.panel);
    $('enabled').checked = options.enabled !== false;
    $('backendUrl').value = options.backendUrl;
    $('readouts').value = options.readouts;
    $('corner').value = panel.corner;
  }

  async function save() {
    const backendUrl = R.normalizeBackendUrl($('backendUrl').value);
    if (!backendUrl) {
      show('The backend URL must be http://127.0.0.1:<port> or http://localhost:<port>.', false);
      return false;
    }
    const readouts = Math.max(16, Math.min(5000, parseInt($('readouts').value, 10) || R.DEFAULTS.readouts));
    const stored = await api.storage.local.get('panel');
    const panel = Object.assign({}, R.DEFAULTS.panel, stored.panel);
    if (panel.corner !== $('corner').value) Object.assign(panel, { corner: $('corner').value, left: null, top: null });
    await api.storage.local.set({ enabled: $('enabled').checked, backendUrl, readouts, panel });
    $('backendUrl').value = backendUrl;
    show('Saved.', true);
    return true;
  }

  async function test() {
    if (!(await save())) return;
    const response = await api.runtime.sendMessage({ type: 'health' });
    if (response && response.ok) {
      const checkpoints = response.data.checkpoints || {};
      show(`Backend up. Policy ${checkpoints.policy}, value ${checkpoints.value}.`, true);
    } else {
      show(`No backend: ${(response && response.error) || 'no answer'}`, false);
    }
  }

  async function reset() {
    const stored = await api.storage.local.get('panel');
    const panel = Object.assign({}, R.DEFAULTS.panel, stored.panel, { corner: $('corner').value, left: null, top: null });
    await api.storage.local.set({ panel });
    show('The panel is back in its corner.', true);
  }

  $('save').addEventListener('click', save);
  $('reset').addEventListener('click', reset);
  $('test').addEventListener('click', test);
  load();
})(globalThis);
