// Unit tests for the background worker's calls to the local backend (src/background.js).
const test = require('node:test');
const assert = require('node:assert/strict');
const { handleMessage } = require('../src/background.js');

function deps({ stored = {}, respond } = {}) {
  const calls = [];
  return {
    calls,
    storage: { get: async (defaults) => Object.assign({}, defaults, stored) },
    fetch: async (url, init) => {
      calls.push({ url, init });
      return respond(url, init);
    },
  };
}

const json = (status, body) => ({ ok: status < 400, status, json: async () => body });

test('advice requests go to the default backend as a JSON POST', async () => {
  const d = deps({ respond: () => json(200, { supported: true }) });
  const answer = await handleMessage({ type: 'advise', game: { id: 1, actions: [] } }, d);
  assert.deepEqual(answer, { ok: true, data: { supported: true } });
  assert.equal(d.calls[0].url, 'http://127.0.0.1:5002/api/advise');
  assert.equal(d.calls[0].init.method, 'POST');
  assert.equal(d.calls[0].init.credentials, 'omit');
  assert.deepEqual(JSON.parse(d.calls[0].init.body), { id: 1, actions: [] });
});

test('the configured backend is used only when it is on this machine', async () => {
  const local = deps({ stored: { backendUrl: 'http://localhost:6001' }, respond: () => json(200, {}) });
  await handleMessage({ type: 'health' }, local);
  assert.equal(local.calls[0].url, 'http://localhost:6001/api/health');
  const remote = deps({ stored: { backendUrl: 'https://example.com' }, respond: () => json(200, {}) });
  await handleMessage({ type: 'health' }, remote);
  assert.equal(remote.calls[0].url, 'http://127.0.0.1:5002/api/health');
});

test('think and its status', async () => {
  const d = deps({ respond: (url) => json(200, { job: 'j1', url }) });
  await handleMessage({ type: 'think', game_id: 5, position: 'p', readouts: 64, extra: 'dropped' }, d);
  assert.deepEqual(JSON.parse(d.calls[0].init.body), { game_id: 5, position: 'p', readouts: 64 });
  await handleMessage({ type: 'think_status', job: 'a/b' }, d);
  assert.equal(d.calls[1].url, 'http://127.0.0.1:5002/api/think/a%2Fb');
});

test('an unreachable backend and an error answer', async () => {
  const down = deps({ respond: () => { throw new TypeError('Failed to fetch'); } });
  const answer = await handleMessage({ type: 'advise', game: {} }, down);
  assert.equal(answer.ok, false);
  assert.equal(answer.unreachable, true);
  assert.match(answer.error, /python main.py advisor/);

  const busy = deps({ respond: () => json(409, { error: 'A search is already running', job: { job: 'j0' } }) });
  const refused = await handleMessage({ type: 'think', game_id: 5 }, busy);
  assert.deepEqual(refused, {
    ok: false,
    status: 409,
    error: 'A search is already running',
    data: { error: 'A search is already running', job: { job: 'j0' } },
  });
  assert.equal((await handleMessage({ type: 'nope' }, busy)).ok, false);
});
