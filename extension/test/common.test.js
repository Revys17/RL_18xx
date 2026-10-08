// Unit tests for src/common.js (run: node --test extension/test/).
const test = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/common.js');
const { fakeDocument } = require('./fakes.js');

test('the game id comes from a /game/<id> path only', () => {
  assert.equal(R.parseGameId('/game/123'), 123);
  assert.equal(R.parseGameId('/game/123/'), 123);
  assert.equal(R.parseGameId('/game/123?action=40'), 123);
  assert.equal(R.parseGameId('/game/abc'), null);
  assert.equal(R.parseGameId('/hotseat/123'), null);
  assert.equal(R.parseGameId('/'), null);
  assert.equal(R.parseGameId(undefined), null);
});

test('chat and user settings stay in the page', () => {
  const game = {
    id: 5,
    user_settings: { notepad: 'secret' },
    actions: [{ type: 'bid', id: 1 }, { type: 'message', message: 'hi', id: 2 }, { type: 'pass', id: 3 }],
  };
  const stripped = R.stripGame(game);
  assert.deepEqual(stripped.actions.map((a) => a.type), ['bid', 'pass']);
  assert.equal('user_settings' in stripped, false);
  assert.equal(game.actions.length, 3, 'the input is not modified');
});

test('a backend URL is http://<host>:<port> with no path', () => {
  assert.equal(R.normalizeBackendUrl('http://127.0.0.1:5002'), 'http://127.0.0.1:5002');
  assert.equal(R.normalizeBackendUrl(' http://localhost:6000/ '), 'http://localhost:6000');
  assert.equal(R.normalizeBackendUrl('http://192.168.0.186:5002'), 'http://192.168.0.186:5002');
  assert.equal(R.normalizeBackendUrl('http://workstation:5002/'), 'http://workstation:5002');
  assert.equal(R.normalizeBackendUrl('https://127.0.0.1:5002'), null);
  assert.equal(R.normalizeBackendUrl('http://user:pw@192.168.0.186:5002'), null);
  assert.equal(R.normalizeBackendUrl('http://127.0.0.1:5002/api'), null);
  assert.equal(R.normalizeBackendUrl('not a url'), null);
});

test('only this machine is a loopback backend', () => {
  assert.equal(R.isLoopbackBackend('http://127.0.0.1:5002'), true);
  assert.equal(R.isLoopbackBackend('http://localhost:5002'), true);
  assert.equal(R.isLoopbackBackend('http://192.168.0.186:5002'), false);
});

test('the fetch gate spaces fetches 10 s apart', () => {
  let now = 50_000;
  const gate = new R.FetchGate(10_000, () => now);
  assert.equal(gate.waitMs(), 0);
  gate.mark();
  now += 4_000;
  assert.equal(gate.waitMs(), 6_000);
  now += 6_000;
  assert.equal(gate.waitMs(), 0);
});

test('the log signature changes with a new log line and is null without a log', () => {
  const doc = fakeDocument({ log: ['-- Stock Round 1 --', 'Player 1 pars PRR at $67'] });
  const before = R.logSignature(doc);
  assert.equal(before, '2|Player 1 pars PRR at $67');
  doc.log.push('Player 2 passes');
  assert.notEqual(R.logSignature(doc), before);
  assert.equal(R.logSignature(fakeDocument({ log: null })), null);
});

test('a hex is found on the map by its translate position', () => {
  assert.deepEqual(R.parseTranslate('translate(1918.65, 850) rotate(30)'), [1918.65, 850]);
  assert.deepEqual(R.parseTranslate('translate(10 20)'), [10, 20]);
  assert.equal(R.parseTranslate('rotate(30)'), null);
  const doc = fakeDocument({ hexes: [[186.6, 250], [1918.65, 850]] });
  assert.equal(R.findHexGroup(doc, [1918.65, 850]), doc.hexGroups[1]);
  assert.equal(R.findHexGroup(doc, [1918.6, 850.4]), doc.hexGroups[1], 'within a pixel');
  assert.equal(R.findHexGroup(doc, [500, 500]), null);
  assert.equal(R.findHexGroup(fakeDocument({ hexes: null }), [186.6, 250]), null, 'no map on screen');
});

test('percentages and escaping', () => {
  assert.equal(R.percent(0.7128), '71%');
  assert.equal(R.percent(0.0186), '1.9%');
  assert.equal(R.percent(null), '–');
  assert.equal(R.escapeHtml('<b a="1">&\''), '&lt;b a=&quot;1&quot;&gt;&amp;&#39;');
});
