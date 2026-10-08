// Unit tests for the panel's rendering (src/panel.js).
const test = require('node:test');
const assert = require('node:assert/strict');
const { renderBody, summary, priceText } = require('../src/panel.js');
const { sampleAdvice } = require('./samples.js');

test('the panel shows win chances, the mover, the moves with prices, and the last moves', () => {
  const html = renderBody({ statuses: [], advice: sampleAdvice(), think: { state: 'idle' }, readouts: 200 });
  for (const text of ['Player 1', 'Player 3', '61%', 'B&amp;M (Boston &amp; Maine Railroad)', 'to move']) {
    assert.ok(html.includes(text), `missing ${text}`);
  }
  assert.ok(html.includes('Lay tile #57 on F22 (Providence), rotation 1'));
  assert.ok(html.includes('71%'));
  assert.ok(html.includes('price $115 (99%); also $120 (0.3%)'));
  assert.ok(html.includes('Think harder (200 readouts)'));
  assert.ok(html.includes('Pass: buy no train') && html.includes('expected'));
});

test('player names and descriptions are escaped', () => {
  const advice = sampleAdvice();
  advice.win.players[0].name = '<img src=x onerror=alert(1)>';
  advice.moves[0].description = '<script>alert(1)</script>';
  const html = renderBody({ statuses: [], advice });
  assert.equal(html.includes('<img'), false);
  assert.equal(html.includes('<script>'), false);
  assert.ok(html.includes('&lt;img src=x'));
});

test('status lines: backend unreachable, unsupported game, untested player count', () => {
  const unreachable = renderBody({ statuses: [{ kind: 'error', text: 'Backend unreachable: no answer' }], advice: null });
  assert.ok(unreachable.includes('status error') && unreachable.includes('Backend unreachable'));

  const unsupported = { supported: false, reason: '1867 isn\'t supported' };
  const html = renderBody({ statuses: [{ kind: 'warn', text: `Unsupported game: ${unsupported.reason}` }], advice: unsupported });
  assert.ok(html.includes('Unsupported game'));
  assert.equal(html.includes('Win chances'), false, 'nothing else for an unsupported game');

  const advice = sampleAdvice();
  advice.warnings = ['Untested: the model was trained on 4-player games; this one has 3 players.'];
  assert.ok(renderBody({ statuses: [], advice }).includes('status warn">Untested'));
});

test('think: progress while running, the search moves when done', () => {
  const running = renderBody({ advice: sampleAdvice(), think: { state: 'running', progress: 0.25 } });
  assert.ok(running.includes('Thinking… 25%') && running.includes('disabled'));
  const result = { visits: 201, moves: [{ share: 0.4, visits: 80, description: 'Lay tile #9 on D22, rotation 1', actor: 'B&M (Player 3)' }], win: [{ name: 'Player 1', probability: 0.5 }] };
  const done = renderBody({ advice: sampleAdvice(), think: { state: 'done', result, position: 'abc' } });
  assert.ok(done.includes('Search: 201 visits') && done.includes('80 visits') && done.includes('Lay tile #9 on D22'));
  const stale = renderBody({ advice: sampleAdvice(), think: { state: 'done', result, position: 'older' } });
  assert.ok(stale.includes('an earlier position'));
});

test('the collapsed summary names the top move', () => {
  assert.equal(summary({ statuses: [], advice: sampleAdvice() }), '71% Lay tile #57 on F22 (Providence), rotation 1');
  assert.equal(summary({ statuses: [{ kind: 'error', text: 'Backend unreachable' }], advice: null }), 'Backend unreachable');
});

test('a fixed price shows no alternatives', () => {
  assert.equal(priceText({ fixed: true, price: 80, options: [] }), '');
  assert.equal(priceText(null), '');
  assert.equal(
    priceText({ fixed: false, price: 50, options: [{ price: 50, low: 40, high: 60, probability: 0.5 }] }),
    'price $40–60 (50%)',
  );
});
