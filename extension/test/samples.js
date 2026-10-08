// A backend answer (POST /api/advise) in the shape rl18xx/agent/advisor/app.py returns.

function sampleAdvice() {
  return {
    supported: true,
    started: true,
    game_id: 123,
    position: 'abc',
    players: [1, 2, 3, 4].map((seat) => ({ seat, id: 100 + seat, name: `Player ${seat}` })),
    untested_player_count: false,
    engine_error: null,
    finished: false,
    acting: { kind: 'corporation', id: 'B&M', name: 'B&M (Boston & Maine Railroad)', player: 'Player 3', player_id: 3 },
    round: { type: 'Operating', step: 'LayTile' },
    policy: { role: 'policy', checkpoint: 'pg4_20261007/learner/600', temperature: 1 },
    warnings: [],
    num_legal: 9,
    forced: false,
    win: {
      final: false,
      players: [
        { seat: 1, name: 'Player 1', probability: 0.05 },
        { seat: 2, name: 'Player 2', probability: 0.2 },
        { seat: 3, name: 'Player 3', probability: 0.613 },
        { seat: 4, name: 'Player 4', probability: 0.137 },
      ],
    },
    moves: [
      {
        index: 6032,
        probability: 0.7128,
        type: 'lay_tile',
        actor: 'B&M (Player 3)',
        description: 'Lay tile #57 on F22 (Providence), rotation 1',
        hex: 'F22',
        map: [1918.65, 850.0],
        tile: '57',
        rotation: 1,
        price: null,
      },
      {
        index: 4,
        probability: 0.2188,
        type: 'bid',
        actor: 'Player 3',
        description: 'Bid $115 on Mohawk & Hudson (MH)',
        price: {
          fixed: false,
          price: 115,
          range: [115, 600],
          options: [
            { price: 115, low: 115, high: 115, probability: 0.9914 },
            { price: 120, low: 120, high: 120, probability: 0.0031 },
          ],
        },
      },
    ],
    recent: [
      {
        action: 267,
        type: 'pass',
        actor: 'PRR (Player 1)',
        description: 'Pass: buy no train',
        probability: 0.953,
        rank: 1,
        num_legal: 2,
        label: 'expected',
      },
    ],
  };
}

module.exports = { sampleAdvice };
