# Per-action state dump of a recorded 18xx.games game in the REAL Ruby
# engine (tobymao/18xx), for Rust-vs-Ruby divergence hunting
# (scripts/g1867_ruby_diff.py drives it inside the ruby:3.2-slim image).
#
# usage: ruby g1867_ruby_dump.rb <game.json> <out.jsonl>
#
# One JSON line per processed FILE action id (Ruby's own undo/redo
# filtering, messages skipped), via Game#process_to_action — so lines are
# keyed by the ids in the file, never by a renumbered index. The last line
# is {"final": true, ...} with Ruby's result and game_end_reason; a line
# {"id": N, "exception": ...} marks where Ruby itself rejects the record.
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'

data = JSON.parse(File.read(ARGV[0]))
out = File.open(ARGV[1], 'w')
actions = data['actions']
filtered, = Engine::Game::Base.filtered_actions(actions)
ids = actions.each_with_index.select { |a, i| filtered[i] && a['type'] != 'message' }.map { |a, _| a['id'] }
seat = {}
data['players'].each_with_index { |p, i| seat[p['id']] = i + 1 }

g = Engine::Game.load(data, at_action: 0)

def snap(g, id, seat)
  step = g.round.active_step rescue nil
  ent = (step&.current_entity rescue nil) || (g.current_entity rescue nil)
  corps = {}
  g.corporations.each do |c|
    corps[c.name] = {
      'p' => c.share_price&.price,
      'cash' => c.cash,
      'loans' => c.loans.size,
      'fl' => c.floated?,
      'tr' => c.trains.map(&:name),
    }
  end
  {
    'id' => id,
    'round' => g.round.class.name.split('::').last,
    'step' => step&.class&.name&.split('::')&.last,
    'ent' => ent&.name,
    'phase' => g.phase.name,
    'corps' => corps,
    # keyed by SEAT (1-based recorded order) — no player names
    'pcash' => g.players.to_h { |p| [seat[p.id] || p.id, p.cash] },
    'finished' => g.finished,
    'bank' => g.bank.cash,
  }
end

ids.each do |id|
  g.process_to_action(id)
  if g.exception
    out.puts({ 'id' => id, 'exception' => g.exception.to_s[0, 300] }.to_json)
    break
  end
  out.puts snap(g, id, seat).to_json
end
result = (g.result rescue {}) || {}
out.puts({ 'final' => true, 'finished' => g.finished,
           'result' => result.to_h { |pid, v| [seat[pid.to_i] || seat[pid] || pid, v] },
           'game_end_reason' => g.game_end_reason.to_s }.to_json)
out.close
