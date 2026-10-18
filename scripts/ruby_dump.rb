# Per-action state dump of a recorded 18xx.games game in the REAL Ruby
# engine (tobymao/18xx), for Rust-vs-Ruby divergence hunting — any title
# (scripts/ruby_diff.py drives it inside the ruby:3.2-slim image).
#
# usage: ruby ruby_dump.rb <game.json> <out.jsonl>
#
# One JSON line per processed FILE action id (Ruby's own undo/redo
# filtering, messages skipped), via Game#process_to_action — so lines are
# keyed by the ids in the file, never by a renumbered index. The last line
# is {"final": true, ...} with Ruby's result and game_end_reason; a line
# {"id": N, "exception": ...} marks where Ruby itself rejects the record.
# Players are keyed by SEAT (1-based recorded order) — no names.
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'
require_relative 'ruby_shims'

data = JSON.parse(File.read(ARGV[0]))
out = File.open(ARGV[1], 'w')
actions = data['actions']
filtered, = Engine::Game::Base.filtered_actions(actions)
ids = actions.each_with_index.select { |a, i| filtered[i] && a['type'] != 'message' }.map { |a, _| a['id'] }
seat = {}
data['players'].each_with_index { |p, i| seat[p['id']] = i + 1 }

g = Engine::Game.load(data, at_action: 0)
title = data['title'].to_s

def snap(g, id, seat, title)
  step = g.round.active_step rescue nil
  ent = (step&.current_entity rescue nil) || (g.current_entity rescue nil)
  corps = {}
  g.corporations.each do |c|
    row = {
      'p' => c.share_price&.price,
      'cash' => c.cash,
      'fl' => c.floated?,
      'tr' => c.trains.map(&:name),
    }
    row['loans'] = c.loans.size if c.respond_to?(:loans)
    corps[c.name] = row
  end
  row = {
    'id' => id,
    'round' => g.round.class.name.split('::').last,
    'step' => step&.class&.name&.split('::')&.last,
    'ent' => (if ent.respond_to?(:player?) && ent.player? && seat[ent.id] then "Player #{seat[ent.id]}" elsif ent.respond_to?(:company?) && ent.company? then ent.id else ent&.name end),
    'phase' => g.phase.name,
    'corps' => corps,
    'pcash' => g.players.to_h { |p| [seat[p.id] || p.id, p.cash] },
    'finished' => g.finished,
    'bank' => g.bank.cash,
  }
  if title.start_with?('1822')
    row['debt'] = g.players.to_h { |p| [seat[p.id] || p.id, p.debt] }
    row['order'] = g.players.map { |p| seat[p.id] }
    row['depot'] = g.depot.upcoming.first(3).map(&:id)
    # city tokens by slot, for the hexes that hold any (ruby_diff --tokens)
    if ENV['DUMP_TOKENS']
      tok = {}
      g.hexes.each do |h|
        cities = h.tile.cities.map { |c| c.tokens.map { |t| t&.corporation&.id } }
        tok[h.id] = cities unless cities.flatten.compact.empty?
      end
      row['tokens'] = tok
    end
    row['companies'] = g.companies.reject(&:closed?).select(&:owner).to_h do |c|
      o = c.owner
      oname = o.player? ? "player:#{seat[o.id]}" : (o.respond_to?(:corporation?) && o.corporation? ? "corp:#{o.id}" : o.name)
      [c.id, oname]
    end
  end
  row
end

ids.each do |id|
  g.process_to_action(id)
  if g.exception
    out.puts({ 'id' => id, 'exception' => g.exception.to_s[0, 300] }.to_json)
    break
  end
  out.puts snap(g, id, seat, title).to_json
end
result = (g.result rescue {}) || {}
out.puts({ 'final' => true, 'finished' => g.finished,
           'result' => result.to_h { |pid, v| [seat[pid.to_i] || seat[pid] || pid, v] },
           'game_end_reason' => g.game_end_reason.to_s }.to_json)
out.close
