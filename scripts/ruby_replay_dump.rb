# Replay a game OUR engine generated (an 18xx.games-format export: self-play,
# eval or random-walk games) in the REAL Ruby engine (tobymao/18xx) and dump
# its state after every action, for scripts/ruby_replay_check.py.
#
# usage: ruby ruby_replay_dump.rb <game.json> <out.jsonl>
#        ruby ruby_replay_dump.rb <game.json> - <id>   # stop before action <id>, print context
#
# Actions are processed one by one with Game#process_action after an empty
# load, so @loading is false and Ruby runs every legality check its UI runs
# (tile rotations, token reachability, ...). Lines:
#   {"id": N, ...state...}         after recorded action N (players by SEAT)
#   {"id": N, "exception": "..."}  Ruby rejects recorded action N (stop)
#   {"final": true, ...}           Ruby's result
#
# Implicit passes: where our engine moves on by itself but Ruby still waits
# in a step that offers "pass" (e.g. BuyTrain when the only trains in reach
# belong to other players' corporations, which our 1830 deliberately doesn't
# let a corporation buy), the next recorded action was taken somewhere else.
# The checker tags every action with our engine's acting entity, step and
# round (`_rs_ent`, `_rs_step`, `_rs_round`; `_rs_finished` on the game when
# ours ended after its last action); wherever Ruby's differ, a pass for
# Ruby's acting entity is processed first and reported in the state line
# ("inserted": [{step, ent, why}]) -- counted, grouped by `why`, never
# silently: an unexpected `why` is a finding.
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'

def label(g, owner, corp, seat)
  return "p#{seat[owner.id]}" if owner.respond_to?(:player?) && owner.player?
  return 'pool' if owner == g.share_pool
  return 'ipo' if owner == corp

  owner.name
end

def entity_label(ent, seat)
  return nil unless ent
  return "p#{seat[ent.id]}" if ent.respond_to?(:player?) && ent.player?
  return ent.sym if ent.respond_to?(:company?) && ent.company?

  ent.name
end

# Who the active step waits for (a pending teleport token: the company).
def acting_entity(g)
  step = g.round.active_step
  (step.respond_to?(:active_entities) && step.active_entities.first) || g.current_entity
rescue StandardError
  g.current_entity
end

def step_name(g)
  g.round.active_step&.class&.name&.split('::')&.last
rescue StandardError
  nil
end

def snap(g, id, seat)
  corps = {}
  g.corporations.each do |c|
    holders = {}
    c.share_holders.each do |owner, percent|
      next if percent.zero?

      holders[label(g, owner, c, seat)] = percent
    end
    corps[c.name] = {
      'p' => c.share_price&.price,
      'cash' => c.cash,
      'fl' => c.floated?,
      'tr' => c.trains.map(&:name).sort,
      'sh' => holders,
    }
  end
  {
    'id' => id,
    'round' => g.round.class.name.split('::').last,
    'step' => step_name(g),
    'ent' => (entity_label(acting_entity(g), seat) rescue nil),
    'phase' => g.phase.name,
    'corps' => corps,
    'pcash' => g.players.to_h { |p| [seat[p.id].to_s, p.cash] },
    'privates' => g.companies.reject(&:closed?).to_h { |co| [co.sym, entity_label(co.owner, seat)] },
    'finished' => g.finished,
    'bank' => g.bank.cash,
  }
end

# Laid tiles and placed tokens (per city, by corporation), hexes sorted.
def map_state(g)
  g.hexes.map do |h|
    t = h.tile
    tokens = t.cities.map { |c| c.tokens.compact.map { |tk| tk.corporation.name }.sort }
    next nil if t.preprinted && tokens.all?(&:empty?)

    [h.id, t.preprinted ? 'pre' : t.name, t.preprinted ? 0 : t.rotation, tokens]
  end.compact.sort_by(&:first)
end

# Why Ruby still waits for `ent` in `step` (for grouping).
def wait_reason(g, step, ent)
  name = step.class.name.split('::').last
  if step.is_a?(Engine::Step::BuyTrain)
    others = g.depot.other_trains(ent)
    own = others.select { |t| t.owner.owner == ent.owner }
    depot_ok = g.depot.depot_trains.any? { |t| t.price <= ent.cash }
    return 'buy_train: only other players\' corporations\' trains in reach' if !depot_ok && own.empty? && !others.empty?

    return 'buy_train: other'
  end
  if step.is_a?(Engine::Step::BuyCompany)
    companies = g.purchasable_companies(ent)
    return 'buy_company: active ability, nothing to buy' if companies.empty?

    own = companies.select { |c| c.owner == ent.owner && c.min_price <= ent.cash }
    return 'buy_company: only other players\' companies' if own.empty?

    return 'buy_company: other'
  end
  if step.is_a?(Engine::Step::BuySellParShares)
    acts = (step.actions(ent) rescue [])
    if (acts - %w[buy_company pass]).empty? && step.buyable_bank_owned_companies(ent).empty?
      return 'stock: only interplayer private purchases'
    end

    return 'stock: other'
  end
  name
end

def round_signature(g)
  type = g.round.class.name.split('::').last
  "#{type}:#{g.turn}:#{type == 'Operating' ? g.round.round_num : 0}"
end

# Does Ruby still wait somewhere our engine wasn't when it recorded this
# action? Another round, another acting entity, or the same entity in another
# step. After our last action our game had ended (GAME_OVER): Ruby still
# waits wherever it is.
GAME_OVER = { '_rs_ent' => '__game_over__' }.freeze

def waits_elsewhere?(g, action, ent, step, seat)
  ours = action['_rs_ent']
  return false if ours.nil?
  return !g.finished if ours == '__game_over__'

  return true if action['_rs_round'] && round_signature(g) != action['_rs_round']
  return true if entity_label(ent, seat) != ours

  # our step names don't separate company abilities (SpecialToken / SpecialTrack)
  name = step.class.name.split('::').last.delete_prefix('Special')
  !action['_rs_step'].nil? && action['_rs_step'].delete_prefix('Special') != name
end

def insert_passes(g, action, seat)
  inserted = []
  20.times do
    break if g.finished

    ent = acting_entity(g)
    step = g.round.active_step
    break if ent.nil? || step.nil?
    break unless waits_elsewhere?(g, action, ent, step, seat)
    break unless (step.actions(ent) rescue []).include?('pass')

    why = wait_reason(g, step, ent)
    type = ent.player? ? 'player' : (ent.corporation? ? 'corporation' : 'company')
    id = ent.player? ? ent.id : (ent.company? ? ent.sym : ent.name)
    g.process_action({ 'type' => 'pass', 'entity' => id, 'entity_type' => type })
    break if g.exception

    inserted << { 'step' => step.class.name.split('::').last, 'ent' => entity_label(ent, seat), 'why' => why }
  end
  inserted
end

# Context for debugging: who acts, the step, everyone's actions.
def probe(g, seat)
  step = g.round.active_step
  ent = g.current_entity
  puts "round=#{g.round.class.name.split('::').last} step=#{step.class.name.split('::').last} " \
       "ent=#{entity_label(ent, seat)} turn=#{g.turn} phase=#{g.phase.name}"
  g.players.each do |p|
    puts "  p#{seat[p.id]} cash=#{p.cash} certs=#{g.num_certs(p)}/#{g.cert_limit(p)} passed=#{p.passed?} " \
         "actions=#{(step.actions(p) rescue []).inspect}"
  end
  puts "  ent actions=#{(step.actions(ent) rescue []).inspect}" if ent && !ent.player?
  g.corporations.select(&:ipoed).each do |c|
    puts "  #{c.name} price=#{c.share_price&.price} par=#{c.par_price&.price} cash=#{c.cash} " \
         "trains=#{c.trains.map(&:name)} holders=#{c.share_holders.to_h { |o, v| [label(g, o, c, seat), v] }}"
  end
  if ent.respond_to?(:corporation?) && ent.corporation?
    puts "  #{ent.name}: trains=#{ent.trains.map(&:name)} " \
         "tokens=#{ent.tokens.select(&:used).map { |t| [t.city.hex.id, t.city.tile.cities.index(t.city)] }} " \
         "route_info=#{(g.graph.route_info(ent) rescue nil).inspect}"
    nodes = (g.graph.connected_nodes(ent).keys rescue [])
    puts "  connected nodes=#{nodes.map { |n| "#{n.hex.id}:#{n.class.name.split('::').last}" }.sort}"
  end
  puts "  log: #{g.log.last(10).map(&:message).join(' | ')}"
end

data = JSON.parse(File.read(ARGV[0]))
probe_at = ARGV[2]&.to_i
out = probe_at ? File.open(File::NULL, 'w') : File.open(ARGV[1], 'w')
seat = {}
data['players'].each_with_index { |p, i| seat[p['id']] = i + 1 }
g = Engine::Game.load(data, at_action: 0)

actions = data['actions']
last_map = nil
actions.each_with_index do |action, i|
  if probe_at && action['id'] >= probe_at
    probe(g, seat)
    exit
  end
  g.process_action(action.reject { |k, _| %w[id auto_actions].include?(k) || k.start_with?('_') })
  if g.exception
    out.puts({ 'id' => action['id'], 'exception' => g.exception.to_s[0, 400] }.to_json)
    break
  end
  nxt = actions[i + 1] || (data['_rs_finished'] ? GAME_OVER : nil)
  inserted = nxt ? insert_passes(g, nxt, seat) : []
  if g.exception
    out.puts({ 'id' => nxt['id'] || action['id'], 'exception' => "inserted pass: #{g.exception.to_s[0, 380]}" }.to_json)
    break
  end
  line = snap(g, action['id'], seat)
  line['inserted'] = inserted unless inserted.empty?
  map = map_state(g)
  # the map only when it changed (the checker carries it forward)
  line['map'] = map if map != last_map
  last_map = map
  out.puts line.to_json
end
result = (g.result rescue {}) || {}
out.puts({ 'final' => true, 'finished' => g.finished,
           'result' => result.to_h { |pid, v| [(seat[pid.to_i] || seat[pid] || pid).to_s, v] },
           'game_end_reason' => g.game_end_reason.to_s }.to_json)
out.close
