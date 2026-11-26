# The reference Ruby twin of the shadow key v1 (docs/SITE_ENGINE_RS.md §5.3 "v1 as built"), in the site's Ruby, to
# check engine-rs's key on the corpus (A) and as a reference for B's lib/engine_shadow.rb (not B's implementation).
# usage: ruby keys_reference.rb <out dir> <game.json>...   -> <out>/<id>.ruby.json {"positions": [{"actionId", "state"}]}
require_relative '/src/lib/engine'
require 'json'

def owner_code(o)
  return 'bank' if o.nil?
  return "p#{o.id}" if o.player?
  return "c#{o.id}" if o.corporation?

  'bank'
end

def key(g, last)
  finished = g.finished
  step = finished ? nil : g.round.active_step
  entity = finished ? nil : (step&.active_entities&.first || g.current_entity)
  players = g.players.map do |p|
    shares = {}
    g.corporations.each { |c| (pct = p.percent_of(c)).positive? && shares[c.id] = pct }
    { 'id' => p.id, 'cash' => p.cash, 'shares' => shares, 'companies' => p.companies.map(&:id).sort,
      'bankrupt' => !!p.bankrupt }
  end
  corps = {}
  g.corporations.each do |c|
    hist = c.operating_history
    lr = nil
    unless hist.empty?
      k = hist.keys.max
      info = hist[k]
      lr = [k[0], k[1], info.revenue, info.dividend.kind]
    end
    corps[c.id] = {
      'cash' => c.cash, 'price' => c.share_price&.price, 'cell' => c.share_price&.coordinates,
      'par' => c.par_price&.price, 'floated' => c.floated?, 'closed' => c.closed?,
      'ipo' => c.ipo_shares.sum(&:percent), 'market' => g.share_pool.percent_of(c),
      'president' => c.owner&.player? ? c.owner.id : nil, 'trains' => c.trains.map(&:id),
      'tokens' => c.tokens.select(&:used).map { |t| "#{t.city.hex.id}-#{t.city.index}" },
      'companies' => c.companies.map(&:id).sort, 'last_run' => lr
    }
  end
  companies = {}
  g.companies.each { |m| companies[m.id] = m.closed? ? 'closed' : owner_code(m.owner) }
  hexes = {}
  g.hexes.each { |h| hexes[h.id] = "#{h.tile.name}@#{h.tile.rotation}" if h.tile != h.original_tile }
  la =
    if last.nil? then nil
    elsif last['type'] == 'run_routes'
      { 'revenues' => last['routes'].map { |r| r['revenue'] }, 'extra_revenue' => last['extra_revenue'] || 0,
        'subsidy' => last['subsidy'] || 0 }
    else { 'type' => last['type'] }
    end
  {
    'v' => 1, 'round' => g.round.name, 'turn' => g.turn, 'round_num' => g.round.round_num, 'phase' => g.phase.name,
    'bank' => g.bank.cash, 'acting' => finished ? [] : g.active_players_id,
    'entity' => entity&.id&.to_s, 'step' => step&.class&.name&.split('::')&.last,
    'actions' => step && entity ? step.actions(entity).map(&:to_s) : [],
    'priority' => g.send(:priority_deal_player)&.id, 'players' => players, 'corps' => corps, 'companies' => companies,
    'depot' => { 'upcoming' => g.depot.upcoming.map(&:id), 'discarded' => g.depot.discarded.map(&:id) },
    'hexes' => hexes, 'finished' => finished, 'result' => g.result.transform_keys(&:to_s),
    'manually_ended' => !!g.manually_ended, 'game_end_reason' => g.game_end_reason&.to_s, 'last_action' => la
  }
end

out = ARGV.shift
ARGV.each do |f|
  data = JSON.parse(File.read(f))
  actions = data['actions'] || []
  filtered, = Engine::Game::Base.filtered_actions(actions)
  g = Engine::Game.load(data.merge('actions' => []), strict: true)
  positions = [{ 'actionId' => 0, 'state' => key(g, nil) }]
  stopped = nil
  filtered.compact.each do |a|
    next if a['type'] == 'message'

    g.process_action(a)
    if g.exception
      stopped = { 'actionId' => a['id'], 'message' => g.exception.to_s }
      break
    end
    positions << { 'actionId' => a['id'], 'state' => key(g, a) }
  end
  File.write(File.join(out, "#{File.basename(f, '.json')}.ruby.json"), JSON.generate({ 'positions' => positions, 'stopped' => stopped }))
rescue StandardError => e
  File.write(File.join(out, "#{File.basename(f, '.json')}.ruby.json"), JSON.generate({ 'error' => "#{e.class}: #{e.message}"[0, 500] }))
end
