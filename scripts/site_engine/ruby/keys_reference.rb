# The reference Ruby twin of the shadow key v2 (docs/SITE_ENGINE_RS.md §5.3 "v1 as built" and
# docs/SITE_ENGINE_RS_1867.md §6 "v2"), in the site's Ruby, to check engine-rs's key on the corpus (A) and as a
# reference for B's lib/engine_shadow.rb (not B's implementation).
# usage: ruby keys_reference.rb <out dir> <game.json>...   -> <out>/<id>.ruby.json {"positions": [{"actionId", "state"}]}
#
# v2 = v1 with the same meaning + fields present only where the title has them (absent otherwise): per corporation
# `loans` (titles with loans) and `interest` (in an operating round); `loans_left`; `national`; `trainless`; `merger` (a
# merger round: the share dealing's lists only while a new major's shares are dealt); and, for every title, `auction`
# (the bids in progress, absent when there are none).
require_relative '/src/lib/engine'
require 'json'

def owner_code(o)
  return 'bank' if o.nil?
  return "p#{o.id}" if o.player?
  return "c#{o.id}" if o.corporation?

  'bank'
end

def token_place(t)
  "#{t.city.hex.id}-#{t.city.index}"
end

# The bids in progress of the active step (an Auctioner: 1830's WaterfallAuction, 1867's SingleItemAuction and its
# stock round's BuySellParSharesViaBid): `lot` the company or corporation up for auction now (@auctioning) or null,
# `bids` every bid as [item id, player id, price] sorted (item, price, player), `dutch` where the step has a dutch
# mode. Absent when nothing is up and no bid is open.
def auction_key(step)
  return nil unless step.respond_to?(:bids) && step.bids.is_a?(Hash)

  lot = step.instance_variable_get(:@auctioning)
  bids = []
  step.bids.each { |item, list| (list || []).each { |b| bids << [item.id.to_s, b.entity.id, b.price] } }
  return nil if lot.nil? && bids.empty?

  out = { 'lot' => lot&.id&.to_s, 'bids' => bids.sort_by { |i, p, price| [i, price, p] } }
  out['dutch'] = !!step.instance_variable_get(:@dutch_mode) if step.instance_variable_defined?(:@dutch_mode)
  out
end

def merger_key(g)
  r = g.round
  return nil unless r.respond_to?(:converts)

  merge = r.steps.find { |s| s.class.name.end_with?('::Merge') }
  merging = merge&.instance_variable_get(:@merging)
  removing = r.respond_to?(:corporations_removing_tokens) ? r.corporations_removing_tokens : nil
  {
    'converted' => r.converted&.id, 'merge_type' => r.merge_type&.to_s, 'converts' => r.converts.map(&:id),
    'dealing' => r.converted ? r.share_dealing_players.map(&:id) : [],
    'multiple' => r.converted ? r.share_dealing_multiple.map(&:id) : [],
    'merging' => (merging || []).map(&:id), 'converting' => !!merge&.instance_variable_get(:@converting),
    'merge_major' => !!merge&.instance_variable_get(:@merge_major), 'removing' => removing&.first&.id
  }
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
  # (base Game#maximum_loans is private: 0 for a title without loans)
  loans = g.corporations.any? { |c| g.send(:maximum_loans, c).positive? }
  in_or = g.round.operating? && g.respond_to?(:loans_due_interest)
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
      'tokens' => c.tokens.select(&:used).map { |t| token_place(t) },
      'companies' => c.companies.map(&:id).sort, 'last_run' => lr
    }
    corps[c.id]['loans'] = c.loans.size if loans
    corps[c.id]['interest'] = g.loans_due_interest(c) if loans && in_or
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
  out = {
    'v' => 2, 'round' => g.round.name, 'turn' => g.turn, 'round_num' => g.round.round_num, 'phase' => g.phase.name,
    'bank' => g.bank.cash, 'acting' => finished ? [] : g.active_players_id,
    'entity' => entity&.id&.to_s, 'step' => step&.class&.name&.split('::')&.last,
    'actions' => step && entity ? step.actions(entity).map(&:to_s) : [],
    'priority' => g.send(:priority_deal_player)&.id, 'players' => players, 'corps' => corps, 'companies' => companies,
    'depot' => { 'upcoming' => g.depot.upcoming.map(&:id), 'discarded' => g.depot.discarded.map(&:id) },
    'hexes' => hexes, 'finished' => finished, 'result' => g.result.transform_keys(&:to_s),
    'manually_ended' => !!g.manually_ended, 'game_end_reason' => g.game_end_reason&.to_s, 'last_action' => la
  }
  out['loans_left'] = g.loans.size if loans
  if g.respond_to?(:national) && (nat = g.national)
    out['national'] = {
      'tokens' => nat.tokens.select(&:used).map { |t| token_place(t) },
      'companies' => nat.companies.map(&:id).sort,
      'reservations' => (g.instance_variable_get(:@national_reservations) || []).dup
    }
  end
  out['trainless'] = (g.trainless_major || []).map(&:id) if g.respond_to?(:trainless_major)
  if (m = merger_key(g))
    out['merger'] = m
  end
  if (a = !finished && step && auction_key(step))
    out['auction'] = a
  end
  out
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
  File.write(File.join(out, "#{File.basename(f, '.json')}.ruby.json"), JSON.generate({ 'error' => "#{e.class}: #{e.message}"[0, 500], 'trace' => e.backtrace&.first(5) }))
end
