# The site's Ruby judging a game move by move as rl18xx-next's record_action does for an Opal page
# (docs/SITE_ENGINE_RS_1867.md §10.4): each action processed live (`@game.loading` false: every player-time check)
# with `validate_auto_actions: true` (its posted auto actions must be Ruby's own), a run's posted revenues stripped
# first (lib/run_revenue.rb: Ruby prices it). A move refused on the game advanced move by move is judged again on the
# game freshly loaded from the actions before it, as the server loads it (`reloaded`: Ruby's caches differ). Writes <out>/<name>.live.json: {moves, stopped: null | {index, id, type,
# message}, finished, result, repriced: [...]} and <out>/<name>.stored.json, the game as the server would store it.
#   ruby live_replay.rb <out dir> <game.json>...
require_relative '/src/lib/engine'
require 'json'

out = ARGV.shift
ARGV.each do |f|
  name = File.basename(f, '.json')
  data = JSON.parse(File.read(f))
  actions = data['actions'] || []
  g = Engine::Game.load(data.merge('actions' => []), strict: true)
  stopped = nil
  repriced = []
  reloaded = []
  actions.each_with_index do |a, i|
    params = a.reject { |k, _| %w[id created_at].include?(k) }
    posted = nil
    if params['type'] == 'run_routes'
      posted = (params['routes'] || []).map { |r| r['revenue'] }
      params = params.merge('routes' => (params['routes'] || []).map { |r| r.reject { |k, _| %w[revenue revenue_str subsidy].include?(k) } })
      params.delete('extra_revenue')
      params.delete('subsidy')
    end
    prior = g.raw_actions.map(&:to_h)
    g = g.process_action(params, validate_auto_actions: true)
    if g.exception
      # the server judges a move on the game freshly loaded from the stored actions (record_action's
      # Engine::Game.load), whose caches can differ from a game advanced move by move (Ruby's graph cache): judge it
      # so before calling it refused
      first = g.exception.to_s[0, 300]
      g = Engine::Game.load(data.merge('actions' => prior), strict: true).process_action(params, validate_auto_actions: true)
      if g.exception
        stopped = { 'index' => i, 'id' => a['id'], 'type' => a['type'], 'message' => g.exception.to_s[0, 300] }
        break
      end
      reloaded << { 'id' => a['id'], 'incremental' => first }
    end
    if posted
      stored = (g.raw_actions.last.to_h['routes'] || []).map { |r| r['revenue'] }
      repriced << { 'id' => a['id'], 'posted' => posted, 'ruby' => stored } if stored != posted
    end
  end
  File.write(File.join(out, "#{name}.live.json"),
             JSON.generate({ 'moves' => actions.size, 'stopped' => stopped, 'finished' => g.finished,
                             'result' => g.result.transform_keys(&:to_s), 'repriced' => repriced.first(20),
                             'repriced_count' => repriced.size, 'reloaded' => reloaded.first(20),
                             'reloaded_count' => reloaded.size }))
  File.write(File.join(out, "#{name}.stored.json"), JSON.generate(data.merge('actions' => g.raw_actions.map(&:to_h))))
rescue StandardError => e
  File.write(File.join(out, "#{name}.live.json"), JSON.generate({ 'error' => "#{e.class}: #{e.message}"[0, 500], 'trace' => e.backtrace&.first(5) }))
end
