# Ruby side of scripts/g1867_router_check.py --ruby: validate and price
# runs in Ruby HEAD (tobymao/18xx) at recorded 1867 decision points.
#
# usage: ruby g1867_router_ruby.rb <game.json> <checks.json> <out.json>
#   checks.json: [{"fid": N, "corp": "CPR", "routes": [run_routes entries]}]
#   (FILE action ids; revenue omitted so Ruby validates and prices every
#   route: Route#revenue runs all of route.rb's checks and G1867's
#   compute_stops / revenue_for)
# out.json: [{"fid": N, "total": T, "revenues": [...]} | {"fid": N, "error": "..."}]
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'

data = JSON.parse(File.read(ARGV[0]))
checks = JSON.parse(File.read(ARGV[1])).sort_by { |c| c['fid'] }
g = Engine::Game.load(data, at_action: 0)
out = []
checks.each do |c|
  begin
    # the state just before the recorded run_routes (file action c['fid'])
    g.process_to_action(c['fid'] - 1)
    raise "Ruby stopped: #{g.exception}" if g.exception

    corp = g.corporation_by_id(c['corp'])
    routes = []
    c['routes'].each do |r|
      train = corp.trains.find { |t| t.id == r['train'] }
      raise "no train #{r['train']}" unless train

      routes << Engine::Route.new(g, g.phase, train,
                                  connection_hexes: r['connections'],
                                  hexes: r['hexes'].map { |id| g.hex_by_id(id) },
                                  nodes: r['nodes'], routes: routes)
    end
    revenues = routes.map(&:revenue)
    out << { fid: c['fid'], total: revenues.sum, revenues: revenues }
  rescue StandardError => e
    out << { fid: c['fid'], error: "#{e.class}: #{e.message}"[0, 200] }
  end
end
File.write(ARGV[2], out.to_json)
