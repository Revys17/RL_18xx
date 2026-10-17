# Ruby side of scripts/g1822_router_check.py --ruby: validate and price the
# engine's runs in Ruby HEAD (tobymao/18xx) at recorded decision points.
#
# usage: ruby g1822_router_ruby.rb <game.json> <checks.json> <out.json>
#   checks.json: [{"fid": N, "corp": "LYR", "routes": [run_routes entries]}]
#   (sorted by fid; revenue omitted so Ruby validates every route)
# out.json: [{"fid": N, "total": T} | {"fid": N, "error": "..."}]
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'

data = JSON.parse(File.read(ARGV[0]))
checks = JSON.parse(File.read(ARGV[1]))
g = Engine::Game.load(data, at_action: 0)
out = []
checks.each do |c|
  g.process_to_action(c['fid'] - 1)
  corp = g.corporation_by_id(c['corp'])
  routes = []
  c['routes'].each do |r|
    train = corp.trains.find { |t| t.id == r['train'] }
    routes << Engine::Route.new(g, g.phase, train,
                                connection_hexes: r['connections'],
                                hexes: r['hexes'].map { |id| g.hex_by_id(id) },
                                nodes: r['nodes'], routes: routes)
  end
  begin
    total = routes.sum(&:revenue)
    out << { fid: c['fid'], total: total }
  rescue StandardError => e
    out << { fid: c['fid'], error: "#{e.class}: #{e.message}"[0, 200] }
  end
end
File.write(ARGV[2], out.to_json)
