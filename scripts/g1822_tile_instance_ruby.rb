# Ruby side of scripts/g1822_tile_instance_check.py --ruby: at recorded
# decision points, the instance of a tile name Ruby HEAD's UI offers -- the
# first of that name in the supply (`Tracker#potential_tiles` keeps
# `@game.tiles.uniq(&:name)`).
#
# usage: ruby g1822_tile_instance_ruby.rb <game.json> <checks.json> <out.json>
#   checks.json: [{"fid": N, "name": "9"}] (sorted by fid; the state BEFORE file action N)
# out.json: [{"fid": N, "name": "9", "first": "9-13"} | {"fid": N, "error": "..."}]
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'
require_relative 'ruby_shims'

data = JSON.parse(File.read(ARGV[0]))
checks = JSON.parse(File.read(ARGV[1]))
g = Engine::Game.load(data, at_action: 0)
out = []
checks.each do |c|
  begin
    g.process_to_action(c['fid'] - 1)
    first = g.tiles.find { |t| t.name == c['name'] }
    out << { fid: c['fid'], name: c['name'], first: first&.id }
  rescue StandardError => e
    out << { fid: c['fid'], name: c['name'], error: "#{e.class}: #{e.message}"[0, 200] }
    break
  end
end
File.write(ARGV[2], out.to_json)
