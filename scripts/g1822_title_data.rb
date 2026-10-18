# Dump an 1822-family title's static data from the REAL Ruby engine as JSON,
# for scripts/g1822_title_data.py (which writes the Rust data module).
#
# usage (inside ruby:3.2-slim, the clone mounted at /18xx):
#   ruby g1822_title_data.rb G1822PNW /out/pnw.json
#
# Everything comes from the class constants the engine itself uses:
# COMPANIES / CORPORATIONS (entities.rb), TRAINS / PHASES / MARKET and the
# game.rb constants, HEXES / TILES / LOCATION_NAMES (map.rb), plus the DSL
# of every standard tile the title's TILES names (Engine::Config::Tile).
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'

mod = Engine::Game.const_get(ARGV[0])
klass = mod.const_get(:Game)

def jsonable(x)
  case x
  when Hash then x.to_h { |k, v| [k.to_s, jsonable(v)] }
  when Array then x.map { |v| jsonable(v) }
  when Symbol then x.to_s
  when Proc then nil
  else x
  end
end

tiles = {}
klass::TILES.each do |name, spec|
  if spec.is_a?(Hash)
    tiles[name] = { 'count' => spec['count'], 'color' => spec['color'].to_s, 'code' => spec['code'],
                    'hidden' => spec['hidden'] ? true : false }
  else
    color, code = nil, nil
    %w[WHITE YELLOW GREEN BROWN GRAY RED BLUE].each do |c|
      h = Engine::Config::Tile.const_get(c) rescue nil
      next unless h && h[name]

      color = c.downcase
      code = h[name]
    end
    raise "no standard tile #{name}" unless code

    tiles[name] = { 'count' => spec, 'color' => color, 'code' => code, 'hidden' => false }
  end
end

hexes = []
klass::HEXES.each do |color, defs|
  defs.each do |coords, dsl|
    # index: the coord's place in its group (init_hexes' preprinted tile id
    # is "<coord>-<index>")
    Array(coords).each_with_index { |c, i| hexes << { 'coord' => c, 'color' => color.to_s, 'code' => dsl, 'index' => i } }
  end
end

consts = {}
klass.constants.sort.each do |c|
  v = klass.const_get(c)
  next if v.is_a?(Module) || v.is_a?(Class)

  j = jsonable(v)
  begin
    JSON.generate(j)
    consts[c.to_s] = j
  rescue StandardError
    nil
  end
end

meta = mod.const_get(:Meta)
out = {
  'title' => meta::GAME_TITLE,
  'player_range' => meta::PLAYER_RANGE,
  'optional_rules' => jsonable(meta::OPTIONAL_RULES),
  'layout' => klass::LAYOUT.to_s,
  'companies' => jsonable(klass::COMPANIES),
  'corporations' => jsonable(klass::CORPORATIONS),
  'trains' => jsonable(klass::TRAINS),
  'phases' => jsonable(klass::PHASES),
  'market' => jsonable(klass::MARKET),
  'hexes' => hexes,
  'tiles' => tiles,
  'location_names' => jsonable(klass::LOCATION_NAMES),
  'consts' => consts,
}
File.write(ARGV[1], JSON.pretty_generate(out))
