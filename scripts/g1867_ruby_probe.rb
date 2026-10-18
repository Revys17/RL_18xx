# Ad-hoc Ruby HEAD probe for 1867 generator bugs (docker rig of
# scripts/ruby_diff.py): load a game, process it to a FILE action id, then
# run a snippet against the live game.
#
# usage: ruby g1867_ruby_probe.rb <game.json> <stop_id> <snippet.rb>
#   the snippet sees `g` (the game, processed through stop_id) and `data`.
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'

data = JSON.parse(File.read(ARGV[0]))
g = Engine::Game.load(data, at_action: 0)
g.process_to_action(ARGV[1].to_i)
puts "exception: #{g.exception}" if g.exception
eval(File.read(ARGV[2]), binding, ARGV[2])
