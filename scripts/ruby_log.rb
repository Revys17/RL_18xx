# Print a recorded game's Ruby log between two FILE action ids (inclusive),
# for divergence hunting next to scripts/ruby_diff.py.
#
# usage: ruby ruby_log.rb <game.json> <from_id> <to_id> [expr]
#   expr: optional Ruby evaluated with `g` bound after <to_id> (printed with p)
require 'json'
$LOAD_PATH.unshift('/18xx/lib')
require 'engine'

data = JSON.parse(File.read(ARGV[0]))
from = ARGV[1].to_i
to = ARGV[2].to_i
g = Engine::Game.load(data, at_action: from - 1)
n0 = g.log.size
g.process_to_action(to)
g.log.to_a[n0..].each { |l| puts(l.respond_to?(:message) ? l.message : l.to_s) }
puts "exception: #{g.exception}" if g.exception
if ARGV[3]
  p(eval(ARGV[3])) # rubocop:disable Security/Eval
end
