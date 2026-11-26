# Replay site-format games in the site's Ruby (read-only source): where each stops and why. ruby replay.rb <game.json>...
require_relative '/src/lib/engine'
ARGV.each do |f|
  data = JSON.parse(File.read(f))
  begin
    g = Engine::Game.load(data, strict: true)
    ex = g.exception
    last = g.raw_actions.last
    puts "#{File.basename(f)}: #{ex ? "STOP #{ex}" : 'ok'} actions=#{g.raw_actions.size} finished=#{g.finished} round=#{g.round.name} turn=#{g.turn}"
  rescue StandardError => e
    puts "#{File.basename(f)}: RAISE #{e.class}: #{e.message[0, 200]}"
  end
end
