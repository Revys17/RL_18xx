# Shims for the Ruby rig (ruby_dump.rb, g1822_router_ruby.rb): places where
# Ruby HEAD under MRI raises on code the site runs fine under Opal (Opal does
# not check method arity). Each mirrors what the Opal build does; none
# changes a rule.

# 1822CA's `check_distance(route, visits, train = nil)` ends with a bare
# `super` into G1822#check_distance(route, visits): MRI raises ArgumentError
# (given 3, expected 2) on every route it validates; Opal drops the extra
# argument. (Human records carry their revenue, so Ruby never validates
# them; the engine's fuzz runs omit it.)
if defined?(Engine::Game::G1822CA::Game)
  Engine::Game::G1822CA::Game.prepend(Module.new do
    def check_distance(route, visits, train = nil)
      raise Engine::GameError, 'Cannot run Grain train' if grain_train?(train || route.train)

      Engine::Game::G1822::Game.instance_method(:check_distance).bind_call(self, route, visits)
    end
  end)
end
