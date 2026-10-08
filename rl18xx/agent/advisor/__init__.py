"""The model advisor: follow a game on an 18xx.games server and show the model's view of it.

- ``live_game``: replays a server game's action list in the Rust engine,
  incrementally.
- ``describe``: human-readable descriptions of the model's moves.
- ``advisor``: loads the policy / auction-policy / value checkpoints and turns a
  position into win estimates, recommended moves, the model's view of the last
  moves, and an optional MCTS search.
- ``app``: the local JSON backend the browser extension (``extension/``) asks
  (``main.py advisor``).
"""

# The default checkpoints (``main.py advisor --policy / --auction-policy / --value``).
DEFAULT_POLICY = "model_checkpoints_pg/pg4_20261007/learner/600.pth"
DEFAULT_AUCTION_POLICY = "model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth"
DEFAULT_VALUE = "model_checkpoints_value/AlphaZeroTransformer/20261004_134558_804974562/26.pth"
