from torch import Tensor


class Model:
    def __init__(self):
        # Define model shape
        # Takes in a game state tensor
        # Returns the results from the policy and value heads.
        # Policy head returns which action to take
        # Value head returns the value of the game state
        pass

    def forward(self, game_state: Tensor):
        # TODO: run the model on the tensor
        # Return the policy and value head results
        pass
