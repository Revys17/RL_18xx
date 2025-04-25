from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_mapper import ACTION_ENCODING_SIZE
from .encoder import Encoder_1830

INPUT_SIZE = Encoder_1830.ENCODING_SIZE
POLICY_OUTPUT_SIZE = ACTION_ENCODING_SIZE
VALUE_OUTPUT_SIZE = 4


class Model(nn.Module):
    """
    Neural Network model for the 18xx AlphaZero agent
    """

    def __init__(self, hidden_neurons=128):
        """
        Initializes the layers of the neural network.
        Args:
            hidden_neurons (int): Number of neurons in the hidden layers.
        """
        super(Model, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, hidden_neurons)
        self.bn1 = nn.BatchNorm1d(hidden_neurons)  # Batch norm often helps training
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.bn2 = nn.BatchNorm1d(hidden_neurons)

        self.policy_head = nn.Linear(hidden_neurons, POLICY_OUTPUT_SIZE)
        self.value_head = nn.Linear(hidden_neurons, VALUE_OUTPUT_SIZE)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): The encoded game state tensor (batch_size, INPUT_SIZE).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - policy_log_probs (torch.Tensor): Log probabilities for each action (batch_size, POLICY_OUTPUT_SIZE).
                - value (torch.Tensor): Estimated state value (batch_size, VALUE_OUTPUT_SIZE).
        """
        x = x.float()

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        policy_log_probs = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))

        return policy_log_probs, value
