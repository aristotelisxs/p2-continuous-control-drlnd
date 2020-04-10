import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import hidden_init


class Actor(nn.Module):
    """Policy updates through the Actor (Policy) Model (towards the direction suggested by the Critic)"""

    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        """
        Initialize parameters and build the actor model.
        :param state_size: (int) State dimensions
        :param action_size: (int) Action dimensions
        :param seed: Random seed
        :param fc1_units: Number of units for the first hidden layer
        :param fc2_units: Number of units for the second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        # Batch normalization can help in reducing the variance between updates of the network weights
        self.batch_norm_1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.batch_norm_2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialise network weights given a range that is inversely proportial to the layer's size (no. of units)"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = F.relu(self.batch_norm_1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
