import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class NetworkParameters:
    n_states: int
    n_actions: int
    seed: int
    fc1_size: int
    fc2_size: int


class QNetwork(nn.Module):

    def __init__(self, parameters: NetworkParameters):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(parameters.seed)
        self.fc1 = nn.Linear(parameters.n_states, parameters.fc1_size)
        self.fc2 = nn.Linear(parameters.fc1_size, parameters.fc2_size)
        self.fc3 = nn.Linear(parameters.fc2_size, parameters.n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
