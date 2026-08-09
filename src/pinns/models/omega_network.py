import torch
import torch.nn as nn


class OmegaNetwork(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        omega_normalized = torch.sigmoid(
            self.net(x)
        )

        return omega_normalized