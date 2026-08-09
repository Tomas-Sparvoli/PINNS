import torch
import torch.nn as nn


class OmegaNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        omega_min=5.0,
        omega_max=50.0
    ):

        super().__init__()

        self.omega_min = omega_min
        self.omega_max = omega_max

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.SiLU(),
            
            nn.Linear(256, 128),
            nn.SiLU(),

            nn.Linear(128, 64),
            nn.SiLU(),

            nn.Linear(64, 32),
            nn.SiLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        omega_raw = self.net(x)

        omega_normalized = torch.sigmoid(
            omega_raw
        )

        omega = (
            self.omega_min
            +
            (
                self.omega_max
                - self.omega_min
            )
            * omega_normalized
        )

        return omega