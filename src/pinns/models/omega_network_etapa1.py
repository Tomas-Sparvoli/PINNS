import torch
import torch.nn as nn


class OmegaNetwork(nn.Module):

    def __init__(
        self,
        omega_min=5.0,
        omega_max=50.0
    ):

        super().__init__()

        self.omega_min = omega_min
        self.omega_max = omega_max

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),

            nn.Linear(32, 32),
            nn.SiLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        # ----------------------------------------------------
        # NORMALIZACIÓN DE LA ENTRADA
        # ----------------------------------------------------

        x_normalized = (
            x - self.omega_min
        ) / (
            self.omega_max - self.omega_min
        )


        # ----------------------------------------------------
        # RED
        # ----------------------------------------------------

        omega_raw = self.net(
            x_normalized
        )


        # ----------------------------------------------------
        # SALIDA ACOTADA
        # ----------------------------------------------------

        omega_normalized = torch.sigmoid(
            omega_raw
        )


        # ----------------------------------------------------
        # DESNORMALIZAR
        # ----------------------------------------------------

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