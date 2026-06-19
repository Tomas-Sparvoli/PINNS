from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected network used as a baseline building block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
