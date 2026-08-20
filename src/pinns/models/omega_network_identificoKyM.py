import torch
import torch.nn as nn


class OmegaNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),

            nn.Linear(32, 32),
            nn.SiLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):

        output = self.net(x)

        output = torch.sigmoid(
            output
        )

        k_pred_norm = output[:, 0:1]
        m_pred_norm = output[:, 1:2]

        return (
            k_pred_norm,
            m_pred_norm
        )