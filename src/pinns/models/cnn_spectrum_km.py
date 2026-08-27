import torch
import torch.nn as nn


class SpectrumCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # EXTRACTOR DE CARACTERÍSTICAS
        # ====================================================

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=7,
                padding=3
            ),

            nn.SiLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),


            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),

            nn.SiLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),


            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.SiLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),


            nn.AdaptiveAvgPool1d(
                16
            )
        )


        # ====================================================
        # CABEZA MLP
        # ====================================================

        self.head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 16,
                128
            ),

            nn.SiLU(),

            nn.Linear(
                128,
                64
            ),

            nn.SiLU(),

            nn.Linear(
                64,
                2
            ),

            # k_norm y m_norm quedan entre 0 y 1
            nn.Sigmoid()
        )


    def forward(self, x):

        x = self.features(
            x
        )

        output = self.head(
            x
        )

        k_norm = output[
            :, 0:1
        ]

        m_norm = output[
            :, 1:2
        ]

        return (
            k_norm,
            m_norm
        )