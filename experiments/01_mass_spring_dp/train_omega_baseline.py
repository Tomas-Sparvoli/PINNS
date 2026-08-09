import torch
import torch.nn as nn

from pinns.models.omega_network import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 1000

OMEGAS = [
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
]

EPOCHS = 1000
LEARNING_RATE = 1e-3

OMEGA_MAX = 50.0


# ----------------------------------
# DATASET
# ----------------------------------

t = torch.linspace(0, 1, N_POINTS)

signals = []
parameters = []


for omega_value in OMEGAS:

    omega = torch.tensor(omega_value)

    x = simulate_oscillator(t, omega)

    signals.append(x)
    parameters.append(omega)


signals = torch.stack(signals)
parameters = torch.stack(parameters)


# ----------------------------------
# NORMALIZACIÓN DEL PARÁMETRO
# ----------------------------------

parameters_normalized = parameters / OMEGA_MAX


# ----------------------------------
# MODELO
# ----------------------------------

model = OmegaNetwork(input_size=N_POINTS)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

loss_function = nn.MSELoss()


# ----------------------------------
# ENTRENAMIENTO
# ----------------------------------

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    omega_pred_normalized = model(signals).squeeze(1)

    loss = loss_function(
        omega_pred_normalized,
        parameters_normalized,
    )

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {loss.item():.6f}"
        )


# ----------------------------------
# RESULTADOS
# ----------------------------------

with torch.no_grad():

    omega_pred_normalized = model(signals).squeeze(1)

    omega_pred = omega_pred_normalized * OMEGA_MAX


print("\nResultados:")

for real, pred in zip(parameters, omega_pred):

    print(
        f"ω real = {real.item():6.2f} | "
        f"ω pred = {pred.item():6.2f}"
    )