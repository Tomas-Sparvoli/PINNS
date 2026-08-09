import torch
import torch.nn as nn

from pinns.models.omega_network import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 1000

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0

EPOCHS = 1000
LEARNING_RATE = 1e-3


# ----------------------------------
# DATASET
# ----------------------------------

omegas = torch.tensor([
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0
])


# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(
    0,
    1,
    N_POINTS
)


# ----------------------------------
# GENERAR SEÑALES
# ----------------------------------

signals = []

for omega in omegas:

    x = simulate_oscillator(
        t,
        omega
    )

    signals.append(x)


signals = torch.stack(signals)


# ----------------------------------
# MODELO
# ----------------------------------

model = OmegaNetwork(
    input_size=N_POINTS,
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX
)


# ----------------------------------
# OPTIMIZADOR
# ----------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ----------------------------------
# LOSS
# ----------------------------------

loss_function = nn.MSELoss()


# ----------------------------------
# PREDICCIONES INICIALES
# ----------------------------------

with torch.no_grad():

    omega_initial = model(
        signals
    ).squeeze(1)


print("\nPredicciones iniciales:")

for real, pred in zip(
    omegas,
    omega_initial
):

    print(
        f"ω real = {real.item():6.2f} | "
        f"ω inicial = {pred.item():6.2f}"
    )


# ----------------------------------
# ENTRENAMIENTO
# ----------------------------------

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    # ----------------------------------
    # RED
    # ----------------------------------

    omega_pred = model(
        signals
    ).squeeze(1)

    # ----------------------------------
    # SIMULADOR
    # ----------------------------------

    predicted_signals = []

    for omega in omega_pred:

        x_pred = simulate_oscillator(
            t,
            omega
        )

        predicted_signals.append(
            x_pred
        )

    predicted_signals = torch.stack(
        predicted_signals
    )

    # ----------------------------------
    # LOSS TEMPORAL
    # ----------------------------------

    loss = loss_function(
        predicted_signals,
        signals
    )

    # ----------------------------------
    # BACKPROPAGATION
    # ----------------------------------

    loss.backward()

    # ----------------------------------
    # ACTUALIZACIÓN
    # ----------------------------------

    optimizer.step()

    # ----------------------------------
    # MOSTRAR LOSS
    # ----------------------------------

    if epoch % 100 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {loss.item():.6f}"
        )


# ----------------------------------
# RESULTADOS FINALES
# ----------------------------------

with torch.no_grad():

    omega_final = model(
        signals
    ).squeeze(1)


print("\nResultados:")

for real, pred in zip(
    omegas,
    omega_final
):

    print(
        f"ω real = {real.item():6.2f} | "
        f"ω pred = {pred.item():6.2f}"
    )