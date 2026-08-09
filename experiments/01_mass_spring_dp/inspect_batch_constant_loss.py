import torch
import torch.nn as nn

from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 1000

OMEGAS_REAL = [
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

OMEGA_MIN = 0.0
OMEGA_MAX = 50.0
OMEGA_STEP = 0.1


# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(
    0,
    1,
    N_POINTS,
)


# ----------------------------------
# GENERAR SEÑALES REALES
# ----------------------------------

signals = []

for omega_value in OMEGAS_REAL:

    omega = torch.tensor(omega_value)

    x_real = simulate_oscillator(
        t,
        omega,
    )

    signals.append(x_real)


signals = torch.stack(signals)


# ----------------------------------
# LOSS
# ----------------------------------

loss_function = nn.MSELoss()


# ----------------------------------
# PROBAR UN ÚNICO OMEGA PARA
# TODAS LAS SEÑALES
# ----------------------------------

omega_values = torch.arange(
    OMEGA_MIN,
    OMEGA_MAX + OMEGA_STEP,
    OMEGA_STEP,
)

results = []


for omega_value in omega_values:

    total_loss = 0.0

    for x_real in signals:

        x_pred = simulate_oscillator(
            t,
            omega_value,
        )

        loss = loss_function(
            x_pred,
            x_real,
        )

        total_loss += loss.item()

    mean_loss = total_loss / len(signals)

    results.append(
        (
            omega_value.item(),
            mean_loss,
        )
    )


# ----------------------------------
# ORDENAR RESULTADOS
# ----------------------------------

results.sort(
    key=lambda item: item[1]
)


# ----------------------------------
# MOSTRAR RESULTADOS
# ----------------------------------

print(
    "Diez valores de omega con menor "
    "Loss promedio:\n"
)

for omega, loss in results[:10]:

    print(
        f"omega = {omega:7.2f} | "
        f"Loss promedio = {loss:.8f}"
    )