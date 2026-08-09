import torch
import torch.nn as nn

from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 1000

OMEGA_REAL = 30.0

OMEGA_MIN = 20.0
OMEGA_MAX = 25.0
OMEGA_STEP = 0.1


# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(0, 1, N_POINTS)


# ----------------------------------
# SEÑAL REAL
# ----------------------------------

omega_real = torch.tensor(OMEGA_REAL)

x_real = simulate_oscillator(
    t,
    omega_real,
)


# ----------------------------------
# FFT DE LA SEÑAL REAL
# ----------------------------------

X_real = torch.fft.fft(x_real)


# ----------------------------------
# LOSS
# ----------------------------------

loss_function = nn.MSELoss()


# ----------------------------------
# RECORRER VALORES DE OMEGA
# ----------------------------------

omega_values = torch.arange(
    OMEGA_MIN,
    OMEGA_MAX + OMEGA_STEP,
    OMEGA_STEP,
)

losses = []


for omega_value in omega_values:

    # ------------------------------
    # SIMULACIÓN
    # ------------------------------

    x_pred = simulate_oscillator(
        t,
        omega_value,
    )

    # ------------------------------
    # FFT
    # ------------------------------

    X_pred = torch.fft.fft(x_pred)

    # ------------------------------
    # MAGNITUD
    # ------------------------------

    magnitude_real = torch.abs(X_real)

    magnitude_pred = torch.abs(X_pred)

    # ------------------------------
    # LOSS FRECUENCIAL
    # ------------------------------

    loss = loss_function(
        magnitude_pred,
        magnitude_real,
    )

    losses.append(loss.item())


# ----------------------------------
# MOSTRAR LOS 10 MEJORES
# ----------------------------------

results = list(
    zip(
        omega_values.tolist(),
        losses,
    )
)

results.sort(
    key=lambda item: item[1]
)


print("Diez valores de omega con menor Loss FFT:\n")

for omega, loss in results[:10]:

    print(
        f"omega = {omega:7.2f} | "
        f"Loss FFT = {loss:.8f}"
    )