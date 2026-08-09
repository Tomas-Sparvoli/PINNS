import torch
import torch.nn as nn

from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 1000

OMEGA_REAL = 30.0

OMEGA_INITIAL = 10.0

EPOCHS = 1000
LEARNING_RATE = 1e-2


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
# PARÁMETRO A OPTIMIZAR
# ----------------------------------

omega = torch.tensor(
    OMEGA_INITIAL,
    requires_grad=True,
)


# ----------------------------------
# OPTIMIZADOR
# ----------------------------------

optimizer = torch.optim.Adam(
    [omega],
    lr=LEARNING_RATE,
)


loss_function = nn.MSELoss()


# ----------------------------------
# ENTRENAMIENTO
# ----------------------------------

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    # ------------------------------
    # SIMULADOR
    # ------------------------------

    x_pred = simulate_oscillator(
        t,
        omega,
    )

    # ------------------------------
    # LOSS
    # ------------------------------

    loss = loss_function(
        x_pred,
        x_real,
    )

    # ------------------------------
    # GRADIENTE
    # ------------------------------

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {loss.item():.6f} | "
            f"omega = {omega.item():.6f}"
        )


print("\nResultado final:")
print(f"ω real = {OMEGA_REAL:.2f}")
print(f"ω encontrado = {omega.item():.2f}")