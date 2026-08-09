import torch

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


# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(0, 1, N_POINTS)


# ----------------------------------
# GENERAR SEÑALES
# ----------------------------------

signals = []

for omega_value in OMEGAS:

    omega = torch.tensor(omega_value)

    x = simulate_oscillator(t, omega)

    signals.append(x)


signals = torch.stack(signals)


# ----------------------------------
# COMPARAR SEÑALES CONSECUTIVAS
# ----------------------------------

print("Diferencia entre señales consecutivas:\n")

for i in range(len(OMEGAS) - 1):

    difference = torch.mean(
        torch.abs(
            signals[i + 1] - signals[i]
        )
    )

    print(
        f"ω = {OMEGAS[i]:5.1f} → "
        f"ω = {OMEGAS[i + 1]:5.1f} | "
        f"MAE = {difference.item():.6f}"
    )


# ----------------------------------
# COMPARAR TODAS CONTRA TODAS
# ----------------------------------

print("\nMatriz de diferencias:\n")

difference_matrix = torch.zeros(
    len(OMEGAS),
    len(OMEGAS)
)


for i in range(len(OMEGAS)):

    for j in range(len(OMEGAS)):

        difference_matrix[i, j] = torch.mean(
            torch.abs(
                signals[i] - signals[j]
            )
        )


print(difference_matrix)