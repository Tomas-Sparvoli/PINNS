import torch

from pinns.models.omega_network import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 1000

T_TOTAL = 10.0

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0

# Valores que NO fueron utilizados
# durante el entrenamiento
OMEGAS_TEST = [
    7.0,
    12.0,
    17.0,
    22.0,
    27.0,
    32.0,
    37.0,
    42.0,
    47.0
]


# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(
    0,
    T_TOTAL,
    N_POINTS
)


# ----------------------------------
# MODELO
# ----------------------------------

model = OmegaNetwork(
    input_size=N_POINTS,
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX
)


# ----------------------------------
# CARGAR RED ENTRENADA
# ----------------------------------

model.load_state_dict(
    torch.load(
        "omega_network_fft.pth",
        weights_only=True
    )
)

model.eval()


# ----------------------------------
# EVALUACIÓN
# ----------------------------------

print("\n========================================")
print("PRUEBA DE GENERALIZACIÓN")
print("========================================")

print(
    "\nLa red fue entrenada con:"
)

print(
    "5, 10, 15, 20, 25, 30, 35, 40, 45, 50"
)

print(
    "\nAhora se evalúa con valores no vistos:"
)

print(
    "7, 12, 17, 22, 27, 32, 37, 42, 47"
)


# ----------------------------------
# PROBAR CADA SEÑAL
# ----------------------------------

with torch.no_grad():

    for omega_real in OMEGAS_TEST:

        # ----------------------------------
        # GENERAR SEÑAL NUEVA
        # ----------------------------------

        signal_test = simulate_oscillator(
            t,
            torch.tensor(omega_real)
        )


        # ----------------------------------
        # AGREGAR DIMENSIÓN BATCH
        # ----------------------------------

        signal_test = signal_test.unsqueeze(0)


        # ----------------------------------
        # INFERENCIA
        # ----------------------------------

        omega_pred = model(
            signal_test
        ).squeeze()


        # ----------------------------------
        # ERROR
        # ----------------------------------

        error = (
            omega_pred.item()
            -
            omega_real
        )


        absolute_error = abs(error)


        # ----------------------------------
        # RESULTADO
        # ----------------------------------

        print(
            f"ω real = {omega_real:6.2f} | "
            f"ω pred = {omega_pred.item():6.2f} | "
            f"error = {error:+7.2f}"
            f" | |error| = {absolute_error:6.2f}"
        )