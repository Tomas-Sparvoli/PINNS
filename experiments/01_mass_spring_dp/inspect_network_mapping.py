import torch
import matplotlib.pyplot as plt

from pinns.models.omega_network import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_POINTS = 1000

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0

T_TOTAL = 10.0

MODEL_PATH = "omega_network_fft.pth"


# ============================================================
# DATASET DE ENTRENAMIENTO
# ============================================================

OMEGAS_TRAIN = torch.tensor([
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    27.0,
    28.0,
    29.0,
    30.0,
    31.0,
    32.0,
    33.0,
    34.0,
    35.0,
    36.0,
    37.0,
    38.0,
    39.0,
    40.0,
    41.0,
    42.0,
    43.0,
    44.0,
    45.0,
    46.0,
    47.0,
    48.0,
    49.0,
    50.0
])


# ============================================================
# GRILLA DENSA PARA EVALUACIÓN
# ============================================================

omegas_test = torch.linspace(
    OMEGA_MIN,
    OMEGA_MAX,
    181
)


# ============================================================
# TIEMPO
# ============================================================

t = torch.linspace(
    0,
    T_TOTAL,
    N_POINTS
)


# ============================================================
# CARGAR MODELO
# ============================================================

model = OmegaNetwork(
    input_size=N_POINTS,
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# GENERAR SEÑALES
# ============================================================

signals = []

for omega in omegas_test:

    signal = simulate_oscillator(
        t,
        omega
    )

    signals.append(signal)


signals = torch.stack(
    signals
)


# ============================================================
# PREDICCIONES
# ============================================================

with torch.no_grad():

    omega_pred = model(
        signals
    ).squeeze(1)


# ============================================================
# MOSTRAR ALGUNOS RESULTADOS
# ============================================================

print()
print("=" * 60)
print("MAPEO DE LA RED: ω REAL → ω PREDICHO")
print("=" * 60)

print()

for real, pred in zip(
    omegas_test,
    omega_pred
):

    print(
        f"ω real = {real.item():6.2f} | "
        f"ω pred = {pred.item():6.2f} | "
        f"error = "
        f"{pred.item() - real.item():+7.2f}"
    )


# ============================================================
# GRÁFICA
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    omegas_test.numpy(),
    omegas_test.numpy(),
    label="Ideal: ω pred = ω real"
)

plt.plot(
    omegas_test.numpy(),
    omega_pred.numpy(),
    label="OmegaNetwork"
)

plt.xlabel(
    "ω real"
)

plt.ylabel(
    "ω pred"
)

plt.title(
    "Mapeo aprendido por OmegaNetwork"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()