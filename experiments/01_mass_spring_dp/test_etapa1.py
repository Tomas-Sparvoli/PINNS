import torch

from pinns.models.omega_network_etapa1 import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_POINTS = 1000

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0

T_TOTAL = 10.0

MODEL_PATH = "omega_network_etapa1.pth"


# ============================================================
# VALORES DE PRUEBA
# ============================================================

OMEGAS_TEST = [
    6.70,
    9.40,
    13.20,
    16.80,
    21.30,
    25.70,
    29.40,
    34.60,
    41.80,
    47.30
]


# ============================================================
# TIEMPO
# ============================================================

t = torch.linspace(
    0,
    T_TOTAL,
    N_POINTS
)

dt = t[1] - t[0]


# ============================================================
# FRECUENCIAS DE LA FFT
# ============================================================

frequencies = torch.fft.rfftfreq(
    N_POINTS,
    d=dt
)

omega_bins = (
    2.0
    * torch.pi
    * frequencies
)


# ============================================================
# FUNCIÓN PARA OBTENER EL PICO FFT
# ============================================================

def fft_peak_omega(signal):

    spectrum = torch.fft.rfft(
        signal
    )

    magnitude = torch.abs(
        spectrum
    )

    # Ignoramos la componente DC
    peak_index = torch.argmax(
        magnitude[1:]
    ) + 1

    omega_peak = omega_bins[
        peak_index
    ]

    return omega_peak


# ============================================================
# GENERAR SEÑALES DE PRUEBA
# ============================================================

x0 = torch.tensor(1.0)
v0 = torch.tensor(0.0)

signals = []

for omega in OMEGAS_TEST:

    x = simulate_oscillator(
        t,
        omega,
        x0,
        v0
    )

    signals.append(x)

signals = torch.stack(
    signals
)


# ============================================================
# CARGAR MODELO
# ============================================================

model = OmegaNetwork(
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
# EVALUACIÓN
# ============================================================

print()
print("============================================================")
print("PRUEBA DE GENERALIZACIÓN")
print("============================================================")

print()
print("Valores utilizados para la prueba:")

print(
    ", ".join(
        f"{omega:.2f}"
        for omega in OMEGAS_TEST
    )
)


print()
print("============================================================")
print("RESULTADOS")
print("============================================================")


with torch.no_grad():

    for omega_real, signal in zip(
        OMEGAS_TEST,
        signals
    ):

        # ----------------------------------------------------
        # FFT
        # ----------------------------------------------------

        omega_fft = fft_peak_omega(
            signal
        )


        # ----------------------------------------------------
        # RED
        # ----------------------------------------------------

        omega_pred = model(
            omega_fft.reshape(1, 1)
        ).item()


        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        error = (
            omega_pred
            - omega_real
        )


        print(
            f"ω real = {omega_real:6.2f} | "
            f"ω FFT = {omega_fft.item():6.2f} | "
            f"ω pred = {omega_pred:6.2f} | "
            f"error = {error:+7.2f}"
        )