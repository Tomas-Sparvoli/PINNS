print("ESTOY EJECUTANDO TEST ETAPA 2 NUEVO")
import torch

from pinns.models.omega_network_etapa2 import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_POINTS = 1000
T_TOTAL = 10.0

MASS = 1.0

# ------------------------------------------------------------
# SISTEMAS DE TEST
#
# Por ahora usamos los mismos sistemas del entrenamiento.
# Esto permite verificar primero que estamos cargando
# correctamente el modelo entrenado.
# ------------------------------------------------------------

OMEGAS_TEST = [
    7.5,
    12.5,
    17.5,
    22.5,
    27.5,
    35.0
]

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0

# ------------------------------------------------------------
# MODELO ENTRENADO EN LA ETAPA 2
# ------------------------------------------------------------

MODEL_PATH = "omega_network_dp_etapa2.pth"


# ============================================================
# SEMILLA
# ============================================================

torch.manual_seed(42)


# ============================================================
# TIEMPO
# ============================================================

t = torch.linspace(
    0,
    T_TOTAL,
    N_POINTS
)


# ============================================================
# CONDICIONES INICIALES
# ============================================================

x0 = torch.tensor(1.0)
v0 = torch.tensor(0.0)


# ============================================================
# MODELO
# ============================================================

model = OmegaNetwork(
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX
)


# ============================================================
# CARGAR MODELO ENTRENADO
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# INFORMACIÓN
# ============================================================

print()
print("============================================================")
print("TEST ETAPA 2")
print("============================================================")

print()
print(
    f"Modelo cargado: {MODEL_PATH}"
)

print()
print(
    "ω real        ω FFT        ω pred        error        error %"
)

print(
    "------------------------------------------------------------"
)


# ============================================================
# EVALUACIÓN
# ============================================================

for omega_real in OMEGAS_TEST:

    # --------------------------------------------------------
    # PARÁMETRO REAL
    # --------------------------------------------------------

    omega_real_tensor = torch.tensor(
        omega_real,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # GENERAR OBSERVACIÓN
    # --------------------------------------------------------

    with torch.no_grad():

        x_obs = simulate_oscillator(
            t,
            omega_real_tensor,
            x0,
            v0
        )


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    dt = float(
        t[1] - t[0]
    )

    frequencies = torch.fft.rfftfreq(
        N_POINTS,
        d=dt
    )

    omega_bins = (
        2.0
        * torch.pi
        * frequencies
    )

    spectrum_obs = torch.abs(
        torch.fft.rfft(
            x_obs
        )
    )


    # --------------------------------------------------------
    # PICO FFT
    # --------------------------------------------------------

    peak_index = (
        torch.argmax(
            spectrum_obs[1:]
        )
        + 1
    )

    omega_fft = omega_bins[
        peak_index
    ]


    # --------------------------------------------------------
    # PREDICCIÓN DE LA RED
    # --------------------------------------------------------

    with torch.no_grad():

        omega_pred = model(
            omega_fft.reshape(1, 1)
        ).squeeze()


    omega_pred = omega_pred.item()


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    error = (
        omega_pred
        -
        omega_real
    )

    error_percent = (
        abs(error)
        /
        omega_real
        *
        100.0
    )


    # --------------------------------------------------------
    # MOSTRAR RESULTADO
    # --------------------------------------------------------

    print(
        f"{omega_real:8.2f}        "
        f"{omega_fft.item():8.4f}        "
        f"{omega_pred:8.4f}        "
        f"{error:+9.4f}        "
        f"{error_percent:8.3f}%"
    )


# ============================================================
# FIN
# ============================================================

print()
print("============================================================")
print("FIN DEL TEST")
print("============================================================")