import torch
import torch.nn as nn

from pinns.models.omega_network_etapa1 import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_POINTS = 1000

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0

T_TOTAL = 10.0

EPOCHS = 1000
LEARNING_RATE = 1e-3

MODEL_PATH = "omega_network_etapa1.pth"

# ============================================================
# SEMILLA
# ============================================================

torch.manual_seed(42)


# ============================================================
# DATASET
# ============================================================

omegas = torch.tensor([
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0
])


# ============================================================
# CONDICIONES INICIALES
# ============================================================

x0 = torch.tensor(1.0)
v0 = torch.tensor(0.0)


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
# GENERAR SEÑALES
# ============================================================

signals = []

for omega in omegas:

    # --------------------------------------------------------
    # ω NUEVO → SIMULADOR
    # --------------------------------------------------------

    x = simulate_oscillator(
        t,
        omega,
        x0,
        v0
    )

    signals.append(x)


signals = torch.stack(signals)


# ============================================================
# FFT
# ============================================================

def fft_peak_omega(signals):

    spectrum = torch.fft.rfft(
        signals
    )

    magnitude = torch.abs(
        spectrum
    )

    # --------------------------------------------------------
    # IGNORAR DC
    # --------------------------------------------------------

    peak_index = torch.argmax(
        magnitude[:, 1:],
        dim=1
    ) + 1

    omega_peak = omega_bins[
        peak_index
    ]

    return omega_peak


# ============================================================
# OBTENER ω FFT
# ============================================================

with torch.no_grad():

    omega_fft = fft_peak_omega(
        signals
    )


# ============================================================
# MOSTRAR FFT
# ============================================================

print()
print("============================================================")
print("ETAPA 1 - FFT")
print("============================================================")

print()
print(
    "ω real        ω FFT"
)

print(
    "--------------------------------------------"
)

for real, fft_value in zip(
    omegas,
    omega_fft
):

    print(
        f"{real.item():8.2f}        "
        f"{fft_value.item():8.4f}"
    )


# ============================================================
# MODELO
# ============================================================

model = OmegaNetwork(
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX
)


# ============================================================
# OPTIMIZADOR
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# LOSS
# ============================================================

loss_function = nn.MSELoss()


# ============================================================
# PREDICCIONES INICIALES
# ============================================================

with torch.no_grad():

    omega_initial = model(
        omega_fft.unsqueeze(1)
    ).squeeze(1)


print()
print("============================================================")
print("PREDICCIONES INICIALES")
print("============================================================")

print()
print(
    "ω real        ω FFT        ω inicial"
)

print(
    "--------------------------------------------"
)

for real, fft_value, pred in zip(
    omegas,
    omega_fft,
    omega_initial
):

    print(
        f"{real.item():8.2f}        "
        f"{fft_value.item():8.4f}        "
        f"{pred.item():10.4f}"
    )


# ============================================================
# ENTRENAMIENTO
# ============================================================

print()
print("============================================================")
print("ENTRENAMIENTO ETAPA 1")
print("============================================================")

print()
print(
    "Epoch        Loss"
)

print(
    "--------------------------------------------"
)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # LIMPIAR GRADIENTES
    # --------------------------------------------------------

    optimizer.zero_grad()


    # --------------------------------------------------------
    # ω FFT → OMEGANETWORK → ω PRED
    # --------------------------------------------------------

    omega_pred = model(
        omega_fft.unsqueeze(1)
    ).squeeze(1)


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = loss_function(
        omega_pred,
        omegas
    )


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # ACTUALIZACIÓN
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # MOSTRAR
    # --------------------------------------------------------

    if epoch % 100 == 0:

        print(
            f"{epoch:5d}        "
            f"{loss.item():16.8f}"
        )


# ============================================================
# RESULTADOS FINALES
# ============================================================

with torch.no_grad():

    omega_final = model(
        omega_fft.unsqueeze(1)
    ).squeeze(1)


print()
print("============================================================")
print("RESULTADOS FINALES ETAPA 1")
print("============================================================")

print()
print(
    "ω real        ω FFT        ω pred        error"
)

print(
    "------------------------------------------------------------"
)

for real, fft_value, pred in zip(
    omegas,
    omega_fft,
    omega_final
):

    error = (
        pred.item()
        -
        real.item()
    )

    print(
        f"{real.item():8.2f}        "
        f"{fft_value.item():8.4f}        "
        f"{pred.item():10.4f}        "
        f"{error:+10.4f}"
    )


# ============================================================
# GUARDAR MODELO
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print()
print(
    "Modelo guardado en: "
    f"{MODEL_PATH}"
)