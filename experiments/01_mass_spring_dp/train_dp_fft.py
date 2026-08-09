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

# Temperatura del soft-argmax
SOFTMAX_TEMPERATURE = 0.05


# ----------------------------------
# DATASET
# ----------------------------------

omegas = torch.arange(
    5.0,
    50.1,
    1.0
)

# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(
    0,
    10,
    N_POINTS
)

dt = t[1] - t[0]


# ----------------------------------
# FRECUENCIAS DE LA FFT
# ----------------------------------

frequencies = torch.fft.rfftfreq(
    N_POINTS,
    d=dt
)

omega_bins = (
    2.0
    * torch.pi
    * frequencies
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
# FUNCIÓN PARA OBTENER
# FRECUENCIA DOMINANTE DIFERENCIABLE
# ----------------------------------

def soft_peak_omega(
    signals,
    temperature=0.05
):

    # FFT real-valued
    spectrum = torch.fft.rfft(
        signals
    )

    # Magnitud
    magnitude = torch.abs(
        spectrum
    )

    # Normalizar cada espectro
    magnitude = (
        magnitude
        /
        (
            magnitude.max(
                dim=1,
                keepdim=True
            ).values
            + 1e-8
        )
    )

    # Soft-argmax
    weights = torch.softmax(
        magnitude / temperature,
        dim=1
    )

    # Frecuencia angular esperada
    omega_estimated = (
        weights
        * omega_bins
    ).sum(
        dim=1
    )

    return omega_estimated


# ----------------------------------
# FRECUENCIA DOMINANTE REAL
# ----------------------------------

with torch.no_grad():

    omega_fft_real = soft_peak_omega(
        signals,
        SOFTMAX_TEMPERATURE
    )


print("\nFrecuencias dominantes reales:")

for real, fft_value in zip(
    omegas,
    omega_fft_real
):

    print(
        f"ω real = {real.item():6.2f} | "
        f"ω FFT = {fft_value.item():6.2f}"
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
    # FRECUENCIA DOMINANTE PREDICHA
    # ----------------------------------

    omega_fft_pred = soft_peak_omega(
        predicted_signals,
        SOFTMAX_TEMPERATURE
    )


    # ----------------------------------
    # LOSS DE FRECUENCIA DOMINANTE
    # ----------------------------------

    loss = loss_function(
        omega_fft_pred,
        omega_fft_real
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
            f"Loss Peak FFT = {loss.item():.6f}"
        )


# ----------------------------------
# RESULTADOS FINALES
# ----------------------------------

with torch.no_grad():

    omega_final = model(
        signals
    ).squeeze(1)

    omega_fft_final = soft_peak_omega(
        predicted_signals,
        SOFTMAX_TEMPERATURE
    )


print("\nResultados:")

for real, pred, fft_pred in zip(
    omegas,
    omega_final,
    omega_fft_final
):

    print(
        f"ω real = {real.item():6.2f} | "
        f"ω pred = {pred.item():6.2f} | "
        f"ω pico FFT = {fft_pred.item():6.2f}"
    )

# ----------------------------------
# GUARDAR MODELO ENTRENADO
# ----------------------------------

torch.save(
    model.state_dict(),
    "omega_network_fft.pth"
)

print(
    "\nModelo guardado en: "
    "omega_network_fft.pth"
)