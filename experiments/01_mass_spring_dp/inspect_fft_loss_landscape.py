import torch
import matplotlib.pyplot as plt

from pinns.physics.mass_spring import simulate_oscillator


# ----------------------------------
# CONFIGURACIÓN
# ----------------------------------

N_POINTS = 1000

T_TOTAL = 10.0

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0
OMEGA_STEP = 0.1

OMEGA_REAL = 10.0

SOFTMAX_TEMPERATURE = 0.05


# ----------------------------------
# TIEMPO
# ----------------------------------

t = torch.linspace(
    0,
    T_TOTAL,
    N_POINTS
)


dt = t[1] - t[0]


# ----------------------------------
# FRECUENCIAS FFT
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
# SOFT PEAK
# ----------------------------------

def soft_peak_omega(
    signal,
    temperature=0.05
):

    spectrum = torch.fft.rfft(
        signal
    )

    magnitude = torch.abs(
        spectrum
    )

    magnitude = (
        magnitude
        /
        (
            magnitude.max(
                dim=-1,
                keepdim=True
            ).values
            + 1e-8
        )
    )

    weights = torch.softmax(
        magnitude / temperature,
        dim=-1
    )

    omega_estimated = (
        weights
        * omega_bins
    ).sum(
        dim=-1
    )

    return omega_estimated


# ----------------------------------
# SEÑAL REAL
# ----------------------------------

x_real = simulate_oscillator(
    t,
    torch.tensor(OMEGA_REAL)
)


# ----------------------------------
# SOFT PEAK REAL
# ----------------------------------

with torch.no_grad():

    omega_peak_real = soft_peak_omega(
        x_real,
        SOFTMAX_TEMPERATURE
    )


print("\n========================================")
print(f"ω real = {OMEGA_REAL:.2f}")
print("========================================")

print(
    f"ω soft-peak real = "
    f"{omega_peak_real.item():.6f}"
)


# ----------------------------------
# VALORES DE OMEGA A PROBAR
# ----------------------------------

omegas_test = torch.arange(
    OMEGA_MIN,
    OMEGA_MAX + OMEGA_STEP,
    OMEGA_STEP
)


# ----------------------------------
# CALCULAR LOSS
# ----------------------------------

losses = []
gradients = []
peak_values = []


for omega_value in omegas_test:

    omega = torch.tensor(
        omega_value.item(),
        dtype=torch.float32,
        requires_grad=True
    )


    # ----------------------------------
    # SEÑAL PREDICHA
    # ----------------------------------

    x_pred = simulate_oscillator(
        t,
        omega
    )


    # ----------------------------------
    # SOFT PEAK PREDICHO
    # ----------------------------------

    omega_peak_pred = soft_peak_omega(
        x_pred,
        SOFTMAX_TEMPERATURE
    )


    # ----------------------------------
    # LOSS
    # ----------------------------------

    loss = (
        omega_peak_pred
        -
        omega_peak_real
    ) ** 2


    # ----------------------------------
    # GRADIENTE
    # ----------------------------------

    loss.backward()


    losses.append(
        loss.item()
    )

    gradients.append(
        omega.grad.item()
    )

    peak_values.append(
        omega_peak_pred.item()
    )


losses = torch.tensor(
    losses
)

gradients = torch.tensor(
    gradients
)

peak_values = torch.tensor(
    peak_values
)


# ----------------------------------
# MÍNIMOS
# ----------------------------------

min_index = torch.argmin(
    losses
)

print("\nMínimo global encontrado:")

print(
    f"ω = "
    f"{omegas_test[min_index].item():.2f}"
    f" | Loss = "
    f"{losses[min_index].item():.8f}"
)


# ----------------------------------
# DIEZ MENORES LOSSES
# ----------------------------------

sorted_indices = torch.argsort(
    losses
)


print(
    "\nDiez valores de omega "
    "con menor Loss Peak FFT:"
)


for i in range(10):

    index = sorted_indices[i]

    print(
        f"omega = "
        f"{omegas_test[index].item():7.2f}"
        f" | Loss = "
        f"{losses[index].item():.8f}"
    )


# ----------------------------------
# GRADIENTE EN ω = 5 Y ω = 10
# ----------------------------------

print("\nGradientes relevantes:")

for target in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:

    index = torch.argmin(
        torch.abs(
            omegas_test - target
        )
    )

    print(
        f"ω = {omegas_test[index].item():5.2f}"
        f" | Loss = {losses[index].item():10.6f}"
        f" | dLoss/dω = "
        f"{gradients[index].item(): .8f}"
        f" | ω peak = "
        f"{peak_values[index].item():8.4f}"
    )


# ----------------------------------
# GRÁFICO 1: LOSS
# ----------------------------------

plt.figure()

plt.plot(
    omegas_test.numpy(),
    losses.numpy()
)

plt.axvline(
    OMEGA_REAL,
    linestyle="--",
    label=f"ω real = {OMEGA_REAL}"
)

plt.xlabel("ω")
plt.ylabel("Loss Peak FFT")
plt.title(
    "Paisaje de Loss Peak FFT - ω real = 10"
)

plt.grid()
plt.legend()

plt.show()


# ----------------------------------
# GRÁFICO 2: GRADIENTE
# ----------------------------------

plt.figure()

plt.plot(
    omegas_test.numpy(),
    gradients.numpy()
)

plt.axhline(
    0,
    linestyle="--"
)

plt.axvline(
    OMEGA_REAL,
    linestyle="--",
    label=f"ω real = {OMEGA_REAL}"
)

plt.xlabel("ω")
plt.ylabel("dLoss / dω")
plt.title(
    "Gradiente del Loss Peak FFT - ω real = 10"
)

plt.grid()
plt.legend()

plt.show()


# ----------------------------------
# GRÁFICO 3: SOFT PEAK
# ----------------------------------

plt.figure()

plt.plot(
    omegas_test.numpy(),
    peak_values.numpy()
)

plt.axline(
    (OMEGA_MIN, OMEGA_MIN),
    slope=1,
    linestyle="--",
    label="ω peak = ω"
)

plt.xlabel("ω")
plt.ylabel("ω estimado por soft-peak")
plt.title(
    "Respuesta del Soft-Peak FFT"
)

plt.grid()
plt.legend()

plt.show()