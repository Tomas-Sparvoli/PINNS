import torch

from pinns.models.omega_network_identificoKyM import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_POINTS = 4000
T_TOTAL = 20.0

N_SYSTEMS = 1000
TRAIN_RATIO = 0.80

N_TRAIN = int(
    N_SYSTEMS * TRAIN_RATIO
)

N_TEST = (
    N_SYSTEMS - N_TRAIN
)


# ============================================================
# RANGO DE FRECUENCIA
# ============================================================

OMEGA_MIN = 5.0
OMEGA_MAX = 100.0


# ============================================================
# RANGO DE RIGIDEZ
# ============================================================

K_MIN = 25.0
K_MAX = 500.0


# ============================================================
# RANGO DE MASA
#
# Se obtiene indirectamente a partir de:
#
# m = k / omega^2
#
# Para la relación definida debajo queda aproximadamente:
#
# 0.5556 <= m <= 1.0
# ============================================================

M_MIN = 0.05
M_MAX = 1.0


# ============================================================
# ENTRENAMIENTO
# ============================================================

EPOCHS = 2000

LEARNING_RATE = 1e-3


MODEL_PATH = (
    "omega_network_etapa3_dataset1000_omega100_sin_fisica.pth"
)


# ============================================================
# SEMILLA
# ============================================================

torch.manual_seed(42)


# ============================================================
# CONDICIONES INICIALES
# ============================================================

x0 = torch.tensor(1.0)

v0 = torch.tensor(0.0)


# ============================================================
# GENERACIÓN DEL DATASET
# ============================================================

print()
print("============================================================")
print("GENERACIÓN DEL DATASET - RELACIÓN NO LINEAL")
print("============================================================")


# ------------------------------------------------------------
# GENERAR ω
#
# ω ∈ [5, 30] rad/s
# ------------------------------------------------------------

omega_real = (
    OMEGA_MIN
    +
    torch.rand(N_SYSTEMS)
    *
    (
        OMEGA_MAX
        -
        OMEGA_MIN
    )
)


# ------------------------------------------------------------
# NORMALIZAR ω AL INTERVALO [0,1]
#
# z = (ω - ω_min) / (ω_max - ω_min)
# ------------------------------------------------------------

z = (
    omega_real
    -
    OMEGA_MIN
) / (
    OMEGA_MAX
    -
    OMEGA_MIN
)


# ------------------------------------------------------------
# RELACIÓN NO LINEAL ENTRE ω Y k
#
# k = K_MIN +
#     (K_MAX - K_MIN) * (0.3*z + 0.7*z²)
#
# Para:
#
# z = 0  -> k = 25
# z = 1  -> k = 500
#
# La relación es creciente pero NO lineal.
# ------------------------------------------------------------

k_values = (
    K_MIN
    +
    (
        K_MAX
        -
        K_MIN
    )
    *
    (
        0.3 * z
        +
        0.7 * z ** 2
    )
)


# ------------------------------------------------------------
# CALCULAR m A PARTIR DE LA FÍSICA
#
# omega = sqrt(k/m)
#
# omega² = k/m
#
# m = k / omega²
# ------------------------------------------------------------

m_values = (
    k_values
    /
    omega_real ** 2
)


# ============================================================
# TIEMPO
# ============================================================

t = torch.linspace(
    0.0,
    T_TOTAL,
    N_POINTS
)

dt = t[1] - t[0]


# ============================================================
# INFORMACIÓN DEL MUESTREO
# ============================================================

sampling_frequency = (
    1.0
    /
    dt
)

nyquist_frequency = (
    sampling_frequency
    /
    2.0
)

delta_f = (
    1.0
    /
    T_TOTAL
)

delta_omega = (
    2.0
    *
    torch.pi
    *
    delta_f
)


print()
print("------------------------------------------------------------")
print("MUESTREO")
print("------------------------------------------------------------")

print(
    f"dt                = "
    f"{dt.item():.6f} s"
)

print(
    f"Frecuencia muestreo = "
    f"{sampling_frequency.item():.4f} Hz"
)

print(
    f"Frecuencia Nyquist  = "
    f"{nyquist_frequency.item():.4f} Hz"
)

print(
    f"Resolución FFT Δω   = "
    f"{delta_omega:.4f} rad/s"
)


# ============================================================
# FRECUENCIAS DE LA FFT
# ============================================================

frequencies = torch.fft.rfftfreq(
    N_POINTS,
    d=dt.item()
)

omega_bins = (
    2.0
    *
    torch.pi
    *
    frequencies
)


# ============================================================
# GENERAR SEÑALES
# ============================================================

signals = []


for omega in omega_real:

    x = simulate_oscillator(
        t,
        omega,
        x0,
        v0
    )

    signals.append(
        x
    )


signals = torch.stack(
    signals
)


# ============================================================
# FUNCIÓN FFT
# ============================================================

def fft_peak_omega(signals):

    spectrum = torch.fft.rfft(
        signals,
        dim=1
    )

    magnitude = torch.abs(
        spectrum
    )


    # --------------------------------------------------------
    # Ignorar componente DC
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
# ω OBTENIDA MEDIANTE FFT
# ============================================================

with torch.no_grad():

    omega_fft = fft_peak_omega(
        signals
    )


# ============================================================
# SPLIT ALEATORIO TRAIN / TEST
# ============================================================

indices = torch.randperm(
    N_SYSTEMS
)


train_indices = indices[
    :N_TRAIN
]

test_indices = indices[
    N_TRAIN:
]


# ============================================================
# DATOS TRAIN
# ============================================================

k_train = k_values[
    train_indices
]

m_train = m_values[
    train_indices
]

omega_train = omega_real[
    train_indices
]

omega_fft_train = omega_fft[
    train_indices
]


# ============================================================
# DATOS TEST
# ============================================================

k_test = k_values[
    test_indices
]

m_test = m_values[
    test_indices
]

omega_test = omega_real[
    test_indices
]

omega_fft_test = omega_fft[
    test_indices
]


# ============================================================
# NORMALIZACIÓN DE TARGETS k Y m
# ============================================================

k_train_norm = (
    k_train
    -
    K_MIN
) / (
    K_MAX
    -
    K_MIN
)


m_train_norm = (
    m_train
    -
    M_MIN
) / (
    M_MAX
    -
    M_MIN
)


# ============================================================
# NORMALIZACIÓN DE LA ENTRADA ωFFT
#
# Se calculan media y desviación SOLO con TRAIN
# ============================================================

omega_mean = (
    omega_fft_train.mean()
)

omega_std = (
    omega_fft_train.std()
)


omega_fft_train_norm = (
    omega_fft_train
    -
    omega_mean
) / omega_std


omega_fft_test_norm = (
    omega_fft_test
    -
    omega_mean
) / omega_std


# ============================================================
# INFORMACIÓN DEL DATASET
# ============================================================

print()
print("============================================================")
print("INFORMACIÓN DEL DATASET")
print("============================================================")

print(
    f"Sistemas totales = {N_SYSTEMS}"
)

print(
    f"Train            = {N_TRAIN}"
)

print(
    f"Test             = {N_TEST}"
)


print()

print(
    f"ω objetivo = "
    f"[{OMEGA_MIN:.4f}, "
    f"{OMEGA_MAX:.4f}] rad/s"
)

print(
    f"ω real dataset = "
    f"[{omega_real.min().item():.4f}, "
    f"{omega_real.max().item():.4f}] rad/s"
)

print(
    f"ω FFT dataset = "
    f"[{omega_fft.min().item():.4f}, "
    f"{omega_fft.max().item():.4f}] rad/s"
)

print()

print(
    f"m rango = "
    f"[{m_values.min().item():.4f}, "
    f"{m_values.max().item():.4f}]"
)

print(
    f"k rango = "
    f"[{k_values.min().item():.4f}, "
    f"{k_values.max().item():.4f}]"
)

print()

print(
    f"K_MIN normalización = "
    f"{K_MIN:.4f}"
)

print(
    f"K_MAX normalización = "
    f"{K_MAX:.4f}"
)

print()

print(
    f"ω train mean = "
    f"{omega_mean.item():.4f}"
)

print(
    f"ω train std  = "
    f"{omega_std.item():.4f}"
)


# ============================================================
# MODELO
# ============================================================

model = OmegaNetwork()


# ============================================================
# OPTIMIZADOR
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print()
print("============================================================")
print("ENTRENAMIENTO ETAPA 4 - DATASET 1000 - ω 5 A 100")
print("============================================================")

print()

print(
    "Epoch        Loss total        "
    "Loss k        Loss m        Loss phys"
)

print(
    "--------------------------------------------------------------------"
)


for epoch in range(
    EPOCHS
):

    optimizer.zero_grad()


    # --------------------------------------------------------
    # PREDICCIÓN NORMALIZADA
    # --------------------------------------------------------

    k_pred_norm, m_pred_norm = model(
        omega_fft_train_norm.unsqueeze(1)
    )


    k_pred_norm = (
        k_pred_norm.squeeze(1)
    )

    m_pred_norm = (
        m_pred_norm.squeeze(1)
    )


    # --------------------------------------------------------
    # DESNORMALIZAR k
    # --------------------------------------------------------

    k_pred = (
        K_MIN
        +
        k_pred_norm
        *
        (
            K_MAX
            -
            K_MIN
        )
    )


    # --------------------------------------------------------
    # DESNORMALIZAR m
    # --------------------------------------------------------

    m_pred = (
        M_MIN
        +
        m_pred_norm
        *
        (
            M_MAX
            -
            M_MIN
        )
    )


    # --------------------------------------------------------
    # LOSS k
    # --------------------------------------------------------

    loss_k = torch.mean(
        (
            k_pred_norm
            -
            k_train_norm
        ) ** 2
    )


    # --------------------------------------------------------
    # LOSS m
    # --------------------------------------------------------

    loss_m = torch.mean(
        (
            m_pred_norm
            -
            m_train_norm
        ) ** 2
    )


    # --------------------------------------------------------
    # DIFFERENTIABLE PHYSICS
    #
    # omega = sqrt(k/m)
    # --------------------------------------------------------

    omega_pred = torch.sqrt(
        k_pred
        /
        m_pred
    )


    # --------------------------------------------------------
    # PHYSICS LOSS
    # --------------------------------------------------------

    loss_phys = torch.mean(
        (
            (
                omega_pred
                -
                omega_fft_train
            )
            /
            omega_fft_train
        ) ** 2
    )


    # --------------------------------------------------------
    # LOSS TOTAL
    # --------------------------------------------------------

    loss = (
        loss_k
        +
        loss_m
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
    # MOSTRAR LOSS
    # --------------------------------------------------------

    if epoch % 100 == 0:

        print(
            f"{epoch:5d}        "
            f"{loss.item():14.8f}        "
            f"{loss_k.item():10.8f}        "
            f"{loss_m.item():10.8f}        "
            f"{loss_phys.item():10.8f}"
        )


# ============================================================
# PREDICCIONES TEST
# ============================================================

model.eval()


with torch.no_grad():

    k_test_pred_norm, m_test_pred_norm = model(
        omega_fft_test_norm.unsqueeze(1)
    )


k_test_pred_norm = (
    k_test_pred_norm.squeeze(1)
)

m_test_pred_norm = (
    m_test_pred_norm.squeeze(1)
)


# ============================================================
# DESNORMALIZAR TEST
# ============================================================

k_test_pred = (
    K_MIN
    +
    k_test_pred_norm
    *
    (
        K_MAX
        -
        K_MIN
    )
)


m_test_pred = (
    M_MIN
    +
    m_test_pred_norm
    *
    (
        M_MAX
        -
        M_MIN
    )
)


# ============================================================
# ω PREDICHA A PARTIR DE k Y m
# ============================================================

with torch.no_grad():

    omega_test_pred = torch.sqrt(
        k_test_pred
        /
        m_test_pred
    )


# ============================================================
# ERRORES TEST
# ============================================================

k_test_error = torch.mean(
    torch.abs(
        (
            k_test_pred
            -
            k_test
        )
        /
        k_test
    )
) * 100.0


m_test_error = torch.mean(
    torch.abs(
        (
            m_test_pred
            -
            m_test
        )
        /
        m_test
    )
) * 100.0


omega_test_error = torch.mean(
    torch.abs(
        (
            omega_test_pred
            -
            omega_test
        )
        /
        omega_test
    )
) * 100.0


# ============================================================
# ERROR DE LA FFT
# ============================================================

omega_fft_error = torch.mean(
    torch.abs(
        (
            omega_fft_test
            -
            omega_test
        )
        /
        omega_test
    )
) * 100.0


# ============================================================
# ALGUNOS RESULTADOS DE TEST
# ============================================================

print()
print("============================================================")
print("MUESTRA DE RESULTADOS TEST")
print("============================================================")

print()

print(
    "ω real      ω FFT      ω pred      "
    "k real        k pred        "
    "m real      m pred"
)

print(
    "--------------------------------------------------------------------------------------"
)


N_SHOW = min(
    20,
    N_TEST
)


for i in range(
    N_SHOW
):

    print(
        f"{omega_test[i].item():8.3f}    "
        f"{omega_fft_test[i].item():8.3f}    "
        f"{omega_test_pred[i].item():8.3f}    "
        f"{k_test[i].item():11.3f}    "
        f"{k_test_pred[i].item():11.3f}    "
        f"{m_test[i].item():8.3f}    "
        f"{m_test_pred[i].item():8.3f}"
    )


# ============================================================
# RESULTADO GLOBAL
# ============================================================

print()
print("============================================================")
print("ERROR MEDIO TEST")
print("============================================================")

print(
    f"Error medio relativo k     = "
    f"{k_test_error.item():.4f}%"
)

print(
    f"Error medio relativo m     = "
    f"{m_test_error.item():.4f}%"
)

print(
    f"Error medio relativo ω pred = "
    f"{omega_test_error.item():.4f}%"
)

print(
    f"Error medio propio FFT      = "
    f"{omega_fft_error.item():.4f}%"
)


# ============================================================
# GUARDAR MODELO + NORMALIZACIÓN
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "omega_mean":
        omega_mean,

    "omega_std":
        omega_std,

    "omega_min":
        OMEGA_MIN,

    "omega_max":
        OMEGA_MAX,

    "m_min":
        M_MIN,

    "m_max":
        M_MAX,

    "k_min":
        K_MIN,

    "k_max":
        K_MAX,
}


torch.save(
    checkpoint,
    MODEL_PATH
)


print()
print(
    "Modelo guardado en: "
    f"{MODEL_PATH}"
)