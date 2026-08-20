import torch

from pinns.models.omega_network_identificoKyM import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_POINTS = 1000
T_TOTAL = 10.0

N_SYSTEMS = 40
N_TRAIN = 32
N_TEST = 8

M_MIN = 0.5
M_MAX = 3.0

K_MIN = 25.0
K_MAX = 1600.0

MIN_OMEGA_SEPARATION = 0.5

EPOCHS = 2000
LEARNING_RATE = 1e-3

MODEL_PATH = "omega_network_identificoKyM_etapa3_normalizado.pth"


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
print("GENERACIÓN DEL DATASET")
print("============================================================")


k_values = []
m_values = []
omega_values = []


while len(k_values) < N_SYSTEMS:

    # --------------------------------------------------------
    # Generar candidato
    # --------------------------------------------------------

    m_candidate = (
        M_MIN
        +
        (M_MAX - M_MIN)
        * torch.rand(1).item()
    )

    k_candidate = (
        K_MIN
        +
        (K_MAX - K_MIN)
        * torch.rand(1).item()
    )


    # --------------------------------------------------------
    # Frecuencia natural
    # --------------------------------------------------------

    omega_candidate = (
        k_candidate
        /
        m_candidate
    ) ** 0.5


    # --------------------------------------------------------
    # Verificar separación
    # --------------------------------------------------------

    valid = True

    for omega_existing in omega_values:

        if abs(
            omega_candidate
            -
            omega_existing
        ) < MIN_OMEGA_SEPARATION:

            valid = False
            break


    # --------------------------------------------------------
    # Guardar sistema
    # --------------------------------------------------------

    if valid:

        k_values.append(
            k_candidate
        )

        m_values.append(
            m_candidate
        )

        omega_values.append(
            omega_candidate
        )


# ============================================================
# CONVERTIR A TENSORES
# ============================================================

k_values = torch.tensor(
    k_values,
    dtype=torch.float32
)

m_values = torch.tensor(
    m_values,
    dtype=torch.float32
)

omega_real = torch.tensor(
    omega_values,
    dtype=torch.float32
)


# ============================================================
# ORDENAR DATASET POR FRECUENCIA
# ============================================================

sort_index = torch.argsort(
    omega_real
)

k_values = k_values[
    sort_index
]

m_values = m_values[
    sort_index
]

omega_real = omega_real[
    sort_index
]


# ============================================================
# SEPARACIÓN TRAIN / TEST
# ============================================================

k_train = k_values[
    :N_TRAIN
]

m_train = m_values[
    :N_TRAIN
]

omega_train = omega_real[
    :N_TRAIN
]


k_test = k_values[
    N_TRAIN:
]

m_test = m_values[
    N_TRAIN:
]

omega_test = omega_real[
    N_TRAIN:
]


# ============================================================
# NORMALIZACIÓN DE LOS TARGETS
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
# OBTENER PICO FFT
# ============================================================

def fft_peak_omega(signals):

    spectrum = torch.fft.rfft(
        signals
    )

    magnitude = torch.abs(
        spectrum
    )

    # Ignorar componente DC

    peak_index = torch.argmax(
        magnitude[:, 1:],
        dim=1
    ) + 1

    omega_peak = omega_bins[
        peak_index
    ]

    return omega_peak


# ============================================================
# FRECUENCIA OBSERVADA
# ============================================================

with torch.no_grad():

    omega_fft = fft_peak_omega(
        signals
    )


# ============================================================
# SEPARAR FFT TRAIN / TEST
# ============================================================

omega_fft_train = omega_fft[
    :N_TRAIN
]

omega_fft_test = omega_fft[
    N_TRAIN:
]


# ============================================================
# INFORMACIÓN DEL DATASET
# ============================================================

print()
print("============================================================")
print("SISTEMAS DE ENTRENAMIENTO")
print("============================================================")

print()
print(
    "k real        m real        ω real        ω FFT"
)

print(
    "------------------------------------------------------------"
)

for k, m, omega, omega_f in zip(
    k_train,
    m_train,
    omega_train,
    omega_fft_train
):

    print(
        f"{k.item():8.2f}        "
        f"{m.item():8.4f}        "
        f"{omega.item():8.4f}        "
        f"{omega_f.item():8.4f}"
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
print("ENTRENAMIENTO ETAPA 3")
print("============================================================")

print()
print(
    "Epoch        Loss total        Loss k        Loss m"
)

print(
    "------------------------------------------------------------"
)


for epoch in range(
    EPOCHS
):

    optimizer.zero_grad()


    # --------------------------------------------------------
    # PREDICCIÓN NORMALIZADA
    # --------------------------------------------------------

    k_pred_norm, m_pred_norm = model(
        omega_fft_train.unsqueeze(1)
    )


    k_pred_norm = (
        k_pred_norm.squeeze(1)
    )

    m_pred_norm = (
        m_pred_norm.squeeze(1)
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
            f"{loss_m.item():10.8f}"
        )


# ============================================================
# PREDICCIONES FINALES TRAIN
# ============================================================

with torch.no_grad():

    k_train_pred_norm, m_train_pred_norm = model(
        omega_fft_train.unsqueeze(1)
    )


k_train_pred_norm = (
    k_train_pred_norm.squeeze(1)
)

m_train_pred_norm = (
    m_train_pred_norm.squeeze(1)
)


# ============================================================
# DESNORMALIZAR TRAIN
# ============================================================

k_train_pred = (
    K_MIN
    +
    k_train_pred_norm
    *
    (
        K_MAX
        -
        K_MIN
    )
)

m_train_pred = (
    M_MIN
    +
    m_train_pred_norm
    *
    (
        M_MAX
        -
        M_MIN
    )
)


# ============================================================
# PREDICCIONES FINALES TEST
# ============================================================

with torch.no_grad():

    k_test_pred_norm, m_test_pred_norm = model(
        omega_fft_test.unsqueeze(1)
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
# RESULTADOS TEST
# ============================================================

print()
print("============================================================")
print("RESULTADOS TEST ETAPA 3")
print("============================================================")

print()
print(
    "ω real        ω FFT        "
    "k real        k pred        "
    "m real        m pred"
)

print(
    "--------------------------------------------------------------------------"
)

for omega_real_value, omega_fft_value, \
    k_real, k_pred_value, \
    m_real, m_pred_value in zip(
        omega_test,
        omega_fft_test,
        k_test,
        k_test_pred,
        m_test,
        m_test_pred
    ):

    print(
        f"{omega_real_value.item():8.4f}        "
        f"{omega_fft_value.item():8.4f}        "
        f"{k_real.item():8.4f}        "
        f"{k_pred_value.item():8.4f}        "
        f"{m_real.item():8.4f}        "
        f"{m_pred_value.item():8.4f}"
    )


# ============================================================
# ERRORES MEDIOS
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


# ============================================================
# FRECUENCIA RECONSTRUIDA
# ============================================================

with torch.no_grad():

    omega_test_pred = torch.sqrt(
        k_test_pred
        /
        m_test_pred
    )


# ============================================================
# ERROR DE FRECUENCIA
# ============================================================

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
# MOSTRAR ERRORES
# ============================================================

print()
print("============================================================")
print("ERROR MEDIO TEST")
print("============================================================")

print(
    f"Error medio relativo k = "
    f"{k_test_error.item():.4f}%"
)

print(
    f"Error medio relativo m = "
    f"{m_test_error.item():.4f}%"
)

print(
    f"Error medio relativo ω = "
    f"{omega_test_error.item():.4f}%"
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