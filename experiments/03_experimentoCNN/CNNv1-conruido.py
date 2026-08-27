import torch

from pinns.models.cnn_spectrum_km import SpectrumCNN
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

N_POINTS = 4000
T_TOTAL = 20.0

N_SYSTEMS = 100
TRAIN_RATIO = 0.80

N_TRAIN = int(
    N_SYSTEMS * TRAIN_RATIO
)

N_TEST = (
    N_SYSTEMS - N_TRAIN
)


# ============================================================
# FÍSICA
#
# True  -> Lk + Lm + Lphys
# False -> Lk + Lm
# ============================================================

USE_PHYSICS = False

# ============================================================
# RUIDO
#
# 15% de la desviación estándar de cada señal
# ============================================================

NOISE_LEVEL = 0.15
NOISE_SEED = 1515


# ============================================================
# RANGOS FÍSICOS
# ============================================================

OMEGA_MIN = 5.0
OMEGA_MAX = 100.0

K_MIN = 25.0
K_MAX = 500.0

M_MIN = 0.05
M_MAX = 1.0


# ============================================================
# RANGO ESPECTRAL ENTREGADO A LA CNN
# ============================================================

CNN_OMEGA_MAX = 120.0


# ============================================================
# ENTRENAMIENTO
# ============================================================

EPOCHS = 2000

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


# ============================================================
# SEMILLA GENERAL
# ============================================================

SEED = 42

torch.manual_seed(
    SEED
)


# ============================================================
# MODEL PATH
# ============================================================

if USE_PHYSICS:

    MODEL_PATH = (
        f"cnn_spectrum_dataset{N_SYSTEMS}_"
        f"ruido15_con_fisica.pth"
    )

else:

    MODEL_PATH = (
        f"cnn_spectrum_dataset{N_SYSTEMS}_"
        f"ruido15_sin_fisica.pth"
    )


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
print("GENERACIÓN DATASET CNN - X(f) CON RUIDO")
print("============================================================")


# ------------------------------------------------------------
# GENERAR ω
# ------------------------------------------------------------

omega_real = (
    OMEGA_MIN
    +
    torch.rand(
        N_SYSTEMS
    )
    *
    (
        OMEGA_MAX
        -
        OMEGA_MIN
    )
)


# ------------------------------------------------------------
# NORMALIZAR ω A [0,1]
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
# RELACIÓN NO LINEAL ω -> k
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
# m = k / omega²
# ------------------------------------------------------------

m_values = (
    k_values
    /
    omega_real ** 2
)


# ============================================================
# VECTOR TEMPORAL
# ============================================================

t = torch.linspace(
    0.0,
    T_TOTAL,
    N_POINTS
)

dt = (
    t[1]
    -
    t[0]
)


# ============================================================
# INFORMACIÓN DE MUESTREO
# ============================================================

sampling_frequency = (
    1.0
    /
    dt.item()
)

nyquist_frequency = (
    sampling_frequency
    /
    2.0
)


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


delta_omega = (
    omega_bins[1]
    -
    omega_bins[0]
)


print()
print("------------------------------------------------------------")
print("MUESTREO")
print("------------------------------------------------------------")

print(
    f"dt                  = "
    f"{dt.item():.6f} s"
)

print(
    f"Frecuencia muestreo = "
    f"{sampling_frequency:.4f} Hz"
)

print(
    f"Frecuencia Nyquist  = "
    f"{nyquist_frequency:.4f} Hz"
)

print(
    f"Resolución FFT Δω   = "
    f"{delta_omega.item():.4f} rad/s"
)


# ============================================================
# GENERAR SEÑALES LIMPIAS x(t)
# ============================================================

signals_clean = []


for omega in omega_real:

    x = simulate_oscillator(
        t,
        omega,
        x0,
        v0
    )

    signals_clean.append(
        x
    )


signals_clean = torch.stack(
    signals_clean
)


# ============================================================
# AGREGAR RUIDO GAUSSIANO
#
# sigma_ruido =
# NOISE_LEVEL * sigma_señal
# ============================================================

noise_generator = torch.Generator()

noise_generator.manual_seed(
    NOISE_SEED
)


signal_std = signals_clean.std(
    dim=1,
    keepdim=True
)


noise = torch.randn(
    signals_clean.shape,
    generator=noise_generator,
    dtype=signals_clean.dtype
)


signals_noisy = (
    signals_clean
    +
    NOISE_LEVEL
    *
    signal_std
    *
    noise
)


# ============================================================
# DIAGNÓSTICO DEL RUIDO
# ============================================================

mean_signal_difference = torch.mean(
    torch.abs(
        signals_noisy
        -
        signals_clean
    )
)


print()
print("------------------------------------------------------------")
print("RUIDO")
print("------------------------------------------------------------")

print(
    f"Nivel de ruido = "
    f"{NOISE_LEVEL * 100:.0f}%"
)

print(
    f"Diferencia media |x_ruido - x_limpia| = "
    f"{mean_signal_difference.item():.6f}"
)


# ============================================================
# FFT DE LA SEÑAL RUIDOSA
#
# ESTA ES LA QUE ENTRA A LA CNN
# ============================================================

with torch.no_grad():

    spectrum_complex = torch.fft.rfft(
        signals_noisy,
        dim=1
    )

    spectrum_magnitude = torch.abs(
        spectrum_complex
    )


# ============================================================
# FFT DE LA SEÑAL LIMPIA
#
# SOLO PARA DIAGNÓSTICO
# ============================================================

with torch.no_grad():

    spectrum_clean_complex = torch.fft.rfft(
        signals_clean,
        dim=1
    )

    spectrum_clean_magnitude = torch.abs(
        spectrum_clean_complex
    )


# ============================================================
# ELIMINAR COMPONENTE DC
# ============================================================

spectrum_magnitude = (
    spectrum_magnitude[
        :, 1:
    ]
)

spectrum_clean_magnitude = (
    spectrum_clean_magnitude[
        :, 1:
    ]
)

omega_bins_no_dc = (
    omega_bins[
        1:
    ]
)


# ============================================================
# OBTENER ωFFT DE SEÑAL RUIDOSA
#
# IMPORTANTE:
# ωFFT NO entra como input a la CNN.
#
# Solo se usa para Loss_phys y métricas.
# ============================================================

with torch.no_grad():

    peak_index_noisy = torch.argmax(
        spectrum_magnitude,
        dim=1
    )

    omega_fft = (
        omega_bins_no_dc[
            peak_index_noisy
        ]
    )


# ============================================================
# OBTENER ωFFT DE SEÑAL LIMPIA
#
# SOLO PARA VER CUÁNTO CAMBIÓ EL PICO
# ============================================================

with torch.no_grad():

    peak_index_clean = torch.argmax(
        spectrum_clean_magnitude,
        dim=1
    )

    omega_fft_clean = (
        omega_bins_no_dc[
            peak_index_clean
        ]
    )


# ============================================================
# DIAGNÓSTICO DEL EFECTO DEL RUIDO EN EL PICO FFT
# ============================================================

different_fft = torch.sum(
    omega_fft
    !=
    omega_fft_clean
)


mean_fft_difference = torch.mean(
    torch.abs(
        omega_fft
        -
        omega_fft_clean
    )
)


print(
    f"Picos FFT modificados = "
    f"{different_fft.item()} / {N_SYSTEMS}"
)

print(
    f"Diferencia media ωFFT ruido/limpia = "
    f"{mean_fft_difference.item():.6f} rad/s"
)


# ============================================================
# RECORTAR ESPECTRO
#
# SOLO:
# 0 < ω <= 120 rad/s
# ============================================================

cnn_mask = (
    omega_bins_no_dc
    <=
    CNN_OMEGA_MAX
)


spectrum_cnn = (
    spectrum_magnitude[
        :,
        cnn_mask
    ]
)


omega_bins_cnn = (
    omega_bins_no_dc[
        cnn_mask
    ]
)


# ============================================================
# NORMALIZAR CADA ESPECTRO
#
# pico máximo = 1
#
# IMPORTANTE:
# estamos normalizando el espectro RUIDOSO
# ============================================================

spectrum_max = (
    spectrum_cnn.max(
        dim=1,
        keepdim=True
    ).values
)


spectrum_cnn = (
    spectrum_cnn
    /
    (
        spectrum_max
        +
        1e-8
    )
)


# ============================================================
# SPLIT TRAIN / TEST
#
# MISMO SPLIT PARA CON Y SIN FÍSICA
# SI SE USA LA MISMA SEED
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
# TRAIN
# ============================================================

spectrum_train = (
    spectrum_cnn[
        train_indices
    ]
)

k_train = (
    k_values[
        train_indices
    ]
)

m_train = (
    m_values[
        train_indices
    ]
)

omega_train = (
    omega_real[
        train_indices
    ]
)

omega_fft_train = (
    omega_fft[
        train_indices
    ]
)


# ============================================================
# TEST
# ============================================================

spectrum_test = (
    spectrum_cnn[
        test_indices
    ]
)

k_test = (
    k_values[
        test_indices
    ]
)

m_test = (
    m_values[
        test_indices
    ]
)

omega_test = (
    omega_real[
        test_indices
    ]
)

omega_fft_test = (
    omega_fft[
        test_indices
    ]
)


# ============================================================
# NORMALIZAR TARGETS
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
# FORMATO PARA CONV1D
#
# [batch, channel, length]
# ============================================================

spectrum_train = (
    spectrum_train.unsqueeze(1)
)

spectrum_test = (
    spectrum_test.unsqueeze(1)
)


# ============================================================
# INFORMACIÓN DEL DATASET
# ============================================================

print()
print("============================================================")
print("INFORMACIÓN DEL DATASET")
print("============================================================")

print(
    f"Sistemas totales = "
    f"{N_SYSTEMS}"
)

print(
    f"Train            = "
    f"{N_TRAIN}"
)

print(
    f"Test             = "
    f"{N_TEST}"
)

print()

print(
    f"ω rango = "
    f"[{omega_real.min().item():.4f}, "
    f"{omega_real.max().item():.4f}]"
)

print(
    f"k rango = "
    f"[{k_values.min().item():.4f}, "
    f"{k_values.max().item():.4f}]"
)

print(
    f"m rango = "
    f"[{m_values.min().item():.4f}, "
    f"{m_values.max().item():.4f}]"
)

print()

print(
    f"Bins entregados a CNN = "
    f"{spectrum_train.shape[2]}"
)

print(
    f"Forma entrada CNN train = "
    f"{tuple(spectrum_train.shape)}"
)

print()

print(
    f"Ruido = "
    f"{NOISE_LEVEL * 100:.0f}%"
)

print(
    f"Física = "
    f"{'Sí' if USE_PHYSICS else 'No'}"
)


# ============================================================
# RESEED ANTES DE CREAR EL MODELO
#
# Esto hace que con/sin física arranquen
# exactamente con los mismos pesos iniciales.
# ============================================================

MODEL_SEED = 1234

torch.manual_seed(
    MODEL_SEED
)


# ============================================================
# MODELO
# ============================================================

model = SpectrumCNN()


# ============================================================
# OPTIMIZADOR
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# ENTRENAMIENTO
# ============================================================

print()
print("============================================================")

if USE_PHYSICS:

    print(
        "CNN X(f) - RUIDO 15% - CON FÍSICA"
    )

else:

    print(
        "CNN X(f) - RUIDO 15% - SIN FÍSICA"
    )

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

    model.train()

    optimizer.zero_grad()


    # --------------------------------------------------------
    # FORWARD CNN
    # --------------------------------------------------------

    k_pred_norm, m_pred_norm = model(
        spectrum_train
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
    # RELACIÓN FÍSICA
    # --------------------------------------------------------

    omega_pred = torch.sqrt(
        k_pred
        /
        m_pred
    )


    # --------------------------------------------------------
    # LOSS FÍSICA
    #
    # TODAVÍA ESTAMOS EN CNN v1:
    #
    # omega_pred vs omega_fft
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

    if USE_PHYSICS:

        loss = (
            loss_k
            +
            loss_m
            +
            loss_phys
        )

    else:

        loss = (
            loss_k
            +
            loss_m
        )


    # --------------------------------------------------------
    # BACKPROP
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # ACTUALIZAR PESOS
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # MOSTRAR
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
# TEST
# ============================================================

model.eval()


with torch.no_grad():

    k_test_pred_norm, m_test_pred_norm = model(
        spectrum_test
    )


    k_test_pred_norm = (
        k_test_pred_norm.squeeze(1)
    )

    m_test_pred_norm = (
        m_test_pred_norm.squeeze(1)
    )


    # --------------------------------------------------------
    # DESNORMALIZAR
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ω PREDICHA
    # --------------------------------------------------------

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
# MUESTRA DE RESULTADOS TEST
# ============================================================

print()
print("============================================================")
print("MUESTRA DE RESULTADOS TEST")
print("============================================================")

print()

print(
    "ω real      ω FFT      ω pred      "
    "k real      k pred      "
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
        f"{k_test[i].item():9.3f}    "
        f"{k_test_pred[i].item():9.3f}    "
        f"{m_test[i].item():8.3f}    "
        f"{m_test_pred[i].item():8.3f}"
    )


# ============================================================
# RESULTADOS GLOBALES
# ============================================================

print()
print("============================================================")
print("ERROR MEDIO TEST")
print("============================================================")

print(
    f"Ruido                     = "
    f"{NOISE_LEVEL * 100:.0f}%"
)

print(
    f"Física                    = "
    f"{'Sí' if USE_PHYSICS else 'No'}"
)

print()

print(
    f"Error medio relativo k      = "
    f"{k_test_error.item():.4f}%"
)

print(
    f"Error medio relativo m      = "
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
# GUARDAR MODELO
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "n_systems":
        N_SYSTEMS,

    "use_physics":
        USE_PHYSICS,

    "noise_level":
        NOISE_LEVEL,

    "noise_seed":
        NOISE_SEED,

    "model_seed":
        MODEL_SEED,

    "omega_min":
        OMEGA_MIN,

    "omega_max":
        OMEGA_MAX,

    "k_min":
        K_MIN,

    "k_max":
        K_MAX,

    "m_min":
        M_MIN,

    "m_max":
        M_MAX,

    "cnn_omega_max":
        CNN_OMEGA_MAX,

    "error_k":
        k_test_error.item(),

    "error_m":
        m_test_error.item(),

    "error_omega":
        omega_test_error.item(),

    "error_fft":
        omega_fft_error.item()
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