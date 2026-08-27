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
# USAR FÍSICA
#
# True:
#   Ltotal = Lk + Lm + Lphys
#
# False:
#   Ltotal = Lk + Lm
#
# En CNN v2.1:
#
# Lphys = MSE(x_pred(t), x_real(t))
# ============================================================

USE_PHYSICS = True


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
# VENTANA Y CURRICULUM DE LA FÍSICA
# ============================================================

PHYS_WINDOW_SECONDS = 1.0

PHYS_WARMUP_EPOCHS = 500
PHYS_RAMP_END_EPOCH = 1000
LAMBDA_PHYS_MAX = 0.01


# ============================================================
# SEMILLAS
# ============================================================

DATASET_SEED = 42
MODEL_SEED = 1234


torch.manual_seed(
    DATASET_SEED
)


# ============================================================
# MODEL PATH
# ============================================================

if USE_PHYSICS:

    MODEL_PATH = (
        f"cnn_v2_1_dataset{N_SYSTEMS}_"
        f"con_fisica.pth"
    )

else:

    MODEL_PATH = (
        f"cnn_v2_1_dataset{N_SYSTEMS}_"
        f"sin_fisica.pth"
    )


# ============================================================
# CONDICIONES INICIALES
# ============================================================

X0 = 1.0
V0 = 0.0

x0 = torch.tensor(
    X0
)

v0 = torch.tensor(
    V0
)


# ============================================================
# GENERACIÓN DEL DATASET
# ============================================================

print()

print(
    "============================================================"
)

print(
    "GENERACIÓN DATASET CNN v2.1"
)

print(
    "============================================================"
)


# ============================================================
# GENERAR ω
# ============================================================

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


# ============================================================
# VARIABLE NORMALIZADA z
# ============================================================

z = (
    omega_real
    -
    OMEGA_MIN
) / (
    OMEGA_MAX
    -
    OMEGA_MIN
)


# ============================================================
# RELACIÓN SINTÉTICA ω -> k
# ============================================================

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


# ============================================================
# m MEDIANTE
#
# omega² = k/m
#
# m = k / omega²
# ============================================================

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
# VENTANA TEMPORAL PARA LA LOSS FÍSICA
# ============================================================

N_PHYS = (
    round(
        PHYS_WINDOW_SECONDS
        /
        dt.item()
    )
    +
    1
)


t_phys = (
    t[
        :N_PHYS
    ]
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

print(
    "------------------------------------------------------------"
)

print(
    "MUESTREO"
)

print(
    "------------------------------------------------------------"
)

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
# GENERAR x(t)
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
# FFT
#
# IMPORTANTE:
#
# Se usa para construir la entrada CNN.
#
# ωFFT NO se usa para entrenar la física.
# ============================================================

with torch.no_grad():

    spectrum_complex = torch.fft.rfft(
        signals,
        dim=1
    )

    spectrum_magnitude = torch.abs(
        spectrum_complex
    )


# ============================================================
# SACAR DC
# ============================================================

spectrum_magnitude = (
    spectrum_magnitude[
        :,
        1:
    ]
)


omega_bins_no_dc = (
    omega_bins[
        1:
    ]
)


# ============================================================
# CALCULAR ωFFT
#
# SOLO PARA MÉTRICA.
#
# NO SE USA EN:
#
# - entrada explícita CNN
# - loss física
# ============================================================

with torch.no_grad():

    peak_index = torch.argmax(
        spectrum_magnitude,
        dim=1
    )

    omega_fft = (
        omega_bins_no_dc[
            peak_index
        ]
    )


# ============================================================
# RECORTAR ESPECTRO
#
# 0 < omega <= 120 rad/s
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


# ============================================================
# NORMALIZACIÓN DEL ESPECTRO
#
# Cada espectro se divide por su máximo.
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
# ============================================================

indices = torch.randperm(
    N_SYSTEMS
)


train_indices = (
    indices[
        :N_TRAIN
    ]
)


test_indices = (
    indices[
        N_TRAIN:
    ]
)


# ============================================================
# TRAIN
# ============================================================

spectrum_train = (
    spectrum_cnn[
        train_indices
    ]
)


signals_train = (
    signals[
        train_indices
    ]
)


signals_train_phys = (
    signals_train[
        :,
        :N_PHYS
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


signals_test = (
    signals[
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
# NORMALIZAR TARGET k
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


# ============================================================
# NORMALIZAR TARGET m
# ============================================================

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
# FORMATO CNN
#
# Conv1D espera:
#
# [batch, channels, length]
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

print(
    "============================================================"
)

print(
    "INFORMACIÓN DEL DATASET"
)

print(
    "============================================================"
)


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

print(
    f"Forma señal física train = "
    f"{tuple(signals_train_phys.shape)}"
)

print(
    f"Ventana física            = "
    f"{t_phys[-1].item():.6f} s "
    f"({N_PHYS} puntos)"
)


print()


print(
    f"Física = "
    f"{'Sí' if USE_PHYSICS else 'No'}"
)


# ============================================================
# MODELO
#
# MISMA CNN QUE v1
# ============================================================

torch.manual_seed(
    MODEL_SEED
)


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

print(
    "============================================================"
)


if USE_PHYSICS:

    print(
        "CNN v2.1 - CON FÍSICA SOBRE VENTANA x(t)"
    )

else:

    print(
        "CNN v2.1 - SIN FÍSICA"
    )


print(
    "============================================================"
)


print()


print(
    "Epoch        Loss total        "
    "Loss k        Loss m        Loss phys    lambda phys"
)


print(
    "--------------------------------------------------------------------"
)


for epoch in range(
    EPOCHS
):

    # ========================================================
    # TRAIN MODE
    # ========================================================

    model.train()


    optimizer.zero_grad()


    # ========================================================
    # CNN
    #
    # |X(f)| -> k_norm, m_norm
    # ========================================================

    k_pred_norm, m_pred_norm = model(
        spectrum_train
    )


    k_pred_norm = (
        k_pred_norm.squeeze(1)
    )


    m_pred_norm = (
        m_pred_norm.squeeze(1)
    )


    # ========================================================
    # DESNORMALIZAR k
    # ========================================================

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


    # ========================================================
    # DESNORMALIZAR m
    # ========================================================

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


    # ========================================================
    # LOSS SUPERVISADA k
    # ========================================================

    loss_k = torch.mean(
        (
            k_pred_norm
            -
            k_train_norm
        ) ** 2
    )


    # ========================================================
    # LOSS SUPERVISADA m
    # ========================================================

    loss_m = torch.mean(
        (
            m_pred_norm
            -
            m_train_norm
        ) ** 2
    )


    # ========================================================
    # FÍSICA
    #
    # omega = sqrt(k/m)
    # ========================================================

    omega_pred = torch.sqrt(
        k_pred
        /
        m_pred
    )


    # ========================================================
    # RECONSTRUCCIÓN DE LA SEÑAL
    #
    # Solución general:
    #
    # x(t) =
    #
    # x0 cos(omega t)
    #
    # +
    #
    # (v0/omega) sin(omega t)
    #
    #
    # En nuestro caso:
    #
    # x0 = 1
    # v0 = 0
    #
    # por lo que queda:
    #
    # x(t) = cos(omega t)
    # ========================================================

    omega_matrix = (
        omega_pred.unsqueeze(1)
    )


    t_phys_matrix = (
        t_phys.unsqueeze(0)
    )


    x_pred_phys = (
        X0
        *
        torch.cos(
            omega_matrix
            *
            t_phys_matrix
        )
        +
        (
            V0
            /
            omega_matrix
        )
        *
        torch.sin(
            omega_matrix
            *
            t_phys_matrix
        )
    )


    # ========================================================
    # LOSS FÍSICA CNN v2.1
    #
    # ACÁ ESTÁ EL CAMBIO IMPORTANTE
    #
    # Ya NO hacemos:
    #
    # omega_pred vs omega_fft
    #
    # Ahora hacemos:
    #
    # x_pred(t) vs x_real(t)
    # ========================================================

    loss_phys = torch.mean(
        (
            x_pred_phys
            -
            signals_train_phys
        ) ** 2
    )


    # ========================================================
    # CURRICULUM DE LA FÍSICA
    # ========================================================

    if epoch < PHYS_WARMUP_EPOCHS:

        lambda_phys = 0.0

    elif epoch < PHYS_RAMP_END_EPOCH:

        lambda_phys = (
            LAMBDA_PHYS_MAX
            *
            (
                epoch
                -
                PHYS_WARMUP_EPOCHS
            )
            /
            (
                PHYS_RAMP_END_EPOCH
                -
                PHYS_WARMUP_EPOCHS
            )
        )

    else:

        lambda_phys = LAMBDA_PHYS_MAX


    # ========================================================
    # LOSS TOTAL
    # ========================================================

    if USE_PHYSICS:

        loss = (
            loss_k
            +
            loss_m
            +
            lambda_phys
            *
            loss_phys
        )

    else:

        loss = (
            loss_k
            +
            loss_m
        )


    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    loss.backward()


    # ========================================================
    # OPTIMIZACIÓN
    # ========================================================

    optimizer.step()


    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    if epoch % 100 == 0:

        print(
            f"{epoch:5d}        "
            f"{loss.item():14.8f}        "
            f"{loss_k.item():10.8f}        "
            f"{loss_m.item():10.8f}        "
            f"{loss_phys.item():10.8f}        "
            f"{lambda_phys:10.6f}"
        )


# ============================================================
# TEST
# ============================================================

model.eval()


with torch.no_grad():

    # ========================================================
    # CNN
    # ========================================================

    k_test_pred_norm, m_test_pred_norm = model(
        spectrum_test
    )


    k_test_pred_norm = (
        k_test_pred_norm.squeeze(1)
    )


    m_test_pred_norm = (
        m_test_pred_norm.squeeze(1)
    )


    # ========================================================
    # DESNORMALIZAR k
    # ========================================================

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


    # ========================================================
    # DESNORMALIZAR m
    # ========================================================

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


    # ========================================================
    # ω PREDICHA
    # ========================================================

    omega_test_pred = torch.sqrt(
        k_test_pred
        /
        m_test_pred
    )


    # ========================================================
    # RECONSTRUCCIÓN x(t) TEST
    # ========================================================

    omega_test_matrix = (
        omega_test_pred.unsqueeze(1)
    )


    x_test_pred = (
        X0
        *
        torch.cos(
            omega_test_matrix
            *
            t.unsqueeze(0)
        )
        +
        (
            V0
            /
            omega_test_matrix
        )
        *
        torch.sin(
            omega_test_matrix
            *
            t.unsqueeze(0)
        )
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
# ERROR DE RECONSTRUCCIÓN DE SEÑAL
# ============================================================

signal_test_mse = torch.mean(
    (
        x_test_pred
        -
        signals_test
    ) ** 2
)


# ============================================================
# RESULTADOS TEST
# ============================================================

print()

print(
    "============================================================"
)

print(
    "MUESTRA DE RESULTADOS TEST"
)

print(
    "============================================================"
)


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

print(
    "============================================================"
)

print(
    "ERROR MEDIO TEST"
)

print(
    "============================================================"
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


print()


print(
    f"MSE reconstrucción x(t)     = "
    f"{signal_test_mse.item():.8f}"
)


# ============================================================
# GUARDAR MODELO
# ============================================================

checkpoint = {

    "model_state_dict":
        model.state_dict(),

    "architecture":
        "CNN_v2.1",

    "n_systems":
        N_SYSTEMS,

    "use_physics":
        USE_PHYSICS,

    "lambda_phys_max":
        LAMBDA_PHYS_MAX,

    "phys_window_seconds":
        PHYS_WINDOW_SECONDS,

    "n_phys":
        N_PHYS,

    "phys_warmup_epochs":
        PHYS_WARMUP_EPOCHS,

    "phys_ramp_end_epoch":
        PHYS_RAMP_END_EPOCH,

    "dataset_seed":
        DATASET_SEED,

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
        omega_fft_error.item(),

    "signal_test_mse":
        signal_test_mse.item()
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
