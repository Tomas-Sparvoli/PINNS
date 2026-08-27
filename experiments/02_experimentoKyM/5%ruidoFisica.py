import torch

from pinns.models.omega_network_identificoKyM import OmegaNetwork
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
# RANGOS FÍSICOS
# ============================================================

OMEGA_MIN = 5.0
OMEGA_MAX = 100.0

K_MIN = 25.0
K_MAX = 500.0

M_MIN = 0.05
M_MAX = 1.0


# ============================================================
# ENTRENAMIENTO
# ============================================================

EPOCHS = 2000

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


# ============================================================
# SEMILLAS
# ============================================================

DATASET_SEED = 42

MODEL_SEED = 1234

NOISE_SEED_5 = 5005

NOISE_SEED_15 = 5015


# ============================================================
# CONDICIONES INICIALES
# ============================================================

x0 = torch.tensor(1.0)

v0 = torch.tensor(0.0)


# ============================================================
# GENERAR DATASET LIMPIO
# ============================================================

torch.manual_seed(
    DATASET_SEED
)


print()
print("============================================================")
print("GENERACIÓN DEL DATASET")
print("============================================================")


# ------------------------------------------------------------
# ω ALEATORIA
#
# ω ∈ [5,100]
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
# CALCULAR m MEDIANTE LA FÍSICA
#
# ω² = k/m
#
# m = k/ω²
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
    f"dt                   = "
    f"{dt.item():.6f} s"
)

print(
    f"Frecuencia muestreo  = "
    f"{sampling_frequency.item():.4f} Hz"
)

print(
    f"Frecuencia Nyquist   = "
    f"{nyquist_frequency.item():.4f} Hz"
)

print(
    f"Resolución FFT Δω    = "
    f"{delta_omega:.4f} rad/s"
)


# ============================================================
# BINS FFT
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
# GENERAR SEÑALES LIMPIAS
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
# SPLIT TRAIN / TEST
#
# Se genera UNA sola vez.
# Las cuatro pruebas usan exactamente los mismos sistemas.
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
# TARGETS TRAIN
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


# ============================================================
# TARGETS TEST
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
# INFORMACIÓN DATASET
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
# FUNCIÓN PARA AGREGAR RUIDO
#
# noise_level:
#
# 0.05 -> 5 %
# 0.15 -> 15 %
# ============================================================

def add_noise(
    signals,
    noise_level,
    seed
):

    # --------------------------------------------------------
    # Generador independiente para poder reutilizar
    # exactamente el mismo ruido en con/sin física.
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    # --------------------------------------------------------
    # Desviación estándar de cada señal
    # --------------------------------------------------------

    signal_std = signals.std(
        dim=1,
        keepdim=True
    )


    # --------------------------------------------------------
    # Ruido gaussiano N(0,1)
    # --------------------------------------------------------

    noise = torch.randn(
        signals.shape,
        generator=generator,
        dtype=signals.dtype
    )


    # --------------------------------------------------------
    # Señal observada
    #
    # x_noisy =
    # x +
    # noise_level * std(x) * ruido
    # --------------------------------------------------------

    signals_noisy = (
        signals
        +
        noise_level
        *
        signal_std
        *
        noise
    )


    return signals_noisy


# ============================================================
# FUNCIÓN DE ENTRENAMIENTO
# ============================================================

def run_experiment(
    noise_level,
    noise_seed,
    use_physics,
    model_path
):

    print()
    print()
    print("============================================================")

    if use_physics:

        print(
            f"RUIDO {noise_level * 100:.0f}% "
            f"- CON FÍSICA"
        )

    else:

        print(
            f"RUIDO {noise_level * 100:.0f}% "
            f"- SIN FÍSICA"
        )

    print("============================================================")


    # ========================================================
    # AGREGAR RUIDO
    # ========================================================

    signals_noisy = add_noise(
        signals_clean,
        noise_level,
        noise_seed
    )


    # ========================================================
    # FFT SOBRE LAS SEÑALES RUIDOSAS
    # ========================================================

    with torch.no_grad():

        omega_fft = fft_peak_omega(
            signals_noisy
        )


    # ========================================================
    # TRAIN / TEST
    # ========================================================

    omega_fft_train = omega_fft[
        train_indices
    ]

    omega_fft_test = omega_fft[
        test_indices
    ]


    # ========================================================
    # NORMALIZACIÓN DE ωFFT
    #
    # La media y std se calculan SOLO sobre train.
    # ========================================================

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


    # ========================================================
    # MISMA INICIALIZACIÓN PARA CON/SIN FÍSICA
    # ========================================================

    torch.manual_seed(
        MODEL_SEED
    )


    model = OmegaNetwork()


    # ========================================================
    # OPTIMIZADOR
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

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


        # ----------------------------------------------------
        # PREDICCIÓN NORMALIZADA
        # ----------------------------------------------------

        k_pred_norm, m_pred_norm = model(
            omega_fft_train_norm.unsqueeze(1)
        )


        k_pred_norm = (
            k_pred_norm.squeeze(1)
        )

        m_pred_norm = (
            m_pred_norm.squeeze(1)
        )


        # ----------------------------------------------------
        # DESNORMALIZAR k
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DESNORMALIZAR m
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # LOSS k
        # ----------------------------------------------------

        loss_k = torch.mean(
            (
                k_pred_norm
                -
                k_train_norm
            ) ** 2
        )


        # ----------------------------------------------------
        # LOSS m
        # ----------------------------------------------------

        loss_m = torch.mean(
            (
                m_pred_norm
                -
                m_train_norm
            ) ** 2
        )


        # ----------------------------------------------------
        # DIFFERENTIABLE PHYSICS
        #
        # ω = sqrt(k/m)
        # ----------------------------------------------------

        omega_pred = torch.sqrt(
            k_pred
            /
            m_pred
        )


        # ----------------------------------------------------
        # LOSS FÍSICA
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # LOSS TOTAL
        # ----------------------------------------------------

        if use_physics:

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


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # ACTUALIZACIÓN
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # MOSTRAR
        # ----------------------------------------------------

        if epoch % 100 == 0:

            print(
                f"{epoch:5d}        "
                f"{loss.item():14.8f}        "
                f"{loss_k.item():10.8f}        "
                f"{loss_m.item():10.8f}        "
                f"{loss_phys.item():10.8f}"
            )


    # ========================================================
    # TEST
    # ========================================================

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


        # ----------------------------------------------------
        # DESNORMALIZAR k
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DESNORMALIZAR m
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ω RECONSTRUIDA
        # ----------------------------------------------------

        omega_test_pred = torch.sqrt(
            k_test_pred
            /
            m_test_pred
        )


    # ========================================================
    # ERRORES
    # ========================================================

    error_k = torch.mean(
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


    error_m = torch.mean(
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


    error_omega = torch.mean(
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


    error_fft = torch.mean(
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


    # ========================================================
    # MOSTRAR 10 EJEMPLOS
    # ========================================================

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
        10,
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


    # ========================================================
    # RESULTADO GLOBAL
    # ========================================================

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
        f"Ruido                     = "
        f"{noise_level * 100:.0f}%"
    )

    print(
        f"Física                    = "
        f"{'Sí' if use_physics else 'No'}"
    )

    print()

    print(
        f"Error medio relativo k     = "
        f"{error_k.item():.4f}%"
    )

    print(
        f"Error medio relativo m     = "
        f"{error_m.item():.4f}%"
    )

    print(
        f"Error medio relativo ω pred = "
        f"{error_omega.item():.4f}%"
    )

    print(
        f"Error medio propio FFT      = "
        f"{error_fft.item():.4f}%"
    )


    # ========================================================
    # GUARDAR MODELO
    # ========================================================

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "noise_level":
            noise_level,

        "use_physics":
            use_physics,

        "omega_mean":
            omega_mean,

        "omega_std":
            omega_std,

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

        "error_k":
            error_k.item(),

        "error_m":
            error_m.item(),

        "error_omega":
            error_omega.item(),

        "error_fft":
            error_fft.item(),
    }


    torch.save(
        checkpoint,
        model_path
    )


    print()

    print(
        "Modelo guardado en: "
        f"{model_path}"
    )


    # ========================================================
    # DEVOLVER RESULTADOS
    # ========================================================

    return {

        "noise":
            noise_level * 100.0,

        "physics":
            use_physics,

        "error_k":
            error_k.item(),

        "error_m":
            error_m.item(),

        "error_omega":
            error_omega.item(),

        "error_fft":
            error_fft.item(),
    }


# ============================================================
# EJECUTAR LAS CUATRO PRUEBAS
# ============================================================

results = []


# ============================================================
# 5 % CON FÍSICA
# ============================================================

results.append(

    run_experiment(

        noise_level=0.05,

        noise_seed=NOISE_SEED_5,

        use_physics=True,

        model_path=(
            "omega_network_ruido5_con_fisica.pth"
        )
    )
)


# ============================================================
# 5 % SIN FÍSICA
# ============================================================

results.append(

    run_experiment(

        noise_level=0.05,

        noise_seed=NOISE_SEED_5,

        use_physics=False,

        model_path=(
            "omega_network_ruido5_sin_fisica.pth"
        )
    )
)


# ============================================================
# 15 % CON FÍSICA
# ============================================================

results.append(

    run_experiment(

        noise_level=0.50,

        noise_seed=NOISE_SEED_15,

        use_physics=True,

        model_path=(
            "omega_network_ruido50_con_fisica.pth"
        )
    )
)


# ============================================================
# 15 % SIN FÍSICA
# ============================================================

results.append(

    run_experiment(

        noise_level=0.50,

        noise_seed=NOISE_SEED_15,

        use_physics=False,

        model_path=(
            "omega_network_ruido50_sin_fisica.pth"
        )
    )
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print()
print("============================================================")
print("RESUMEN FINAL - ROBUSTEZ AL RUIDO")
print("============================================================")

print()

print(
    "Ruido      Física      "
    "Error k      Error m      Error ω      Error FFT"
)

print(
    "--------------------------------------------------------------------------"
)


for result in results:

    physics_text = (
        "Sí"
        if result["physics"]
        else "No"
    )

    print(
        f"{result['noise']:5.0f}%      "
        f"{physics_text:>3}        "
        f"{result['error_k']:8.4f}%    "
        f"{result['error_m']:8.4f}%    "
        f"{result['error_omega']:8.4f}%    "
        f"{result['error_fft']:8.4f}%"
    )