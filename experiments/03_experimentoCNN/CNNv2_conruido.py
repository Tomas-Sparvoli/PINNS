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

USE_PHYSICS = True


# ============================================================
# NIVEL DE RUIDO
# ============================================================
#
# 0.05 = ruido equivalente al 5 % del desvío estándar
# 0.15 = ruido equivalente al 15 %
#
# Para comparar experimentos:
#   0.00 -> sin ruido
#   0.05 -> 5 %
#   0.15 -> 15 %
# ============================================================

NOISE_LEVEL = 0.15


# ============================================================
# RANGOS DE PARÁMETROS
# ============================================================

OMEGA_MIN = 5.0
OMEGA_MAX = 100.0

K_MIN = 25.0
K_MAX = 500.0

M_MIN = 0.05
M_MAX = 1.0


# ============================================================
# CONFIGURACIÓN DE LA CNN
# ============================================================

CNN_OMEGA_MAX = 120.0


# ============================================================
# ENTRENAMIENTO
# ============================================================

EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# ============================================================
# CURRÍCULUM PARA LA PÉRDIDA FÍSICA
# ============================================================
#
# La pérdida física:
#
#   0 - 499       -> lambda = 0
#   500 - 999     -> lambda aumenta progresivamente
#   >= 1000       -> lambda = 0.01
#
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

torch.manual_seed(DATASET_SEED)


# ============================================================
# CONDICIONES INICIALES
# ============================================================

X0 = 1.0
V0 = 0.0


# ============================================================
# GENERACIÓN DE LOS PARÁMETROS SINTÉTICOS
# ============================================================
#
# Primero generamos omega.
#
# Después construimos k mediante una relación determinística.
#
# Finalmente:
#
#       omega² = k/m
#
# por lo tanto:
#
#       m = k / omega²
#
# ============================================================

omega_real = (
    OMEGA_MIN
    + (OMEGA_MAX - OMEGA_MIN)
    * torch.rand(N_SYSTEMS)
)

z = (
    (omega_real - OMEGA_MIN)
    / (OMEGA_MAX - OMEGA_MIN)
)

k_real = (
    K_MIN
    + (K_MAX - K_MIN)
    * (
        0.3 * z
        + 0.7 * z**2
    )
)

m_real = (
    k_real
    / omega_real**2
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
    t[1] - t[0]
).item()


# ============================================================
# VENTANA TEMPORAL PARA LA PÉRDIDA FÍSICA
# ============================================================

N_PHYS = int(
    PHYS_WINDOW_SECONDS / dt
) + 1

t_phys = t[
    :N_PHYS
]


# ============================================================
# INFORMACIÓN DE MUESTREO
# ============================================================

sampling_frequency = (
    1.0 / dt
)

nyquist_frequency = (
    sampling_frequency / 2.0
)

delta_f = (
    1.0 / T_TOTAL
)

delta_omega = (
    2.0
    * torch.pi
    * delta_f
)

print()
print("====================================================")
print("CONFIGURACIÓN TEMPORAL")
print("====================================================")

print(
    f"dt = {dt:.8f} s"
)

print(
    f"Frecuencia de muestreo = "
    f"{sampling_frequency:.4f} Hz"
)

print(
    f"Nyquist = "
    f"{nyquist_frequency:.4f} Hz"
)

print(
    f"Resolución espectral Δω = "
    f"{delta_omega:.6f} rad/s"
)

print(
    f"Ventana física = "
    f"{t_phys[-1].item():.6f} s"
)

print(
    f"Puntos usados en pérdida física = "
    f"{N_PHYS}"
)

print(
    f"Nivel de ruido = "
    f"{NOISE_LEVEL * 100:.1f} %"
)


# ============================================================
# GENERACIÓN DE LAS SEÑALES LIMPIAS
# ============================================================
#
# signals_clean representa la señal ideal del oscilador.
#
# Estas señales NO contienen ruido.
#
# ============================================================

signals_clean = simulate_oscillator(
    t=t.unsqueeze(0),              # [4000] -> [1, 4000]
    omega=omega_real.unsqueeze(1), # [100]  -> [100, 1]
    x0=X0,
    v0=V0
)


# ============================================================
# AGREGADO DE RUIDO GAUSSIANO
# ============================================================
#
# Para cada señal calculamos su desvío estándar.
#
# sigma_ruido =
#
#       NOISE_LEVEL * std(señal)
#
# Luego generamos ruido gaussiano:
#
#       epsilon ~ N(0, sigma_ruido²)
#
# Y construimos:
#
#       x_observada = x_limpia + epsilon
#
# La CNN y la pérdida física trabajarán con esta señal
# observada ruidosa.
#
# ============================================================

signal_std = signals_clean.std(
    dim=1,
    keepdim=True
)

noise = (
    NOISE_LEVEL
    * signal_std
    * torch.randn_like(
        signals_clean
    )
)

signals_noisy = (
    signals_clean
    + noise
)


# ============================================================
# FFT DE LAS SEÑALES RUIDOSAS
# ============================================================
#
# IMPORTANTE:
#
# La CNN NO recibe la señal temporal directamente.
#
# Recibe la magnitud del espectro obtenido a partir
# de la señal ruidosa.
#
# ============================================================

with torch.no_grad():

    spectrum = torch.fft.rfft(
        signals_noisy,
        dim=1
    )

    spectrum_magnitude = torch.abs(
        spectrum
    )


# ============================================================
# BINS DE FRECUENCIA
# ============================================================

frequency_bins = torch.fft.rfftfreq(
    N_POINTS,
    d=dt
)

omega_bins = (
    2.0
    * torch.pi
    * frequency_bins
)


# ============================================================
# ELIMINACIÓN DE LA COMPONENTE DC
# ============================================================
#
# El primer bin corresponde a:
#
#       omega = 0
#
# Se elimina tanto de la magnitud como del vector
# de frecuencias para mantener la correspondencia.
#
# ============================================================

spectrum_magnitude = (
    spectrum_magnitude[:, 1:]
)

omega_bins_no_dc = (
    omega_bins[1:]
)


# ============================================================
# ESTIMACIÓN FFT DE OMEGA
# ============================================================
#
# Esta omega_fft NO se usa como entrada directa de la CNN
# ni como target de la pérdida física.
#
# Se conserva solamente como métrica de referencia.
#
# ============================================================

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
# RECORTE DEL ESPECTRO PARA LA CNN
# ============================================================

cnn_mask = (
    omega_bins_no_dc
    <= CNN_OMEGA_MAX
)

spectrum_cnn = (
    spectrum_magnitude[
        :,
        cnn_mask
    ]
)


# ============================================================
# NORMALIZACIÓN INDIVIDUAL DE CADA ESPECTRO
# ============================================================
#
# Cada espectro se divide por su propio máximo.
#
# Así la CNN trabaja principalmente con la forma
# espectral y no con la escala absoluta.
#
# ============================================================

spectrum_max = spectrum_cnn.max(
    dim=1,
    keepdim=True
).values

spectrum_cnn = (
    spectrum_cnn
    / (
        spectrum_max
        + 1e-8
    )
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

N_TRAIN = int(
    N_SYSTEMS
    * TRAIN_RATIO
)

indices = torch.randperm(
    N_SYSTEMS
)

train_indices = indices[
    :N_TRAIN
]

test_indices = indices[
    N_TRAIN:
]


spectrum_train = spectrum_cnn[
    train_indices
]

spectrum_test = spectrum_cnn[
    test_indices
]


# Señales ruidosas:
#
# Son las señales que simulan lo que observaríamos
# experimentalmente.

signals_train_noisy = signals_noisy[
    train_indices
]

signals_test_noisy = signals_noisy[
    test_indices
]


# Señales limpias:
#
# Las conservamos solamente para poder evaluar después
# qué tan cerca está la reconstrucción de la dinámica ideal.

signals_train_clean = signals_clean[
    train_indices
]

signals_test_clean = signals_clean[
    test_indices
]


k_train = k_real[
    train_indices
]

k_test = k_real[
    test_indices
]

m_train = m_real[
    train_indices
]

m_test = m_real[
    test_indices
]

omega_train = omega_real[
    train_indices
]

omega_test = omega_real[
    test_indices
]

omega_fft_train = omega_fft[
    train_indices
]

omega_fft_test = omega_fft[
    test_indices
]


# ============================================================
# TARGET TEMPORAL PARA LA PÉRDIDA FÍSICA
# ============================================================
#
# La pérdida física compara la reconstrucción del modelo
# contra la señal OBSERVADA, que en este experimento
# contiene ruido.
#
# Solo utilizamos el primer segundo.
#
# ============================================================

signals_train_phys = (
    signals_train_noisy[
        :,
        :N_PHYS
    ]
)


# ============================================================
# NORMALIZACIÓN DE k Y m
# ============================================================

k_train_norm = (
    (k_train - K_MIN)
    / (K_MAX - K_MIN)
)

m_train_norm = (
    (m_train - M_MIN)
    / (M_MAX - M_MIN)
)


# ============================================================
# FORMATO PARA CONV1D
# ============================================================
#
# Antes:
#
#       [batch, bins]
#
# Después:
#
#       [batch, channels, bins]
#
# En nuestro caso:
#
#       [80, 1, 382]
#
# ============================================================

spectrum_train = (
    spectrum_train.unsqueeze(1)
)

spectrum_test = (
    spectrum_test.unsqueeze(1)
)


print()
print("====================================================")
print("DATASET")
print("====================================================")

print(
    f"Sistemas totales = {N_SYSTEMS}"
)

print(
    f"Train = {len(train_indices)}"
)

print(
    f"Test = {len(test_indices)}"
)

print(
    f"Bins usados por CNN = "
    f"{spectrum_train.shape[-1]}"
)

print(
    f"Forma input train = "
    f"{tuple(spectrum_train.shape)}"
)

print(
    f"Forma señal física train = "
    f"{tuple(signals_train_phys.shape)}"
)


# ============================================================
# MODELO
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

model.train()

for epoch in range(EPOCHS):

    optimizer.zero_grad()


    # ========================================================
    # FORWARD CNN
    # ========================================================

    (
        k_pred_norm,
        m_pred_norm
    ) = model(
        spectrum_train
    )

    k_pred_norm = (
        k_pred_norm.squeeze(1)
    )

    m_pred_norm = (
        m_pred_norm.squeeze(1)
    )


    # ========================================================
    # DESNORMALIZACIÓN
    # ========================================================

    k_pred = (
        K_MIN
        + k_pred_norm
        * (K_MAX - K_MIN)
    )

    m_pred = (
        M_MIN
        + m_pred_norm
        * (M_MAX - M_MIN)
    )


    # ========================================================
    # PÉRDIDAS SUPERVISADAS
    # ========================================================

    loss_k = torch.mean(
        (
            k_pred_norm
            - k_train_norm
        ) ** 2
    )

    loss_m = torch.mean(
        (
            m_pred_norm
            - m_train_norm
        ) ** 2
    )


    # ========================================================
    # OMEGA PREDICHA A PARTIR DE k Y m
    # ========================================================

    omega_pred = torch.sqrt(
        k_pred
        / m_pred
    )


    # ========================================================
    # RECONSTRUCCIÓN DE LA SEÑAL x(t)
    # ========================================================
    #
    # Ecuación:
    #
    #       x¨ + omega² x = 0
    #
    # Solución:
    #
    #       x(t)
    #       =
    #       x0*cos(omega*t)
    #       +
    #       (v0/omega)*sin(omega*t)
    #
    # omega_matrix:
    #
    #       [batch]
    #           ->
    #       [batch, 1]
    #
    # t_phys_matrix:
    #
    #       [N_PHYS]
    #           ->
    #       [1, N_PHYS]
    #
    # Broadcasting:
    #
    #       [batch,1] * [1,N_PHYS]
    #
    #           ->
    #
    #       [batch,N_PHYS]
    #
    # ========================================================

    omega_matrix = (
        omega_pred.unsqueeze(1)
    )

    t_phys_matrix = (
        t_phys.unsqueeze(0)
    )

    x_pred_phys = (
        X0
        * torch.cos(
            omega_matrix
            * t_phys_matrix
        )
        +
        (
            V0
            / omega_matrix
        )
        * torch.sin(
            omega_matrix
            * t_phys_matrix
        )
    )


    # ========================================================
    # PÉRDIDA FÍSICA
    # ========================================================
    #
    # Comparamos:
    #
    #       trayectoria reconstruida por k_pred,m_pred
    #
    # contra:
    #
    #       señal temporal observada RUIDOSA
    #
    # ========================================================

    loss_phys = torch.mean(
        (
            x_pred_phys
            - signals_train_phys
        ) ** 2
    )   


    # ========================================================
    # CURRÍCULUM DE LA PÉRDIDA FÍSICA
    # ========================================================

    if epoch < PHYS_WARMUP_EPOCHS:

        lambda_phys = 0.0

    elif epoch < PHYS_RAMP_END_EPOCH:

        lambda_phys = (
            LAMBDA_PHYS_MAX
            * (
                epoch
                - PHYS_WARMUP_EPOCHS
            )
            / (
                PHYS_RAMP_END_EPOCH
                - PHYS_WARMUP_EPOCHS
            )
        )

    else:

        lambda_phys = (
            LAMBDA_PHYS_MAX
        )


    # ========================================================
    # LOSS TOTAL
    # ========================================================

    if USE_PHYSICS:

        loss = (
            loss_k
            + loss_m
            + lambda_phys
            * loss_phys
        )

    else:

        loss = (
            loss_k
            + loss_m
        )


    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    loss.backward()

    optimizer.step()


    # ========================================================
    # PRINT
    # ========================================================

    if (
        epoch % 100 == 0
        or epoch == EPOCHS - 1
    ):

        print(
            f"Epoch {epoch:4d} | "
            f"Loss = {loss.item():.8f} | "
            f"Lk = {loss_k.item():.8f} | "
            f"Lm = {loss_m.item():.8f} | "
            f"Lphys = {loss_phys.item():.8f} | "
            f"lambda = {lambda_phys:.5f}"
        )


# ============================================================
# TEST
# ============================================================

model.eval()

with torch.no_grad():

    (
        k_test_pred_norm,
        m_test_pred_norm
    ) = model(
        spectrum_test
    )

    k_test_pred_norm = (
        k_test_pred_norm.squeeze(1)
    )

    m_test_pred_norm = (
        m_test_pred_norm.squeeze(1)
    )


    # ========================================================
    # DESNORMALIZACIÓN
    # ========================================================

    k_test_pred = (
        K_MIN
        + k_test_pred_norm
        * (K_MAX - K_MIN)
    )

    m_test_pred = (
        M_MIN
        + m_test_pred_norm
        * (M_MAX - M_MIN)
    )


    # ========================================================
    # OMEGA PREDICHA
    # ========================================================

    omega_test_pred = torch.sqrt(
        k_test_pred
        / m_test_pred
    )


    # ========================================================
    # RECONSTRUCCIÓN COMPLETA DE 20 SEGUNDOS
    # ========================================================

    omega_test_matrix = (
        omega_test_pred.unsqueeze(1)
    )

    t_matrix = (
        t.unsqueeze(0)
    )

    x_test_pred = (
        X0
        * torch.cos(
            omega_test_matrix
            * t_matrix
        )
        +
        (
            V0
            / omega_test_matrix
        )
        * torch.sin(
            omega_test_matrix
            * t_matrix
        )
    )


    # ========================================================
    # ERRORES RELATIVOS
    # ========================================================

    error_k = torch.mean(
        torch.abs(
            (
                k_test_pred
                - k_test
            )
            / k_test
        )
    ) * 100.0


    error_m = torch.mean(
        torch.abs(
            (
                m_test_pred
                - m_test
            )
            / m_test
        )
    ) * 100.0


    # ========================================================
    # ERROR TEMPORAL VS SEÑAL RUIDOSA
    # ========================================================
    #
    # Indica qué tan bien reproduce la observación
    # que realmente recibió el modelo.
    #
    # ========================================================

    signal_mse_noisy = torch.mean(
        (
            x_test_pred
            - signals_test_noisy
        ) ** 2
    )


    # ========================================================
    # ERROR TEMPORAL VS SEÑAL LIMPIA
    # ========================================================
    #
    # Esta métrica es especialmente interesante.
    #
    # Aunque la CNN recibe una observación ruidosa,
    # podemos comprobar si los parámetros estimados
    # reconstruyen correctamente la dinámica ideal.
    #
    # ========================================================

    signal_mse_clean = torch.mean(
        (
            x_test_pred
            - signals_test_clean
        ) ** 2
    )


    # ========================================================
    # ERROR EN EL PRIMER SEGUNDO
    # ========================================================

    signal_mse_clean_1s = torch.mean(
        (
            x_test_pred[:, :N_PHYS]
            - signals_test_clean[:, :N_PHYS]
        ) ** 2
    )

    signal_mse_noisy_1s = torch.mean(
        (
            x_test_pred[:, :N_PHYS]
            - signals_test_noisy[:, :N_PHYS]
        ) ** 2
    )


# ============================================================
# RESULTADOS
# ============================================================

print()
print("====================================================")
print("RESULTADOS CNN v2.1 CON RUIDO")
print("====================================================")

print(
    f"Nivel de ruido: "
    f"{NOISE_LEVEL * 100:.1f} %"
)

print(
    f"Error medio k: "
    f"{error_k.item():.4f} %"
)

print(
    f"Error medio m: "
    f"{error_m.item():.4f} %"
)


# ============================================================
# GUARDADO DEL MODELO
# ============================================================

noise_percent = int(
    NOISE_LEVEL
    * 100
)

physics_name = (
    "con_fisica"
    if USE_PHYSICS
    else "sin_fisica"
)

model_filename = (
    f"cnn_v2_1_dataset{N_SYSTEMS}"
    f"_ruido{noise_percent}"
    f"_{physics_name}.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "N_SYSTEMS":
            N_SYSTEMS,

        "NOISE_LEVEL":
            NOISE_LEVEL,

        "USE_PHYSICS":
            USE_PHYSICS,

        "PHYS_WINDOW_SECONDS":
            PHYS_WINDOW_SECONDS,

        "LAMBDA_PHYS_MAX":
            LAMBDA_PHYS_MAX,

        "PHYS_WARMUP_EPOCHS":
            PHYS_WARMUP_EPOCHS,

        "PHYS_RAMP_END_EPOCH":
            PHYS_RAMP_END_EPOCH,

        "error_k_percent":
            error_k.item(),

        "error_m_percent":
            error_m.item(),
    },
    model_filename
)

print()
print(
    f"Modelo guardado como: "
    f"{model_filename}"
)