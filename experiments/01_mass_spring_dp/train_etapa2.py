import torch

from pinns.models.omega_network_etapa2 import OmegaNetwork
from pinns.physics.mass_spring import simulate_oscillator


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_POINTS = 1000
T_TOTAL = 10.0

MASS = 1.0

OMEGAS_REAL = [
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0
]

OMEGA_MIN = 5.0
OMEGA_MAX = 50.0

EPOCHS = 5000
LEARNING_RATE = 1e-3


# ============================================================
# SEMILLA
# ============================================================

torch.manual_seed(42)


# ============================================================
# TIEMPO
# ============================================================

t_base = torch.linspace(
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
# GENERAR SISTEMAS
# ============================================================

systems = []


for omega_real in OMEGAS_REAL:

    omega_real_tensor = torch.tensor(
        omega_real,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # PARÁMETRO FÍSICO k
    # --------------------------------------------------------

    k_real = MASS * omega_real ** 2


    # --------------------------------------------------------
    # OBSERVACIÓN
    # --------------------------------------------------------

    with torch.no_grad():

        x_obs = simulate_oscillator(
            t_base,
            omega_real_tensor,
            x0,
            v0
        )


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    dt = float(
        t_base[1] - t_base[0]
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
        torch.fft.rfft(x_obs)
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
    # GUARDAR SISTEMA
    # --------------------------------------------------------

    systems.append(
        {
            "omega_real": omega_real,
            "k_real": k_real,
            "omega_fft": omega_fft.detach()
        }
    )


# ============================================================
# MOSTRAR SISTEMAS
# ============================================================

print()
print("============================================================")
print("SISTEMAS DE ENTRENAMIENTO")
print("============================================================")

print()
print(
    "ω real        k real        ω FFT"
)

print(
    "--------------------------------------------"
)

for system in systems:

    print(
        f"{system['omega_real']:8.2f}        "
        f"{system['k_real']:8.2f}        "
        f"{system['omega_fft'].item():8.4f}"
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
# PREDICCIONES INICIALES
# ============================================================

print()
print("============================================================")
print("PREDICCIONES INICIALES")
print("============================================================")

print()
print(
    "ω real        ω FFT        ω pred"
)

print(
    "--------------------------------------------"
)

with torch.no_grad():

    for system in systems:

        omega_pred = model(
            system["omega_fft"].reshape(1, 1)
        ).squeeze()

        print(
            f"{system['omega_real']:8.2f}        "
            f"{system['omega_fft'].item():8.4f}        "
            f"{omega_pred.item():8.4f}"
        )


# ============================================================
# ENTRENAMIENTO
# ============================================================

print()
print("============================================================")
print("ENTRENAMIENTO MULTI-SISTEMA")
print("============================================================")

print()
print(
    "Epoch        Loss total"
)

print(
    "--------------------------------------------"
)


for epoch in range(EPOCHS):

    optimizer.zero_grad()

    total_loss = 0.0


    # ========================================================
    # RECORRER TODOS LOS SISTEMAS
    # ========================================================

    for system in systems:

        # ----------------------------------------------------
        # TIEMPO
        # ----------------------------------------------------

        t = (
            t_base
            .clone()
            .detach()
            .requires_grad_(True)
        )


        # ----------------------------------------------------
        # PARÁMETROS
        # ----------------------------------------------------

        omega_fft = system["omega_fft"]

        k_real = system["k_real"]


        # ----------------------------------------------------
        # OMEGANETWORK
        # ----------------------------------------------------

        omega_pred = model(
            omega_fft.reshape(1, 1)
        ).squeeze()


        # ----------------------------------------------------
        # MODELO FÍSICO
        # ----------------------------------------------------

        x_pred = simulate_oscillator(
            t,
            omega_pred,
            x0,
            v0
        )


        # ----------------------------------------------------
        # PRIMERA DERIVADA
        # ----------------------------------------------------

        dx_dt = torch.autograd.grad(
            x_pred,
            t,
            grad_outputs=torch.ones_like(x_pred),
            create_graph=True
        )[0]


        # ----------------------------------------------------
        # SEGUNDA DERIVADA
        # ----------------------------------------------------

        d2x_dt2 = torch.autograd.grad(
            dx_dt,
            t,
            grad_outputs=torch.ones_like(dx_dt),
            create_graph=True
        )[0]


        # ----------------------------------------------------
        # RESIDUAL FÍSICO
        # ----------------------------------------------------

        residual = (
            MASS * d2x_dt2
            +
            k_real * x_pred
        )


        # ----------------------------------------------------
        # LOSS FÍSICA
        # ----------------------------------------------------

        loss_system = torch.mean(
            residual ** 2
        ) / k_real


        # ----------------------------------------------------
        # ACUMULAR
        # ----------------------------------------------------

        total_loss = (
            total_loss
            +
            loss_system
        )


    # ========================================================
    # PROMEDIO ENTRE SISTEMAS
    # ========================================================

    total_loss = (
        total_loss
        /
        len(systems)
    )


    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    total_loss.backward()

    optimizer.step()


    # ========================================================
    # RESULTADOS
    # ========================================================

    if epoch % 100 == 0:

        print(
            f"{epoch:5d}        "
            f"{total_loss.item():16.8f}"
        )


# ============================================================
# RESULTADOS FINALES
# ============================================================

print()
print("============================================================")
print("RESULTADOS FINALES")
print("============================================================")

print()
print(
    "ω real        ω FFT        ω pred        error"
)

print(
    "------------------------------------------------------------"
)

with torch.no_grad():

    for system in systems:

        omega_real = system["omega_real"]

        omega_fft = system["omega_fft"]

        omega_pred = model(
            omega_fft.reshape(1, 1)
        ).squeeze().item()

        error = (
            omega_pred
            -
            omega_real
        )

        print(
            f"{omega_real:8.2f}        "
            f"{omega_fft.item():8.4f}        "
            f"{omega_pred:8.4f}        "
            f"{error:+10.4f}"
        )

    # ============================================================
    # GUARDAR MODELO
    # ============================================================

    MODEL_PATH = "omega_network_dp_etapa2.pth"

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print(
        "Modelo guardado en: "
        f"{MODEL_PATH}"
    )