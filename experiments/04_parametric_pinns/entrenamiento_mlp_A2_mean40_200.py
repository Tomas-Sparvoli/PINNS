# ============================================================
# MODELO HIBRIDO FISICAMENTE INFORMADO
#
# RESTRICCION ANALITICA:
#
# A2 < 0:
#     cierre / sobreaduccion
#     NO se penaliza
#
# A2 >= 0:
#     se compara contra Ag_mean40 real
#
#
# ECUACION:
#
# A2 =
# 0.25 * L0^2 *
# [1 + 0.6*CT - 0.2*TA - 0.2*LCA] *
# [1 - 2*LCA]
#
# Para hombre:
#
# L0 = 1.6 cm
#
# 0.25 * L0^2 = 0.64 cm^2
#
#
# LOSS FISICA:
#
# L_A2_mean40 =
#
# mean[
#     I(A2 >= 0)
#     *
#     (A2 - Ag_mean40_real)^2
# ]
#
# NO SE UTILIZA SURROGATE.
# ============================================================


import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. CONFIGURACION
# ============================================================

SEED = 42


CSV_PATH = Path(
    r"C:\Users\sparv\OneDrive\Escritorio\PhonationModel"
) / "dataset_features_200_final.csv"


AG40_PATH = Path(
    r"C:\Users\sparv\OneDrive\Escritorio\PhonationModel"
) / "dataset_sintetico_200_agmean40.csv"


BASE_DIR = Path(
    __file__
).resolve().parent


BASELINE_DIR = (
    BASE_DIR
    / "resultados_mlp_200_v2"
)


SPLIT_PATH = (
    BASELINE_DIR
    / "split_indices.npz"
)


OUTPUT_DIR = (
    BASE_DIR
    / "resultados_mlp_A2_mean40_200"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. HIPERPARAMETROS
# ============================================================

BATCH_SIZE = 16

LEARNING_RATE = 1e-3

EPOCHS = 2000

PATIENCE = 150

MIN_DELTA = 1e-6


# ============================================================
# 3. PARAMETROS FISICOS
# ============================================================

# Longitud vocal de referencia masculina
# 16 mm = 1.6 cm

L0_CM = 1.6


A2_FACTOR = (
    0.25
    * L0_CM ** 2
)


LAMBDA_A2_MEAN40 = 1.0


# ============================================================
# 4. PESOS LOSS SUPERVISADA
# ============================================================

W_LCA = 1.5

W_TA = 1.5

W_CT = 1.0


# ============================================================
# 5. REPRODUCIBILIDAD
# ============================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 6. DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()

print("=" * 90)

print(
    "MODELO FISICAMENTE INFORMADO "
    "- A2 vs Ag_mean40"
)

print("=" * 90)


print(
    f"Dispositivo: {device}"
)

print(
    f"Seed: {SEED}"
)


print()

print(
    f"L0 = "
    f"{L0_CM:.3f} cm"
)


print(
    f"0.25 * L0^2 = "
    f"{A2_FACTOR:.6f} cm^2"
)


print()

print(
    "Hipotesis:"
)

print(
    "  A2 < 0 -> cierre/sobreaduccion "
    "(sin penalizacion)"
)

print(
    "  A2 >= 0 -> comparar contra Ag_mean40"
)


print()

print(
    f"Lambda A2-mean40 = "
    f"{LAMBDA_A2_MEAN40}"
)


print()

print(
    "Pesos loss supervisada:"
)

print(
    f"  LCA = {W_LCA}"
)

print(
    f"  TA  = {W_TA}"
)

print(
    f"  CT  = {W_CT}"
)


# ============================================================
# 7. CARGAR DATASET PRINCIPAL
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


print()

print("=" * 90)
print("DATASET PRINCIPAL")
print("=" * 90)


print(
    f"Muestras antes del merge: "
    f"{len(df)}"
)


# ============================================================
# 8. CARGAR Ag_mean40
# ============================================================

df_ag40 = pd.read_csv(
    AG40_PATH
)


# ============================================================
# PREPARAR INDICE DEL DATASET Ag_mean40
# ============================================================

if "Ag_mean40_cm2" not in df_ag40.columns:

    raise ValueError(
        "dataset_sintetico_200_agmean40.csv "
        "no contiene Ag_mean40_cm2.\n"
        f"Columnas disponibles: {df_ag40.columns.tolist()}"
    )


# ------------------------------------------------------------
# Si el archivo Ag_mean40 no tiene OriginalIndex,
# lo reconstruimos a partir del orden original de las
# 200 simulaciones.
# ------------------------------------------------------------

if "OriginalIndex" not in df_ag40.columns:

    print(
        "El archivo Ag_mean40 no contiene OriginalIndex."
    )

    print(
        "Se reconstruira a partir del orden "
        "original de las simulaciones."
    )


    # Dataset principal usa indices MATLAB 1...200
    if (
        df["OriginalIndex"].min() >= 1
        and
        df["OriginalIndex"].max() <= len(df_ag40)
    ):

        df_ag40["OriginalIndex"] = np.arange(
            1,
            len(df_ag40) + 1,
        )


        print(
            "OriginalIndex reconstruido como 1...N"
        )


    # Por seguridad, contemplamos tambien indice 0...199
    elif (
        df["OriginalIndex"].min() >= 0
        and
        df["OriginalIndex"].max() < len(df_ag40)
    ):

        df_ag40["OriginalIndex"] = np.arange(
            0,
            len(df_ag40),
        )


        print(
            "OriginalIndex reconstruido como 0...N-1"
        )


    else:

        raise ValueError(
            "No se pudo determinar automaticamente "
            "la correspondencia de OriginalIndex."
        )


df_ag40 = df_ag40[
    [
        "OriginalIndex",
        "Ag_mean40_cm2",
    ]
].copy()


# Evitar problemas por diferencias de tipo

df["OriginalIndex"] = (
    df["OriginalIndex"]
    .astype(int)
)


df_ag40["OriginalIndex"] = (
    df_ag40["OriginalIndex"]
    .astype(int)
)


df = df.merge(

    df_ag40,

    on="OriginalIndex",

    how="left",

    validate="one_to_one",
)


missing_ag40 = (
    df["Ag_mean40_cm2"]
    .isna()
    .sum()
)


print(
    f"Muestras despues del merge: "
    f"{len(df)}"
)


print(
    f"Ag_mean40 faltantes: "
    f"{missing_ag40}"
)


if missing_ag40 > 0:

    raise ValueError(
        "Hay muestras sin Ag_mean40 "
        "despues del merge."
    )


# ============================================================
# 9. COLUMNAS
# ============================================================

feature_columns = [

    "F0_Hz",

    "MFDR_L_s2",

    "OQ",

    "SQ",

    "ACFL_mL_s",

    "H1H2_dB",

    "SPL_dB",
]


target_columns = [

    "LCA",

    "TA",

    "CT",
]


X = df[
    feature_columns
].to_numpy(
    dtype=np.float32
)


y = df[
    target_columns
].to_numpy(
    dtype=np.float32
)


Ag_mean40_all = df[
    "Ag_mean40_cm2"
].to_numpy(
    dtype=np.float32
).reshape(
    -1,
    1,
)


# ============================================================
# 10. CARGAR MISMO SPLIT DEL BASELINE
# ============================================================

split = np.load(
    SPLIT_PATH
)


train_idx = split[
    "train_idx"
]

val_idx = split[
    "val_idx"
]

test_idx = split[
    "test_idx"
]


print()

print("=" * 90)
print("DIVISION DEL DATASET")
print("=" * 90)


print(
    f"Train:      "
    f"{len(train_idx)}"
)

print(
    f"Validation: "
    f"{len(val_idx)}"
)

print(
    f"Test:       "
    f"{len(test_idx)}"
)


# ============================================================
# 11. SEPARAR DATOS
# ============================================================

X_train = X[
    train_idx
]

X_val = X[
    val_idx
]

X_test = X[
    test_idx
]


y_train = y[
    train_idx
]

y_val = y[
    val_idx
]

y_test = y[
    test_idx
]


Ag_mean40_train = Ag_mean40_all[
    train_idx
]

Ag_mean40_val = Ag_mean40_all[
    val_idx
]

Ag_mean40_test = Ag_mean40_all[
    test_idx
]


# ============================================================
# 12. ESTADISTICAS Ag_mean40
# ============================================================

print()

print("=" * 90)
print("Ag_mean40 - TRAIN")
print("=" * 90)


print(
    f"Min     = "
    f"{np.min(Ag_mean40_train):.6f} cm^2"
)

print(
    f"Media   = "
    f"{np.mean(Ag_mean40_train):.6f} cm^2"
)

print(
    f"Mediana = "
    f"{np.median(Ag_mean40_train):.6f} cm^2"
)

print(
    f"Max     = "
    f"{np.max(Ag_mean40_train):.6f} cm^2"
)


# ============================================================
# 13. NORMALIZACION DE FEATURES
# ============================================================

scaler_X = StandardScaler()


X_train_scaled = scaler_X.fit_transform(
    X_train
)


X_val_scaled = scaler_X.transform(
    X_val
)


X_test_scaled = scaler_X.transform(
    X_test
)


# ============================================================
# 14. ECUACION ANALITICA A2 - NUMPY
# ============================================================

def calculate_A2_numpy(
    activations,
):

    LCA = activations[
        :,
        0
    ]

    TA = activations[
        :,
        1
    ]

    CT = activations[
        :,
        2
    ]


    epsilon = (

        0.6 * CT

        - 0.2 * TA

        - 0.2 * LCA

    )


    A2 = (

        A2_FACTOR

        * (
            1.0
            + epsilon
        )

        * (
            1.0
            - 2.0 * LCA
        )

    )


    return A2


# ============================================================
# 15. DIAGNOSTICO A2 REAL vs Ag_mean40
# ============================================================

A2_real_all = calculate_A2_numpy(
    y
)


A2_real_train = A2_real_all[
    train_idx
]

A2_real_val = A2_real_all[
    val_idx
]

A2_real_test = A2_real_all[
    test_idx
]


def print_A2_diagnostics(
    name,
    A2_values,
    Ag_mean40_values,
):

    Ag_mean40_values = (
        Ag_mean40_values
        .reshape(-1)
    )


    closed = (
        A2_values
        < 0
    )


    opened = (
        A2_values
        >= 0
    )


    print()

    print("=" * 90)

    print(
        f"DIAGNOSTICO A2 vs Ag_mean40 - "
        f"{name}"
    )

    print("=" * 90)


    print(
        f"N = {len(A2_values)}"
    )


    print(
        f"A2 min     = "
        f"{np.min(A2_values):.6f} cm^2"
    )


    print(
        f"A2 media   = "
        f"{np.mean(A2_values):.6f} cm^2"
    )


    print(
        f"A2 mediana = "
        f"{np.median(A2_values):.6f} cm^2"
    )


    print(
        f"A2 max     = "
        f"{np.max(A2_values):.6f} cm^2"
    )


    print()

    print(
        f"A2 < 0 "
        f"(cierre/sobreaduccion): "
        f"{np.sum(closed)} / "
        f"{len(A2_values)}"
    )


    print(
        f"A2 >= 0 "
        f"(comparables con Ag_mean40): "
        f"{np.sum(opened)} / "
        f"{len(A2_values)}"
    )


    if np.any(
        opened
    ):

        diff = (

            A2_values[
                opened
            ]

            -

            Ag_mean40_values[
                opened
            ]

        )


        mae = np.mean(
            np.abs(
                diff
            )
        )


        rmse = np.sqrt(
            np.mean(
                diff ** 2
            )
        )


        print()

        print(
            f"MAE A2 vs Ag_mean40 "
            f"(solo A2 >= 0) = "
            f"{mae:.6f} cm^2"
        )


        print(
            f"RMSE A2 vs Ag_mean40 "
            f"(solo A2 >= 0) = "
            f"{rmse:.6f} cm^2"
        )


print_A2_diagnostics(
    "TRAIN",
    A2_real_train,
    Ag_mean40_train,
)


print_A2_diagnostics(
    "VALIDATION",
    A2_real_val,
    Ag_mean40_val,
)


print_A2_diagnostics(
    "TEST",
    A2_real_test,
    Ag_mean40_test,
)


# ============================================================
# 16. TENSORES
# ============================================================

X_train_tensor = torch.tensor(

    X_train_scaled,

    dtype=torch.float32,
)


y_train_tensor = torch.tensor(

    y_train,

    dtype=torch.float32,
)


Ag_mean40_train_tensor = torch.tensor(

    Ag_mean40_train,

    dtype=torch.float32,
)


X_val_tensor = torch.tensor(

    X_val_scaled,

    dtype=torch.float32,

    device=device,
)


y_val_tensor = torch.tensor(

    y_val,

    dtype=torch.float32,

    device=device,
)


Ag_mean40_val_tensor = torch.tensor(

    Ag_mean40_val,

    dtype=torch.float32,

    device=device,
)


X_test_tensor = torch.tensor(

    X_test_scaled,

    dtype=torch.float32,

    device=device,
)


Ag_mean40_test_tensor = torch.tensor(

    Ag_mean40_test,

    dtype=torch.float32,

    device=device,
)


# ============================================================
# 17. DATALOADER
# ============================================================

train_dataset = TensorDataset(

    X_train_tensor,

    y_train_tensor,

    Ag_mean40_train_tensor,
)


generator = torch.Generator()


generator.manual_seed(
    SEED
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    generator=generator,
)


# ============================================================
# 18. RED PRINCIPAL
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                7,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32,
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                3,
            ),

            nn.Sigmoid(),
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# 19. ECUACION ANALITICA A2 - PYTORCH
#
# Participa en autograd.
# ============================================================

def calculate_A2_torch(
    activation_prediction,
):

    LCA = activation_prediction[
        :,
        0:1
    ]


    TA = activation_prediction[
        :,
        1:2
    ]


    CT = activation_prediction[
        :,
        2:3
    ]


    epsilon = (

        0.6 * CT

        - 0.2 * TA

        - 0.2 * LCA

    )


    A2 = (

        A2_FACTOR

        * (
            1.0
            + epsilon
        )

        * (
            1.0
            - 2.0 * LCA
        )

    )


    return A2


# ============================================================
# 20. LOSS SUPERVISADA PONDERADA
# ============================================================

def weighted_supervised_loss(
    prediction,
    target,
):

    loss_lca = torch.mean(

        (
            prediction[:, 0]
            - target[:, 0]
        ) ** 2

    )


    loss_ta = torch.mean(

        (
            prediction[:, 1]
            - target[:, 1]
        ) ** 2

    )


    loss_ct = torch.mean(

        (
            prediction[:, 2]
            - target[:, 2]
        ) ** 2

    )


    loss_sup = (

        W_LCA
        * loss_lca

        +

        W_TA
        * loss_ta

        +

        W_CT
        * loss_ct

    ) / (

        W_LCA
        + W_TA
        + W_CT

    )


    return (
        loss_sup,
        loss_lca,
        loss_ta,
        loss_ct,
    )


# ============================================================
# 21. LOSS FISICA A2 vs Ag_mean40
#
# A2 < 0:
#     cierre/sobreaduccion
#     NO se penaliza
#
# A2 >= 0:
#     A2 debe aproximarse a Ag_mean40
# ============================================================

def physics_loss_A2_mean40(
    activation_prediction,
    Ag_mean40_target,
):

    A2_pred = calculate_A2_torch(
        activation_prediction
    )


    # --------------------------------------------------------
    # Mascara:
    #
    # 1 si A2 >= 0
    # 0 si A2 < 0
    # --------------------------------------------------------

    open_mask = (
        A2_pred
        >= 0
    ).float()


    # --------------------------------------------------------
    # Error respecto al promedio real
    # --------------------------------------------------------

    error = (

        A2_pred

        - Ag_mean40_target

    )


    # --------------------------------------------------------
    # Solo se considera el error si A2 >= 0
    # --------------------------------------------------------

    squared_error = (

        open_mask

        * error ** 2

    )


    loss_A2_mean40 = torch.mean(
        squared_error
    )


    return (
        loss_A2_mean40,
        A2_pred,
        open_mask,
        error,
    )


# ============================================================
# 22. INICIALIZAR MODELO
# ============================================================

torch.manual_seed(
    SEED
)


model = MLP().to(
    device
)


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,
)


# ============================================================
# 23. ENTRENAMIENTO
# ============================================================

best_val_loss = np.inf

best_state = None

best_epoch = 0

epochs_without_improvement = 0


history_train_total = []

history_val_total = []


print()

print("=" * 90)

print(
    "ENTRENAMIENTO MODELO PRINCIPAL "
    "- A2 vs Ag_mean40"
)

print("=" * 90)


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()


    total_sum = 0.0

    sup_sum = 0.0

    lca_sum = 0.0

    ta_sum = 0.0

    ct_sum = 0.0

    A2_sum = 0.0


    for (
        X_batch,
        y_batch,
        Ag_mean40_batch,
    ) in train_loader:


        X_batch = X_batch.to(
            device
        )


        y_batch = y_batch.to(
            device
        )


        Ag_mean40_batch = (
            Ag_mean40_batch.to(
                device
            )
        )


        optimizer.zero_grad()


        prediction = model(
            X_batch
        )


        (
            loss_sup,
            loss_lca,
            loss_ta,
            loss_ct,

        ) = weighted_supervised_loss(

            prediction,

            y_batch,
        )


        (
            loss_A2,
            _,
            _,
            _,

        ) = physics_loss_A2_mean40(

            prediction,

            Ag_mean40_batch,
        )


        loss_total = (

            loss_sup

            +

            LAMBDA_A2_MEAN40
            * loss_A2

        )


        loss_total.backward()


        optimizer.step()


        batch_size = X_batch.size(
            0
        )


        total_sum += (
            loss_total.item()
            * batch_size
        )


        sup_sum += (
            loss_sup.item()
            * batch_size
        )


        lca_sum += (
            loss_lca.item()
            * batch_size
        )


        ta_sum += (
            loss_ta.item()
            * batch_size
        )


        ct_sum += (
            loss_ct.item()
            * batch_size
        )


        A2_sum += (
            loss_A2.item()
            * batch_size
        )


    n_train = len(
        train_loader.dataset
    )


    train_total = (
        total_sum
        / n_train
    )


    train_sup = (
        sup_sum
        / n_train
    )


    train_lca = (
        lca_sum
        / n_train
    )


    train_ta = (
        ta_sum
        / n_train
    )


    train_ct = (
        ct_sum
        / n_train
    )


    train_A2 = (
        A2_sum
        / n_train
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    with torch.no_grad():

        val_prediction = model(
            X_val_tensor
        )


        (
            val_sup,
            val_lca,
            val_ta,
            val_ct,

        ) = weighted_supervised_loss(

            val_prediction,

            y_val_tensor,
        )


        (
            val_A2,
            _,
            _,
            _,

        ) = physics_loss_A2_mean40(

            val_prediction,

            Ag_mean40_val_tensor,
        )


        val_total = (

            val_sup

            +

            LAMBDA_A2_MEAN40
            * val_A2

        )


    val_total_value = (
        val_total.item()
    )


    history_train_total.append(
        train_total
    )


    history_val_total.append(
        val_total_value
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if val_total_value < (

        best_val_loss
        - MIN_DELTA

    ):

        best_val_loss = (
            val_total_value
        )


        best_epoch = epoch


        best_state = {

            key: value.cpu().clone()

            for key, value
            in model.state_dict().items()

        }


        epochs_without_improvement = 0


    else:

        epochs_without_improvement += 1


    if (
        epoch == 1
        or
        epoch % 50 == 0
    ):

        print(

            f"Epoch {epoch:4d} | "

            f"Total={train_total:.6f} | "

            f"Sup={train_sup:.6f} | "

            f"LCA={train_lca:.6f} | "

            f"TA={train_ta:.6f} | "

            f"CT={train_ct:.6f} | "

            f"A2mean40={train_A2:.6f} | "

            f"Val={val_total_value:.6f}"

        )


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            f"Early stopping en epoch "
            f"{epoch}"
        )

        break


# ============================================================
# 24. RECUPERAR MEJOR MODELO
# ============================================================

if best_state is None:

    raise RuntimeError(
        "No se encontro un estado valido del modelo."
    )


model.load_state_dict(
    best_state
)


model = model.to(
    device
)


print()

print(
    f"Mejor epoch: "
    f"{best_epoch}"
)


print(
    f"Mejor validation loss total: "
    f"{best_val_loss:.6f}"
)


# ============================================================
# 25. TEST
# ============================================================

model.eval()


with torch.no_grad():

    y_pred_tensor = model(
        X_test_tensor
    )


    (
        test_loss_A2,
        A2_pred_tensor,
        open_mask_tensor,
        A2_error_tensor,

    ) = physics_loss_A2_mean40(

        y_pred_tensor,

        Ag_mean40_test_tensor,
    )


y_pred = (

    y_pred_tensor
    .cpu()
    .numpy()

)


A2_pred = (

    A2_pred_tensor
    .cpu()
    .numpy()
    .reshape(-1)

)


open_mask = (

    open_mask_tensor
    .cpu()
    .numpy()
    .reshape(-1)

)


A2_error = (

    A2_error_tensor
    .cpu()
    .numpy()
    .reshape(-1)

)


Ag_mean40_test_flat = (

    Ag_mean40_test
    .reshape(-1)

)


# ============================================================
# 26. METRICAS TEST
# ============================================================

print()

print("=" * 90)
print("RESULTADOS TEST")
print("=" * 90)


for i, target in enumerate(
    target_columns
):

    y_true_i = y_test[
        :,
        i
    ]


    y_pred_i = y_pred[
        :,
        i
    ]


    mse = mean_squared_error(
        y_true_i,
        y_pred_i,
    )


    rmse = np.sqrt(
        mse
    )


    mae = mean_absolute_error(
        y_true_i,
        y_pred_i,
    )


    r2 = r2_score(
        y_true_i,
        y_pred_i,
    )


    print()

    print(
        target
    )


    print(
        f"  MAE  = "
        f"{mae:.6f}"
    )


    print(
        f"  RMSE = "
        f"{rmse:.6f}"
    )


    print(
        f"  R2   = "
        f"{r2:.6f}"
    )


# ============================================================
# 27. METRICAS GLOBALES
# ============================================================

global_mse = mean_squared_error(
    y_test,
    y_pred,
)


global_rmse = np.sqrt(
    global_mse
)


global_mae = mean_absolute_error(
    y_test,
    y_pred,
)


print()

print("-" * 90)


print(
    f"MAE global  = "
    f"{global_mae:.6f}"
)


print(
    f"RMSE global = "
    f"{global_rmse:.6f}"
)


# ============================================================
# 28. CONSISTENCIA FISICA TEST
# ============================================================

A2_closed = (
    A2_pred
    < 0
)


A2_open = (
    A2_pred
    >= 0
)


print()

print("=" * 90)

print(
    "CONSISTENCIA FISICA "
    "A2 vs Ag_mean40 - TEST"
)

print("=" * 90)


print(
    f"Loss A2-mean40 test = "
    f"{test_loss_A2.item():.8f}"
)


print()

print(
    f"A2 predicho min   = "
    f"{np.min(A2_pred):.6f} cm^2"
)


print(
    f"A2 predicho medio = "
    f"{np.mean(A2_pred):.6f} cm^2"
)


print(
    f"A2 predicho max   = "
    f"{np.max(A2_pred):.6f} cm^2"
)


print()

print(
    f"A2 < 0 "
    f"(cierre/sobreaduccion): "
    f"{np.sum(A2_closed)} / "
    f"{len(test_idx)}"
)


print(
    f"A2 >= 0 "
    f"(comparados con Ag_mean40): "
    f"{np.sum(A2_open)} / "
    f"{len(test_idx)}"
)


if np.any(
    A2_open
):

    area_mae = np.mean(

        np.abs(

            A2_pred[
                A2_open
            ]

            -

            Ag_mean40_test_flat[
                A2_open
            ]

        )

    )


    area_rmse = np.sqrt(

        np.mean(

            (

                A2_pred[
                    A2_open
                ]

                -

                Ag_mean40_test_flat[
                    A2_open
                ]

            ) ** 2

        )

    )


    print()

    print(
        f"MAE A2 vs Ag_mean40 "
        f"(A2 >= 0) = "
        f"{area_mae:.6f} cm^2"
    )


    print(
        f"RMSE A2 vs Ag_mean40 "
        f"(A2 >= 0) = "
        f"{area_rmse:.6f} cm^2"
    )


# ============================================================
# 29. PREDICCIONES TEST
# ============================================================

print()

print("=" * 175)
print("PREDICCIONES DEL TEST")
print("=" * 175)


for i in range(
    len(test_idx)
):

    original_idx = int(

        df.iloc[
            test_idx[i]
        ][
            "OriginalIndex"
        ]

    )


    if A2_pred[i] < 0:

        estado = "CIERRE"

    else:

        estado = "ABIERTO"


    print(

        f"Index {original_idx:3d} | "

        f"LCA "
        f"{y_test[i,0]:.4f}"
        f"->{y_pred[i,0]:.4f} | "

        f"TA "
        f"{y_test[i,1]:.4f}"
        f"->{y_pred[i,1]:.4f} | "

        f"CT "
        f"{y_test[i,2]:.4f}"
        f"->{y_pred[i,2]:.4f} | "

        f"A2="
        f"{A2_pred[i]:.4f} | "

        f"Ag_mean40="
        f"{Ag_mean40_test_flat[i]:.4f} | "

        f"{estado}"

    )


# ============================================================
# 30. GUARDAR MODELO
# ============================================================

torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "feature_columns":
            feature_columns,

        "target_columns":
            target_columns,

        "train_idx":
            train_idx,

        "val_idx":
            val_idx,

        "test_idx":
            test_idx,

        "scaler_mean":
            scaler_X.mean_,

        "scaler_scale":
            scaler_X.scale_,

        "L0_cm":
            L0_CM,

        "A2_factor":
            A2_FACTOR,

        "lambda_A2_mean40":
            LAMBDA_A2_MEAN40,

        "A2_negative_interpretation":
            "cierre_sobreaduccion_sin_penalizacion",

        "W_LCA":
            W_LCA,

        "W_TA":
            W_TA,

        "W_CT":
            W_CT,

        "seed":
            SEED,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

    },

    OUTPUT_DIR
    / "modelo_mlp_A2_mean40.pt",
)


# ============================================================
# 31. GRAFICO LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history_train_total,
    label="Train total",
)


plt.plot(
    history_val_total,
    label="Validation total",
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "MLP fisicamente informada "
    "- A2 vs Ag_mean40"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "loss_total.png",

    dpi=300,
)


plt.close()


# ============================================================
# 32. FIN
# ============================================================

print()

print("=" * 90)
print("FIN")
print("=" * 90)


print()

print(
    "Configuracion:"
)


print(
    f"  L0 = "
    f"{L0_CM} cm"
)


print(
    f"  A2 factor = "
    f"{A2_FACTOR:.6f} cm^2"
)


print(
    "  A2 < 0 = "
    "cierre/sobreaduccion "
    "(sin penalizacion)"
)


print(
    "  A2 >= 0 = "
    "comparado contra Ag_mean40"
)


print(
    f"  W_LCA = "
    f"{W_LCA}"
)


print(
    f"  W_TA  = "
    f"{W_TA}"
)


print(
    f"  W_CT  = "
    f"{W_CT}"
)


print(
    f"  Lambda A2-mean40 = "
    f"{LAMBDA_A2_MEAN40}"
)


print()

print(
    "Modelo y grafico guardados en:"
)


print(
    OUTPUT_DIR
)