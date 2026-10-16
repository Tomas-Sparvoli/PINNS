# ============================================================
# MODELO HIBRIDO FISICAMENTE INFORMADO
# AREAS GLOTICAS + LOSS SUPERVISADA PONDERADA
#
# Entradas:
#   F0, MFDR, OQ, SQ, ACFL, H1-H2, SPL
#
# Salidas:
#   LCA, TA, CT
#
# Restricciones empiricas:
#
#   0.01 <= Ag_mean <= 0.09 cm^2
#   Ag_max <= 0.27 cm^2
#
# Peso fisica:
#
#   lambda_mean = 1.0
#   lambda_max  = 1.0
#
# Pesos supervisados:
#
#   LCA = 1.5
#   TA  = 1.5
#   CT  = 1.0
#
# ============================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

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


BASE_DIR = Path(__file__).resolve().parent


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
    / "resultados_mlp_fisica_ponderada_200"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. HIPERPARAMETROS RED PRINCIPAL
# ============================================================

BATCH_SIZE = 16

LEARNING_RATE = 1e-3

EPOCHS = 2000

PATIENCE = 150

MIN_DELTA = 1e-6


# ============================================================
# 3. HIPERPARAMETROS SURROGATES
# ============================================================

SURROGATE_LR = 1e-3

SURROGATE_EPOCHS = 3000

SURROGATE_PATIENCE = 200


# ============================================================
# 4. RESTRICCIONES FISICAS
# ============================================================

AG_MEAN_MIN = 0.01

AG_MEAN_MAX = 0.09

AG_MAX_LIMIT = 0.27


LAMBDA_MEAN = 1.0

LAMBDA_MAX = 1.0


# ============================================================
# 5. PESOS LOSS SUPERVISADA
# ============================================================

W_LCA = 1.5

W_TA = 1.5

W_CT = 1.0


# ============================================================
# 6. REPRODUCIBILIDAD
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 7. DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()

print("=" * 80)

print(
    "MODELO FISICAMENTE INFORMADO "
    "- AREAS GLOTICAS + LOSS PONDERADA"
)

print("=" * 80)


print(
    f"Dispositivo: {device}"
)

print(
    f"Seed: {SEED}"
)


print()

print(
    f"Ag_mean permitido: "
    f"[{AG_MEAN_MIN:.3f}, "
    f"{AG_MEAN_MAX:.3f}] cm^2"
)


print(
    f"Ag_max permitido: "
    f"<= {AG_MAX_LIMIT:.3f} cm^2"
)


print()

print(
    f"Lambda mean: {LAMBDA_MEAN}"
)

print(
    f"Lambda max:  {LAMBDA_MAX}"
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
# 8. CARGAR DATASET
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


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


Ag_mean_all = df[
    "Ag_mean_cm2"
].to_numpy(
    dtype=np.float32
).reshape(
    -1,
    1,
)


Ag_max_all = df[
    "Ag_max_cm2"
].to_numpy(
    dtype=np.float32
).reshape(
    -1,
    1,
)


print()

print("=" * 80)

print("DATASET")

print("=" * 80)


print(
    f"Muestras totales: {len(df)}"
)


# ============================================================
# 9. CARGAR MISMO SPLIT DEL BASELINE
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

print("=" * 80)

print("DIVISION DEL DATASET")

print("=" * 80)


print(
    f"Train:      {len(train_idx)}"
)

print(
    f"Validation: {len(val_idx)}"
)

print(
    f"Test:       {len(test_idx)}"
)


# ============================================================
# 10. SEPARAR DATOS
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


Ag_mean_train = Ag_mean_all[
    train_idx
]


Ag_mean_val = Ag_mean_all[
    val_idx
]


Ag_mean_test = Ag_mean_all[
    test_idx
]


Ag_max_train = Ag_max_all[
    train_idx
]


Ag_max_val = Ag_max_all[
    val_idx
]


Ag_max_test = Ag_max_all[
    test_idx
]


# ============================================================
# 11. NORMALIZACION DE FEATURES
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
# 12. NORMALIZACION DE AREAS
# ============================================================

Ag_mean_mu = float(
    np.mean(
        Ag_mean_train
    )
)


Ag_mean_sigma = float(
    np.std(
        Ag_mean_train
    )
)


Ag_max_mu = float(
    np.mean(
        Ag_max_train
    )
)


Ag_max_sigma = float(
    np.std(
        Ag_max_train
    )
)


Ag_mean_train_scaled = (
    Ag_mean_train
    - Ag_mean_mu
) / Ag_mean_sigma


Ag_mean_val_scaled = (
    Ag_mean_val
    - Ag_mean_mu
) / Ag_mean_sigma


Ag_max_train_scaled = (
    Ag_max_train
    - Ag_max_mu
) / Ag_max_sigma


Ag_max_val_scaled = (
    Ag_max_val
    - Ag_max_mu
) / Ag_max_sigma


print()

print("=" * 80)

print("ESTADISTICAS TRAIN")

print("=" * 80)


print(
    f"Ag_mean: "
    f"media={Ag_mean_mu:.6f} | "
    f"std={Ag_mean_sigma:.6f}"
)


print(
    f"Ag_max:  "
    f"media={Ag_max_mu:.6f} | "
    f"std={Ag_max_sigma:.6f}"
)


# ============================================================
# 13. MODELO SURROGATE
# ============================================================

class AreaSurrogate(nn.Module):

    def __init__(self):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                3,
                32,
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                32,
            ),

            nn.Tanh(),

            nn.Linear(
                32,
                1,
            ),
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# 14. ENTRENAMIENTO SURROGATES
# ============================================================

def train_surrogate(
    model,
    target_train,
    target_val,
    name,
):


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=SURROGATE_LR,
    )


    criterion = nn.MSELoss()


    y_train_tensor = torch.tensor(

        y_train,

        dtype=torch.float32,

        device=device,
    )


    y_val_tensor = torch.tensor(

        y_val,

        dtype=torch.float32,

        device=device,
    )


    target_train_tensor = torch.tensor(

        target_train,

        dtype=torch.float32,

        device=device,
    )


    target_val_tensor = torch.tensor(

        target_val,

        dtype=torch.float32,

        device=device,
    )


    best_val = np.inf

    best_state = None

    best_epoch = 0

    no_improvement = 0


    print()

    print("=" * 80)

    print(
        f"ENTRENAMIENTO SURROGATE {name}"
    )

    print("=" * 80)


    for epoch in range(
        1,
        SURROGATE_EPOCHS + 1,
    ):


        model.train()


        optimizer.zero_grad()


        pred_train = model(
            y_train_tensor
        )


        train_loss = criterion(

            pred_train,

            target_train_tensor,
        )


        train_loss.backward()


        optimizer.step()


        model.eval()


        with torch.no_grad():


            pred_val = model(
                y_val_tensor
            )


            val_loss = criterion(

                pred_val,

                target_val_tensor,
            ).item()


        if val_loss < (
            best_val
            - MIN_DELTA
        ):


            best_val = val_loss

            best_epoch = epoch


            best_state = {

                key: value.cpu().clone()

                for key, value
                in model.state_dict().items()

            }


            no_improvement = 0


        else:


            no_improvement += 1


        if (
            epoch == 1
            or
            epoch % 200 == 0
        ):


            print(

                f"Epoch {epoch:4d} | "
                f"Train={train_loss.item():.6f} | "
                f"Val={val_loss:.6f}"

            )


        if (
            no_improvement
            >= SURROGATE_PATIENCE
        ):


            print()

            print(

                f"Early stopping {name}: "
                f"epoch {epoch}"

            )

            break


    model.load_state_dict(
        best_state
    )


    model.to(
        device
    )


    model.eval()


    for parameter in model.parameters():

        parameter.requires_grad = False


    print()

    print(
        f"Mejor epoch {name}: "
        f"{best_epoch}"
    )


    print(
        f"Mejor val loss {name}: "
        f"{best_val:.6f}"
    )


    return model


# ============================================================
# 15. SURROGATE Ag_mean
# ============================================================

torch.manual_seed(
    SEED
)


mean_model = AreaSurrogate().to(
    device
)


mean_model = train_surrogate(

    mean_model,

    Ag_mean_train_scaled,

    Ag_mean_val_scaled,

    "Ag_mean",
)


# ============================================================
# 16. SURROGATE Ag_max
# ============================================================

torch.manual_seed(
    SEED + 1
)


max_model = AreaSurrogate().to(
    device
)


max_model = train_surrogate(

    max_model,

    Ag_max_train_scaled,

    Ag_max_val_scaled,

    "Ag_max",
)


# ============================================================
# 17. EVALUAR SURROGATES EN TEST
# ============================================================

y_test_tensor_surrogate = torch.tensor(

    y_test,

    dtype=torch.float32,

    device=device,
)


with torch.no_grad():


    Ag_mean_test_pred_scaled = mean_model(
        y_test_tensor_surrogate
    )


    Ag_max_test_pred_scaled = max_model(
        y_test_tensor_surrogate
    )


Ag_mean_test_pred = (

    Ag_mean_test_pred_scaled
    .cpu()
    .numpy()

    * Ag_mean_sigma

    + Ag_mean_mu
)


Ag_max_test_pred = (

    Ag_max_test_pred_scaled
    .cpu()
    .numpy()

    * Ag_max_sigma

    + Ag_max_mu
)


mean_mae = mean_absolute_error(

    Ag_mean_test,

    Ag_mean_test_pred,
)


mean_rmse = np.sqrt(

    mean_squared_error(

        Ag_mean_test,

        Ag_mean_test_pred,
    )
)


mean_r2 = r2_score(

    Ag_mean_test,

    Ag_mean_test_pred,
)


max_mae = mean_absolute_error(

    Ag_max_test,

    Ag_max_test_pred,
)


max_rmse = np.sqrt(

    mean_squared_error(

        Ag_max_test,

        Ag_max_test_pred,
    )
)


max_r2 = r2_score(

    Ag_max_test,

    Ag_max_test_pred,
)


print()

print("=" * 80)

print("SURROGATE Ag_mean - TEST")

print("=" * 80)


print(
    f"MAE  = {mean_mae:.6f} cm^2"
)

print(
    f"RMSE = {mean_rmse:.6f} cm^2"
)

print(
    f"R2   = {mean_r2:.6f}"
)


print()

print("=" * 80)

print("SURROGATE Ag_max - TEST")

print("=" * 80)


print(
    f"MAE  = {max_mae:.6f} cm^2"
)

print(
    f"RMSE = {max_rmse:.6f} cm^2"
)

print(
    f"R2   = {max_r2:.6f}"
)


# ============================================================
# 18. PREPARAR TENSORES
# ============================================================

X_train_tensor = torch.tensor(

    X_train_scaled,

    dtype=torch.float32,
)


y_train_tensor = torch.tensor(

    y_train,

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


X_test_tensor = torch.tensor(

    X_test_scaled,

    dtype=torch.float32,

    device=device,
)


# ============================================================
# 19. DATALOADER
# ============================================================

train_dataset = TensorDataset(

    X_train_tensor,

    y_train_tensor,
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
# 20. RED PRINCIPAL
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
# 21. LOSS SUPERVISADA PONDERADA
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


    loss_total = (

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
        loss_total,
        loss_lca,
        loss_ta,
        loss_ct,
    )


# ============================================================
# 22. LOSS FISICA
# ============================================================

def physics_loss(
    activation_prediction,
):


    # --------------------------------------------------------
    # Ag_mean
    # --------------------------------------------------------

    Ag_mean_scaled = mean_model(
        activation_prediction
    )


    Ag_mean_pred = (

        Ag_mean_scaled

        * Ag_mean_sigma

        + Ag_mean_mu

    )


    # --------------------------------------------------------
    # Ag_max
    # --------------------------------------------------------

    Ag_max_scaled = max_model(
        activation_prediction
    )


    Ag_max_pred = (

        Ag_max_scaled

        * Ag_max_sigma

        + Ag_max_mu

    )


    # --------------------------------------------------------
    # Restriccion Ag_mean
    # --------------------------------------------------------

    mean_lower_excess = torch.relu(

        AG_MEAN_MIN
        - Ag_mean_pred

    )


    mean_upper_excess = torch.relu(

        Ag_mean_pred
        - AG_MEAN_MAX

    )


    mean_range = (

        AG_MEAN_MAX
        - AG_MEAN_MIN

    )


    loss_mean = torch.mean(

        (
            mean_lower_excess
            / mean_range
        ) ** 2

        +

        (
            mean_upper_excess
            / mean_range
        ) ** 2

    )


    # --------------------------------------------------------
    # Restriccion Ag_max
    # --------------------------------------------------------

    max_excess = torch.relu(

        Ag_max_pred
        - AG_MAX_LIMIT

    )


    loss_max = torch.mean(

        (
            max_excess
            / AG_MAX_LIMIT
        ) ** 2

    )


    return (
        loss_mean,
        loss_max,
        Ag_mean_pred,
        Ag_max_pred,
    )


# ============================================================
# 23. INICIALIZAR MODELO PRINCIPAL
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
# 24. ENTRENAMIENTO
# ============================================================

best_val_loss = np.inf

best_state = None

best_epoch = 0

epochs_without_improvement = 0


history_train_total = []

history_val_total = []


print()

print("=" * 80)

print(
    "ENTRENAMIENTO MODELO FISICAMENTE INFORMADO "
    "+ LOSS PONDERADA"
)

print("=" * 80)


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

    mean_sum = 0.0

    max_sum = 0.0


    for X_batch, y_batch in train_loader:


        X_batch = X_batch.to(
            device
        )


        y_batch = y_batch.to(
            device
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
            loss_mean,
            loss_max,
            _,
            _,

        ) = physics_loss(
            prediction
        )


        loss_total = (

            loss_sup

            +

            LAMBDA_MEAN
            * loss_mean

            +

            LAMBDA_MAX
            * loss_max

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


        mean_sum += (
            loss_mean.item()
            * batch_size
        )


        max_sum += (
            loss_max.item()
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


    train_mean = (
        mean_sum
        / n_train
    )


    train_max = (
        max_sum
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
            val_mean,
            val_max,
            _,
            _,

        ) = physics_loss(
            val_prediction
        )


        val_total = (

            val_sup

            +

            LAMBDA_MEAN
            * val_mean

            +

            LAMBDA_MAX
            * val_max

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

            f"Mean={train_mean:.6f} | "

            f"Max={train_max:.6f} | "

            f"Val={val_total_value:.6f}"

        )


    if (
        epochs_without_improvement
        >= PATIENCE
    ):


        print()

        print(
            f"Early stopping en epoch {epoch}"
        )


        break


# ============================================================
# 25. RECUPERAR MEJOR MODELO
# ============================================================

model.load_state_dict(
    best_state
)


model = model.to(
    device
)


print()

print(
    f"Mejor epoch: {best_epoch}"
)


print(
    f"Mejor validation loss total: "
    f"{best_val_loss:.6f}"
)


# ============================================================
# 26. TEST
# ============================================================

model.eval()


with torch.no_grad():


    y_pred_tensor = model(
        X_test_tensor
    )


    (
        test_loss_mean,
        test_loss_max,
        Ag_mean_pred_tensor,
        Ag_max_pred_tensor,

    ) = physics_loss(
        y_pred_tensor
    )


y_pred = (

    y_pred_tensor
    .cpu()
    .numpy()

)


Ag_mean_pred = (

    Ag_mean_pred_tensor
    .cpu()
    .numpy()
    .reshape(-1)

)


Ag_max_pred = (

    Ag_max_pred_tensor
    .cpu()
    .numpy()
    .reshape(-1)

)


# ============================================================
# 27. METRICAS TEST
# ============================================================

print()

print("=" * 80)

print("RESULTADOS TEST")

print("=" * 80)


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
        f"  MAE  = {mae:.6f}"
    )


    print(
        f"  RMSE = {rmse:.6f}"
    )


    print(
        f"  R2   = {r2:.6f}"
    )


# ============================================================
# 28. METRICAS GLOBALES
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

print("-" * 80)


print(
    f"MAE global  = {global_mae:.6f}"
)


print(
    f"RMSE global = {global_rmse:.6f}"
)


# ============================================================
# 29. CONSISTENCIA FISICA
# ============================================================

mean_lower_violation = (

    Ag_mean_pred
    < AG_MEAN_MIN

)


mean_upper_violation = (

    Ag_mean_pred
    > AG_MEAN_MAX

)


mean_violation = (

    mean_lower_violation

    |

    mean_upper_violation

)


max_violation = (

    Ag_max_pred
    > AG_MAX_LIMIT

)


mean_violations = int(

    np.sum(
        mean_violation
    )

)


max_violations = int(

    np.sum(
        max_violation
    )

)


print()

print("=" * 80)

print("CONSISTENCIA FISICA - TEST")

print("=" * 80)


print(
    f"Ag_mean permitido: "
    f"[{AG_MEAN_MIN:.3f}, "
    f"{AG_MEAN_MAX:.3f}] cm^2"
)


print(
    f"Violaciones Ag_mean: "
    f"{mean_violations} / {len(test_idx)}"
)


print()

print(
    f"Ag_mean predicho min   = "
    f"{np.min(Ag_mean_pred):.6f}"
)


print(
    f"Ag_mean predicho medio = "
    f"{np.mean(Ag_mean_pred):.6f}"
)


print(
    f"Ag_mean predicho max   = "
    f"{np.max(Ag_mean_pred):.6f}"
)


print()

print(
    f"Ag_max permitido: "
    f"<= {AG_MAX_LIMIT:.3f} cm^2"
)


print(
    f"Violaciones Ag_max: "
    f"{max_violations} / {len(test_idx)}"
)


print()

print(
    f"Ag_max predicho min   = "
    f"{np.min(Ag_max_pred):.6f}"
)


print(
    f"Ag_max predicho medio = "
    f"{np.mean(Ag_max_pred):.6f}"
)


print(
    f"Ag_max predicho max   = "
    f"{np.max(Ag_max_pred):.6f}"
)


# ============================================================
# 30. PREDICCIONES TEST
# ============================================================

print()

print("=" * 130)

print("PREDICCIONES DEL TEST")

print("=" * 130)


for i in range(
    len(test_idx)
):


    if "OriginalIndex" in df.columns:


        original_idx = df.iloc[
            test_idx[i]
        ][
            "OriginalIndex"
        ]


    else:


        original_idx = test_idx[i]


    print(

        f"Index {int(original_idx):3d} | "

        f"LCA "
        f"{y_test[i,0]:.4f}"
        f"->{y_pred[i,0]:.4f} | "

        f"TA "
        f"{y_test[i,1]:.4f}"
        f"->{y_pred[i,1]:.4f} | "

        f"CT "
        f"{y_test[i,2]:.4f}"
        f"->{y_pred[i,2]:.4f} | "

        f"Ag_mean="
        f"{Ag_mean_pred[i]:.4f} | "

        f"Ag_max="
        f"{Ag_max_pred[i]:.4f}"

    )


# ============================================================
# 31. GUARDAR MODELO
# ============================================================

torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "mean_surrogate_state_dict":
            mean_model.state_dict(),

        "max_surrogate_state_dict":
            max_model.state_dict(),

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

        "Ag_mean_min":
            AG_MEAN_MIN,

        "Ag_mean_max":
            AG_MEAN_MAX,

        "Ag_max_limit":
            AG_MAX_LIMIT,

        "lambda_mean":
            LAMBDA_MEAN,

        "lambda_max":
            LAMBDA_MAX,

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
    / "modelo_mlp_fisica_ponderada.pt",
)


# ============================================================
# 32. GRAFICO LOSS
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
    "- loss supervisada ponderada"
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
# 33. FIN
# ============================================================

print()

print("=" * 80)

print("FIN")

print("=" * 80)


print()

print(
    "Configuracion:"
)


print(
    f"  W_LCA = {W_LCA}"
)

print(
    f"  W_TA  = {W_TA}"
)

print(
    f"  W_CT  = {W_CT}"
)


print(
    f"  Lambda mean = {LAMBDA_MEAN}"
)

print(
    f"  Lambda max  = {LAMBDA_MAX}"
)


print()

print(
    "Modelo y grafico guardados en:"
)


print(
    OUTPUT_DIR
)