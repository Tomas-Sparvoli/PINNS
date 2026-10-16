# ============================================================
# MLP SUPERVISADA V2
# DATASET SINTETICO DE FONACION
#
# Entradas:
#   F0
#   MFDR
#   OQ
#   SQ
#   ACFL
#   H1-H2
#   SPL
#
# Salidas:
#   LCA
#   TA
#   CT
#
# Objetivo:
#   Construir un baseline supervisado con:
#       train / validation / test
#       early stopping sobre validation
#       metricas finales sobre test
#       curvas de aprendizaje
#       graficos real vs predicho
#
# Este modelo servira como referencia para comparar posteriormente
# contra el modelo con perdida fisica.
# ============================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
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


OUTPUT_DIR = Path(__file__).resolve().parent / "resultados_mlp_200_v2"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TEST_SIZE = 0.15

VAL_SIZE = 0.15

BATCH_SIZE = 16

LEARNING_RATE = 1e-3

EPOCHS = 2000

PATIENCE = 150

MIN_DELTA = 1e-6


# ============================================================
# 2. REPRODUCIBILIDAD
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DISPOSITIVO
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()

print("=" * 60)

print("CONFIGURACION")

print("=" * 60)

print(
    f"Dispositivo: {device}"
)

print(
    f"Seed: {SEED}"
)


# ============================================================
# 4. CARGAR DATASET
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


print()

print("=" * 60)

print("DATASET")

print("=" * 60)

print(
    f"Muestras totales: {len(df)}"
)

print()

print(
    "Columnas disponibles:"
)

print(
    df.columns.tolist()
)


# ============================================================
# 5. DEFINIR FEATURES Y TARGETS
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


# ============================================================
# 6. VERIFICAR COLUMNAS
# ============================================================

missing_features = [

    col

    for col in feature_columns

    if col not in df.columns
]


missing_targets = [

    col

    for col in target_columns

    if col not in df.columns
]


if missing_features:

    raise ValueError(
        f"Faltan features: {missing_features}"
    )


if missing_targets:

    raise ValueError(
        f"Faltan targets: {missing_targets}"
    )


# ============================================================
# 7. EXTRAER DATOS
# ============================================================

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


indices = np.arange(
    len(df)
)


# ============================================================
# 8. SPLIT TRAIN / VALIDATION / TEST
# ============================================================

# Primero separamos test.

train_val_idx, test_idx = train_test_split(

    indices,

    test_size=TEST_SIZE,

    random_state=SEED,

    shuffle=True,
)


# Ahora calculamos cuanto debe ocupar validation
# dentro del bloque train_val.

val_relative_size = (

    VAL_SIZE

    / (

        1.0

        - TEST_SIZE

    )
)


train_idx, val_idx = train_test_split(

    train_val_idx,

    test_size=val_relative_size,

    random_state=SEED,

    shuffle=True,
)


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


print()

print("=" * 60)

print("DIVISION DEL DATASET")

print("=" * 60)

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
# 9. NORMALIZACION DE ENTRADAS
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
# 10. CONVERSION A TENSORES
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
)


y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32,
)


X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)


y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32,
)


# ============================================================
# 11. DATALOADER
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
# 12. MODELO
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
# 13. INICIALIZAR MODELO
# ============================================================

torch.manual_seed(
    SEED
)


model = MLP().to(
    device
)


# ============================================================
# 14. LOSS Y OPTIMIZADOR
# ============================================================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE,
)


# ============================================================
# 15. ENTRENAMIENTO
# ============================================================

best_val_loss = np.inf

best_state = None

best_epoch = 0

epochs_without_improvement = 0


history_train = []

history_val = []


print()

print("=" * 60)

print("ENTRENAMIENTO")

print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1,
):


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()


    train_loss_sum = 0.0


    for X_batch, y_batch in train_loader:


        X_batch = X_batch.to(
            device
        )


        y_batch = y_batch.to(
            device
        )


        optimizer.zero_grad()


        predictions = model(
            X_batch
        )


        loss = criterion(

            predictions,

            y_batch,
        )


        loss.backward()


        optimizer.step()


        train_loss_sum += (

            loss.item()

            * X_batch.size(0)

        )


    train_loss = (

        train_loss_sum

        / len(
            train_loader.dataset
        )

    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():


        val_predictions = model(

            X_val_tensor.to(
                device
            )

        )


        val_loss = criterion(

            val_predictions,

            y_val_tensor.to(
                device
            ),

        ).item()


    history_train.append(
        train_loss
    )


    history_val.append(
        val_loss
    )


    # --------------------------------------------------------
    # GUARDAR MEJOR MODELO
    # --------------------------------------------------------

    if val_loss < (

        best_val_loss

        - MIN_DELTA

    ):


        best_val_loss = val_loss


        best_epoch = epoch


        best_state = {

            key: value.cpu().clone()

            for key, value

            in model.state_dict().items()

        }


        epochs_without_improvement = 0


    else:


        epochs_without_improvement += 1


    # --------------------------------------------------------
    # MOSTRAR PROGRESO
    # --------------------------------------------------------

    if (

        epoch == 1

        or

        epoch % 50 == 0

    ):


        print(

            f"Epoch {epoch:4d} | "

            f"Train = {train_loss:.6f} | "

            f"Val = {val_loss:.6f}"

        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

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
# 16. RECUPERAR MEJOR MODELO
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
    f"Mejor validation loss: {best_val_loss:.6f}"
)


# ============================================================
# 17. PREDICCION SOBRE TEST
# ============================================================

model.eval()


with torch.no_grad():


    y_pred_tensor = model(

        X_test_tensor.to(
            device
        )

    )


y_pred = (

    y_pred_tensor

    .cpu()

    .numpy()

)


# ============================================================
# 18. METRICAS POR MUSCULO
# ============================================================

print()

print("=" * 60)

print("RESULTADOS TEST")

print("=" * 60)


metrics_rows = []


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


    metrics_rows.append({

        "Target": target,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,
    })


# ============================================================
# 19. METRICAS GLOBALES
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

print("-" * 60)


print(
    f"MAE global  = {global_mae:.6f}"
)


print(
    f"RMSE global = {global_rmse:.6f}"
)


# ============================================================
# 20. GUARDAR METRICAS
# ============================================================

metrics_df = pd.DataFrame(
    metrics_rows
)


metrics_df.to_csv(

    OUTPUT_DIR / "metricas_test.csv",

    index=False,
)


# ============================================================
# 21. GUARDAR PREDICCIONES
# ============================================================

if "OriginalIndex" in df.columns:

    original_index_test = (

        df.iloc[
            test_idx
        ][
            "OriginalIndex"
        ]

        .to_numpy()

    )

else:

    original_index_test = test_idx


results = pd.DataFrame({

    "OriginalIndex":
        original_index_test,

    "LCA_real":
        y_test[:, 0],

    "LCA_pred":
        y_pred[:, 0],

    "TA_real":
        y_test[:, 1],

    "TA_pred":
        y_pred[:, 1],

    "CT_real":
        y_test[:, 2],

    "CT_pred":
        y_pred[:, 2],
})


results.to_csv(

    OUTPUT_DIR
    / "predicciones_test.csv",

    index=False,
)


# ============================================================
# 22. GUARDAR SPLIT
# ============================================================

np.savez(

    OUTPUT_DIR
    / "split_indices.npz",

    train_idx=train_idx,

    val_idx=val_idx,

    test_idx=test_idx,
)


# ============================================================
# 23. GUARDAR MODELO
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

        "seed":
            SEED,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

    },

    OUTPUT_DIR
    / "modelo_mlp_supervisada_v2.pt",
)


# ============================================================
# 24. CURVA DE LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history_train,

    label="Train",
)


plt.plot(

    history_val,

    label="Validation",
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "MSE Loss"
)


plt.title(
    "Curva de entrenamiento"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "loss_train_validation.png",

    dpi=300,
)


plt.close()


# ============================================================
# 25. GRAFICOS REAL VS PREDICHO
# ============================================================

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


    plt.figure(
        figsize=(6, 6)
    )


    plt.scatter(

        y_true_i,

        y_pred_i,

        alpha=0.8,
    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",
    )


    plt.xlim(
        0,
        1,
    )


    plt.ylim(
        0,
        1,
    )


    plt.xlabel(
        f"{target} real"
    )


    plt.ylabel(
        f"{target} predicho"
    )


    plt.title(
        f"{target}: real vs predicho"
    )


    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR
        / f"{target}_real_vs_pred.png",

        dpi=300,
    )


    plt.close()


# ============================================================
# 26. GUARDAR HISTORIAL DE LOSS
# ============================================================

history_df = pd.DataFrame({

    "epoch":
        np.arange(
            1,
            len(history_train) + 1,
        ),

    "train_loss":
        history_train,

    "val_loss":
        history_val,
})


history_df.to_csv(

    OUTPUT_DIR
    / "historial_loss.csv",

    index=False,
)

# ============================================================
# MOSTRAR PREDICCIONES DEL TEST EN TERMINAL
# ============================================================

print()
print("=" * 90)
print("PREDICCIONES DEL TEST")
print("=" * 90)

for i in range(len(test_idx)):

    original_idx = (
        df.iloc[test_idx[i]]["OriginalIndex"]
        if "OriginalIndex" in df.columns
        else test_idx[i]
    )

    print(
        f"Index {int(original_idx):3d} | "
        f"LCA: real={y_test[i,0]:.4f} pred={y_pred[i,0]:.4f} | "
        f"TA: real={y_test[i,1]:.4f} pred={y_pred[i,1]:.4f} | "
        f"CT: real={y_test[i,2]:.4f} pred={y_pred[i,2]:.4f}"
    )

# ============================================================
# 27. FIN
# ============================================================

print()

print("=" * 60)

print("FIN")

print("=" * 60)


print()

print(
    "Resultados guardados en:"
)

print(
    OUTPUT_DIR
)