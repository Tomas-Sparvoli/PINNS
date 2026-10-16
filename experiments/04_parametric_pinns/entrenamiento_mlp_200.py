# ============================================================
# ENTRENAMIENTO MLP - DATASET SINTETICO DE FONACION
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
#   Entrenar una red MLP supervisada como baseline.
#
# Posteriormente se comparara contra la misma arquitectura
# incorporando terminos de perdida basados en fisica.
# ============================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURACION
# ============================================================

SEED = 42

CSV_PATH = r"C:\Users\sparv\OneDrive\Escritorio\PhonationModel\dataset_features_200_final.csv"

TEST_SIZE = 0.20

BATCH_SIZE = 16

LEARNING_RATE = 1e-3

EPOCHS = 1000

PATIENCE = 100


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

print(f"Dispositivo: {device}")


# ============================================================
# 4. CARGAR DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print()
print("=" * 60)
print("DATASET")
print("=" * 60)

print(f"Muestras totales: {len(df)}")

print()
print("Columnas disponibles:")
print(df.columns.tolist())


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


# Verificacion de columnas

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
        f"Faltan features en el CSV: {missing_features}"
    )

if missing_targets:
    raise ValueError(
        f"Faltan targets en el CSV: {missing_targets}"
    )


# ============================================================
# 6. EXTRAER X E Y
# ============================================================

X = df[feature_columns].to_numpy(
    dtype=np.float32
)

y = df[target_columns].to_numpy(
    dtype=np.float32
)


# ============================================================
# 7. TRAIN / TEST
# ============================================================

indices = np.arange(
    len(df)
)

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True,
)


X_train = X[train_idx]

X_test = X[test_idx]

y_train = y[train_idx]

y_test = y[test_idx]


print()
print("=" * 60)
print("DIVISION DEL DATASET")
print("=" * 60)

print(
    f"Train: {len(train_idx)}"
)

print(
    f"Test:  {len(test_idx)}"
)


# ============================================================
# 8. NORMALIZACION DE ENTRADAS
# ============================================================

scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(
    X_train
)

X_test_scaled = scaler_X.transform(
    X_test
)


# ============================================================
# 9. CONVERSION A TENSORES
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train,
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
# 10. DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)


generator = torch.Generator()

generator.manual_seed(SEED)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)


# ============================================================
# 11. MODELO
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(7, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 3),

            nn.Sigmoid(),
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 12. INICIALIZACION DEL MODELO
# ============================================================

torch.manual_seed(SEED)

model = MLP().to(device)


# ============================================================
# 13. LOSS Y OPTIMIZADOR
# ============================================================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 14. ENTRENAMIENTO
# ============================================================

best_loss = np.inf

best_state = None

epochs_without_improvement = 0


history_train = []


print()
print("=" * 60)
print("ENTRENAMIENTO")
print("=" * 60)


for epoch in range(1, EPOCHS + 1):

    model.train()

    epoch_loss = 0.0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)


        optimizer.zero_grad()


        predictions = model(
            X_batch
        )


        loss = criterion(
            predictions,
            y_batch
        )


        loss.backward()


        optimizer.step()


        epoch_loss += (
            loss.item()
            *
            X_batch.size(0)
        )


    epoch_loss /= len(
        train_loader.dataset
    )


    history_train.append(
        epoch_loss
    )


    # --------------------------------------------------------
    # Early stopping sobre training loss
    # --------------------------------------------------------

    if epoch_loss < best_loss:

        best_loss = epoch_loss

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
            f"Loss = {epoch_loss:.6f}"
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
# 15. RECUPERAR MEJOR MODELO
# ============================================================

model.load_state_dict(
    best_state
)

model = model.to(device)


# ============================================================
# 16. PREDICCION SOBRE TEST
# ============================================================

model.eval()


with torch.no_grad():

    y_pred_tensor = model(
        X_test_tensor.to(device)
    )


y_pred = (
    y_pred_tensor
    .cpu()
    .numpy()
)


# ============================================================
# 17. METRICAS
# ============================================================

print()
print("=" * 60)
print("RESULTADOS TEST")
print("=" * 60)


for i, target in enumerate(
    target_columns
):

    y_true_i = y_test[:, i]

    y_pred_i = y_pred[:, i]


    mse = mean_squared_error(
        y_true_i,
        y_pred_i
    )


    rmse = np.sqrt(
        mse
    )


    mae = mean_absolute_error(
        y_true_i,
        y_pred_i
    )


    r2 = r2_score(
        y_true_i,
        y_pred_i
    )


    print()
    print(target)

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
# 18. ERROR GLOBAL
# ============================================================

global_mse = mean_squared_error(
    y_test,
    y_pred
)


global_rmse = np.sqrt(
    global_mse
)


global_mae = mean_absolute_error(
    y_test,
    y_pred
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
# 19. GUARDAR PREDICCIONES
# ============================================================

results = pd.DataFrame({

    "OriginalIndex":
        df.iloc[test_idx][
            "OriginalIndex"
        ].values
        if "OriginalIndex" in df.columns
        else test_idx,

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
    "resultados_mlp_supervisada_200.csv",
    index=False,
)


# ============================================================
# 20. GUARDAR MODELO
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

        "test_idx":
            test_idx,

        "scaler_mean":
            scaler_X.mean_,

        "scaler_scale":
            scaler_X.scale_,

        "seed":
            SEED,
    },

    "modelo_mlp_supervisada_200.pt",
)


# ============================================================
# 21. FIN
# ============================================================

print()
print("=" * 60)
print("FIN")
print("=" * 60)

print(
    "Resultados guardados en:"
)

print(
    "resultados_mlp_supervisada_200.csv"
)

print(
    "modelo_mlp_supervisada_200.pt"
)