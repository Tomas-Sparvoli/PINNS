# ============================================================
# COMPARACION DE CONSISTENCIA FISICA
#
# Compara:
#
#   1. MLP supervisada baseline
#   2. MLP fisicamente informada con restricciones de area
#
# usando los MISMOS:
#
#   - casos de test
#   - surrogates de Ag_mean y Ag_max
#   - limites empiricos
#
# Restricciones:
#
#   0.01 <= Ag_mean <= 0.09 cm^2
#   Ag_max <= 0.27 cm^2
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# 1. RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = Path(
    r"C:\Users\sparv\OneDrive\Escritorio\PhonationModel"
) / "dataset_features_200_final.csv"


BASELINE_MODEL_PATH = (
    BASE_DIR
    / "resultados_mlp_200_v2"
    / "modelo_mlp_supervisada_v2.pt"
)


PHYSICAL_MODEL_PATH = (
    BASE_DIR
    / "resultados_mlp_fisica_areas_200"
    / "modelo_mlp_fisica_areas.pt"
)


PHYSICAL_RESULTS_PATH = (
    BASE_DIR
    / "resultados_mlp_fisica_areas_200"
    / "predicciones_test.csv"
)


OUTPUT_PATH = (
    BASE_DIR
    / "comparacion_consistencia_fisica.csv"
)


# ============================================================
# 2. CONFIGURACION
# ============================================================

AG_MEAN_MIN = 0.01

AG_MEAN_MAX = 0.09

AG_MAX_LIMIT = 0.27


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()

print("=" * 80)
print("COMPARACION DE CONSISTENCIA FISICA")
print("=" * 80)

print(
    f"Dispositivo: {device}"
)


# ============================================================
# 3. CARGAR DATASET
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
)


Ag_max_all = df[
    "Ag_max_cm2"
].to_numpy(
    dtype=np.float32
)


# ============================================================
# 4. CARGAR CHECKPOINTS
# ============================================================

baseline_checkpoint = torch.load(

    BASELINE_MODEL_PATH,

    map_location=device,

    weights_only=False,
)


physical_checkpoint = torch.load(

    PHYSICAL_MODEL_PATH,

    map_location=device,

    weights_only=False,
)


train_idx = baseline_checkpoint[
    "train_idx"
]

test_idx = baseline_checkpoint[
    "test_idx"
]


# ============================================================
# 5. VERIFICAR QUE EL SPLIT SEA EL MISMO
# ============================================================

physical_test_idx = physical_checkpoint[
    "test_idx"
]


if not np.array_equal(
    test_idx,
    physical_test_idx,
):

    raise ValueError(
        "El baseline y el modelo fisico no usan el mismo test."
    )


print()

print(
    f"Muestras de test: {len(test_idx)}"
)


# ============================================================
# 6. NORMALIZACION DE FEATURES DEL BASELINE
# ============================================================

scaler_mean = np.asarray(
    baseline_checkpoint[
        "scaler_mean"
    ],
    dtype=np.float32,
)


scaler_scale = np.asarray(
    baseline_checkpoint[
        "scaler_scale"
    ],
    dtype=np.float32,
)


X_test = X[
    test_idx
]


X_test_scaled = (

    X_test
    - scaler_mean

) / scaler_scale


X_test_tensor = torch.tensor(

    X_test_scaled,

    dtype=torch.float32,

    device=device,
)


# ============================================================
# 7. RED PRINCIPAL
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
# 8. SURROGATE DE AREA
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
# 9. CARGAR BASELINE
# ============================================================

baseline_model = MLP().to(
    device
)


baseline_model.load_state_dict(

    baseline_checkpoint[
        "model_state_dict"
    ]

)


baseline_model.eval()


# ============================================================
# 10. CARGAR SURROGATES
# ============================================================

mean_model = AreaSurrogate().to(
    device
)


max_model = AreaSurrogate().to(
    device
)


mean_model.load_state_dict(

    physical_checkpoint[
        "mean_surrogate_state_dict"
    ]

)


max_model.load_state_dict(

    physical_checkpoint[
        "max_surrogate_state_dict"
    ]

)


mean_model.eval()

max_model.eval()


# ============================================================
# 11. RECALCULAR NORMALIZACION DE AREAS
#
# Es la misma utilizada para entrenar los surrogates:
# exclusivamente a partir de TRAIN.
# ============================================================

Ag_mean_train = Ag_mean_all[
    train_idx
]


Ag_max_train = Ag_max_all[
    train_idx
]


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


print()

print("=" * 80)
print("NORMALIZACION DE AREAS")
print("=" * 80)

print(
    f"Ag_mean: media={Ag_mean_mu:.6f} | "
    f"std={Ag_mean_sigma:.6f}"
)

print(
    f"Ag_max:  media={Ag_max_mu:.6f} | "
    f"std={Ag_max_sigma:.6f}"
)


# ============================================================
# 12. PREDICCIONES DEL BASELINE
# ============================================================

with torch.no_grad():


    baseline_activation_tensor = baseline_model(
        X_test_tensor
    )


baseline_activations = (

    baseline_activation_tensor
    .cpu()
    .numpy()

)


# ============================================================
# 13. PASAR LAS PREDICCIONES BASELINE POR LOS SURROGATES
# ============================================================

with torch.no_grad():


    baseline_Ag_mean_scaled = mean_model(
        baseline_activation_tensor
    )


    baseline_Ag_max_scaled = max_model(
        baseline_activation_tensor
    )


baseline_Ag_mean = (

    baseline_Ag_mean_scaled
    .cpu()
    .numpy()
    .reshape(-1)

    * Ag_mean_sigma

    + Ag_mean_mu
)


baseline_Ag_max = (

    baseline_Ag_max_scaled
    .cpu()
    .numpy()
    .reshape(-1)

    * Ag_max_sigma

    + Ag_max_mu
)


# ============================================================
# 14. VIOLACIONES DEL BASELINE
# ============================================================

baseline_mean_lower = (

    baseline_Ag_mean
    < AG_MEAN_MIN

)


baseline_mean_upper = (

    baseline_Ag_mean
    > AG_MEAN_MAX

)


baseline_mean_violation = (

    baseline_mean_lower

    |

    baseline_mean_upper

)


baseline_max_violation = (

    baseline_Ag_max
    > AG_MAX_LIMIT

)


N_baseline_mean = int(
    np.sum(
        baseline_mean_violation
    )
)


N_baseline_max = int(
    np.sum(
        baseline_max_violation
    )
)


# ============================================================
# 15. CARGAR RESULTADOS DEL MODELO FISICO
# ============================================================

physical_results = pd.read_csv(
    PHYSICAL_RESULTS_PATH
)


physical_Ag_mean = physical_results[
    "Ag_mean_pred_cm2"
].to_numpy()


physical_Ag_max = physical_results[
    "Ag_max_pred_cm2"
].to_numpy()


physical_mean_violation = (

    (physical_Ag_mean < AG_MEAN_MIN)

    |

    (physical_Ag_mean > AG_MEAN_MAX)

)


physical_max_violation = (

    physical_Ag_max
    > AG_MAX_LIMIT

)


N_physical_mean = int(
    np.sum(
        physical_mean_violation
    )
)


N_physical_max = int(
    np.sum(
        physical_max_violation
    )
)


# ============================================================
# 16. RESULTADO GLOBAL
# ============================================================

print()

print("=" * 80)
print("RESULTADOS DE CONSISTENCIA FISICA")
print("=" * 80)


print()

print("BASELINE SUPERVISADO")

print("-" * 80)

print(
    f"Violaciones Ag_mean: "
    f"{N_baseline_mean} / {len(test_idx)}"
)

print(
    f"Violaciones Ag_max:  "
    f"{N_baseline_max} / {len(test_idx)}"
)


print()

print(
    f"Ag_mean predicho:"
)

print(
    f"  min   = {np.min(baseline_Ag_mean):.6f}"
)

print(
    f"  medio = {np.mean(baseline_Ag_mean):.6f}"
)

print(
    f"  max   = {np.max(baseline_Ag_mean):.6f}"
)


print()

print(
    f"Ag_max predicho:"
)

print(
    f"  min   = {np.min(baseline_Ag_max):.6f}"
)

print(
    f"  medio = {np.mean(baseline_Ag_max):.6f}"
)

print(
    f"  max   = {np.max(baseline_Ag_max):.6f}"
)


print()

print("MODELO FISICAMENTE INFORMADO")

print("-" * 80)

print(
    f"Violaciones Ag_mean: "
    f"{N_physical_mean} / {len(test_idx)}"
)

print(
    f"Violaciones Ag_max:  "
    f"{N_physical_max} / {len(test_idx)}"
)


print()

print(
    f"Ag_mean predicho:"
)

print(
    f"  min   = {np.min(physical_Ag_mean):.6f}"
)

print(
    f"  medio = {np.mean(physical_Ag_mean):.6f}"
)

print(
    f"  max   = {np.max(physical_Ag_mean):.6f}"
)


print()

print(
    f"Ag_max predicho:"
)

print(
    f"  min   = {np.min(physical_Ag_max):.6f}"
)

print(
    f"  medio = {np.mean(physical_Ag_max):.6f}"
)

print(
    f"  max   = {np.max(physical_Ag_max):.6f}"
)


# ============================================================
# 17. COMPARACION CASO POR CASO
# ============================================================

print()

print("=" * 120)
print("COMPARACION CASO POR CASO")
print("=" * 120)


rows = []


for i in range(
    len(test_idx)
):


    if "OriginalIndex" in df.columns:

        original_idx = int(

            df.iloc[
                test_idx[i]
            ][
                "OriginalIndex"
            ]

        )

    else:

        original_idx = int(
            test_idx[i]
        )


    print(

        f"Index {original_idx:3d} | "

        f"Baseline: "
        f"Mean={baseline_Ag_mean[i]:.4f} "
        f"{'*' if baseline_mean_violation[i] else ' '} | "
        f"Max={baseline_Ag_max[i]:.4f} "
        f"{'*' if baseline_max_violation[i] else ' '} || "

        f"Fisico: "
        f"Mean={physical_Ag_mean[i]:.4f} "
        f"{'*' if physical_mean_violation[i] else ' '} | "
        f"Max={physical_Ag_max[i]:.4f} "
        f"{'*' if physical_max_violation[i] else ' '}"

    )


    rows.append({

        "OriginalIndex":
            original_idx,

        "Baseline_Ag_mean":
            baseline_Ag_mean[i],

        "Baseline_Ag_mean_violation":
            baseline_mean_violation[i],

        "Baseline_Ag_max":
            baseline_Ag_max[i],

        "Baseline_Ag_max_violation":
            baseline_max_violation[i],

        "Fisico_Ag_mean":
            physical_Ag_mean[i],

        "Fisico_Ag_mean_violation":
            physical_mean_violation[i],

        "Fisico_Ag_max":
            physical_Ag_max[i],

        "Fisico_Ag_max_violation":
            physical_max_violation[i],
    })


# ============================================================
# 18. GUARDAR COMPARACION
# ============================================================

comparison_df = pd.DataFrame(
    rows
)


comparison_df.to_csv(

    OUTPUT_PATH,

    index=False,
)


# ============================================================
# 19. RESUMEN
# ============================================================

print()

print("=" * 80)
print("RESUMEN")
print("=" * 80)


print(

    f"Ag_mean: "

    f"Baseline {N_baseline_mean} violaciones -> "

    f"Fisico {N_physical_mean} violaciones"

)


print(

    f"Ag_max:  "

    f"Baseline {N_baseline_max} violaciones -> "

    f"Fisico {N_physical_max} violaciones"

)


print()

print(
    "El simbolo * indica una violacion del rango empirico."
)


print()

print(
    f"Resultados guardados en:"
)

print(
    OUTPUT_PATH
)