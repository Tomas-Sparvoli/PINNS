import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. RUTAS
# ============================================================

DATA_DIR = Path(
    r"C:\Users\sparv\OneDrive\Escritorio\PhonationModel"
)


FULL_CSV = (
    DATA_DIR
    / "dataset_sintetico_200_agmean40.csv"
)


FINAL_CSV = (
    DATA_DIR
    / "dataset_features_200_final.csv"
)


SPLIT_PATH = Path(
    r"C:\Users\sparv\OneDrive\Escritorio\Facultad\Tesis\ProgramaPINNS\PINNS"
) / "experiments" / "04_parametric_pinns" / \
    "resultados_mlp_200_v2" / "split_indices.npz"


# ============================================================
# 2. CARGAR DATASETS
# ============================================================

df_full = pd.read_csv(
    FULL_CSV
)


df_final = pd.read_csv(
    FINAL_CSV
)


print()
print("=" * 80)
print("DATASETS")
print("=" * 80)

print(
    f"Dataset completo: {len(df_full)}"
)

print(
    f"Dataset final:    {len(df_final)}"
)


# ============================================================
# 3. CREAR OriginalIndex EN DATASET COMPLETO
#
# Las filas del dataset completo corresponden
# a las simulaciones originales 1...200.
# ============================================================

df_full = df_full.copy()


df_full[
    "OriginalIndex"
] = np.arange(
    1,
    len(df_full) + 1
)


# ============================================================
# 4. VERIFICAR OriginalIndex EN DATASET FINAL
# ============================================================

if "OriginalIndex" not in df_final.columns:

    raise KeyError(
        "dataset_features_200_final.csv "
        "no contiene OriginalIndex."
    )


# ============================================================
# 5. HACER MERGE
# ============================================================

df = df_final.merge(

    df_full[
        [
            "OriginalIndex",
            "Ag_mean40_cm2",
            "Ag_threshold40_cm2",
            "Ag_fraction_above40",
        ]
    ],

    on="OriginalIndex",

    how="left",
)


print()
print("=" * 80)
print("MERGE")
print("=" * 80)

print(
    f"Filas despues del merge: "
    f"{len(df)}"
)


missing = df[
    "Ag_mean40_cm2"
].isna()


print(
    f"Ag_mean40 faltantes: "
    f"{missing.sum()}"
)


if missing.any():

    print()

    print(
        "OriginalIndex faltantes:"
    )

    print(
        df.loc[
            missing,
            "OriginalIndex"
        ].to_list()
    )


# ============================================================
# 6. CARGAR MISMO SPLIT
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


# ============================================================
# 7. FUNCION PARA MOSTRAR ESTADISTICAS
# ============================================================

def mostrar_estadisticas(
    nombre,
    indices,
):

    valores = (

        df.iloc[
            indices
        ][
            "Ag_mean40_cm2"
        ]

        .to_numpy(
            dtype=float
        )
    )


    valores = valores[
        np.isfinite(
            valores
        )
    ]


    print()
    print("=" * 80)
    print(nombre)
    print("=" * 80)


    print(
        f"N       = {len(valores)}"
    )

    print(
        f"Min     = {np.min(valores):.6f} cm^2"
    )

    print(
        f"P5      = {np.percentile(valores, 5):.6f} cm^2"
    )

    print(
        f"P10     = {np.percentile(valores, 10):.6f} cm^2"
    )

    print(
        f"Media   = {np.mean(valores):.6f} cm^2"
    )

    print(
        f"Mediana = {np.median(valores):.6f} cm^2"
    )

    print(
        f"P90     = {np.percentile(valores, 90):.6f} cm^2"
    )

    print(
        f"P95     = {np.percentile(valores, 95):.6f} cm^2"
    )

    print(
        f"Max     = {np.max(valores):.6f} cm^2"
    )

    print(
        f"Std     = {np.std(valores):.6f} cm^2"
    )


# ============================================================
# 8. ESTADISTICAS
# ============================================================

mostrar_estadisticas(
    "TRAIN",
    train_idx,
)


mostrar_estadisticas(
    "VALIDATION",
    val_idx,
)


mostrar_estadisticas(
    "TEST",
    test_idx,
)


# ============================================================
# 9. COMPARACION Ag_mean VS Ag_mean40 EN TRAIN
# ============================================================

old_train = (

    df.iloc[
        train_idx
    ][
        "Ag_mean_cm2"
    ]

    .to_numpy(
        dtype=float
    )
)


new_train = (

    df.iloc[
        train_idx
    ][
        "Ag_mean40_cm2"
    ]

    .to_numpy(
        dtype=float
    )
)


valid_compare = (
    np.isfinite(old_train)
    &
    np.isfinite(new_train)
)


print()
print("=" * 80)
print("COMPARACION TRAIN")
print("=" * 80)


print(
    f"Ag_mean original medio = "
    f"{np.mean(old_train[valid_compare]):.6f} cm^2"
)


print(
    f"Ag_mean40 medio        = "
    f"{np.mean(new_train[valid_compare]):.6f} cm^2"
)


print(
    f"Diferencia media       = "
    f"{np.mean(
        new_train[valid_compare]
        - old_train[valid_compare]
    ):.6f} cm^2"
)