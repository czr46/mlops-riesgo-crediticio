"""
ft_engineering.py

Componente de ingeniería de características del pipeline de modelamiento de
riesgo crediticio (Avance 2 - Proyecto Integrador).

Toma el dataset crudo (`Base_de_datos.csv`), aplica las reglas de limpieza y
transformación identificadas en `comprension_eda.ipynb`, construye variables
derivadas, arma un pipeline de preprocesamiento (feature-engine + sklearn) y
retorna los conjuntos de datos de entrenamiento y evaluación listos para
alimentar a `model_training_evaluation.py`.

Uso como módulo:
    from ft_engineering import prepare_datasets
    X_train, X_test, y_train, y_test, pipeline, feature_lists = prepare_datasets()

Uso como script:
    python ft_engineering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from feature_engine.encoding import OneHotEncoder as FEOneHotEncoder
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "Pago_atiempo"

DATA_PATH = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"

# Variable con alto riesgo de fuga de información detectada en el EDA
# (correlación ~0.92 con la variable objetivo). Se excluye por defecto.
LEAKY_FEATURE = "puntaje"

VALID_TENDENCIA = {"Creciente", "Estable", "Decreciente"}

# Variables numéricas con outliers/errores de captura extremos (EDA, sección 3)
WINSORIZE_VARS = ["salario_cliente", "total_otros_prestamos"]

# Variables numéricas con nulos que requieren imputación (EDA, sección 1.3)
NUMERIC_NULL_VARS = [
    "puntaje_datacredito",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "promedio_ingresos_datacredito",
]


# --------------------------------------------------------------------------- #
# 1. Carga
# --------------------------------------------------------------------------- #
def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset crudo de créditos."""
    df = pd.read_csv(path, parse_dates=["fecha_prestamo"])
    return df


# --------------------------------------------------------------------------- #
# 2. Limpieza (reglas deterministas, no dependen de estadísticas del split)
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza / unificación de nulos encontradas en el EDA.

    - Unifica como NaN los valores no categóricos de `tendencia_ingresos`.
    - Acota `edad_cliente` a un rango plausible [18, 90] (regla de negocio).
    - Corrige tipos de datos.
    """
    df = df.copy()

    df.loc[~df["tendencia_ingresos"].isin(VALID_TENDENCIA), "tendencia_ingresos"] = np.nan

    df["edad_cliente"] = df["edad_cliente"].clip(lower=18, upper=90)

    df["tipo_credito"] = df["tipo_credito"].astype(str)
    df["tipo_laboral"] = df["tipo_laboral"].astype(str)
    df["tendencia_ingresos"] = df["tendencia_ingresos"].astype(object)

    return df


# --------------------------------------------------------------------------- #
# 3. Ingeniería de características (row-wise, deterministas)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera atributos derivados propuestos en el EDA (sección 4)."""
    df = df.copy()

    # Relaciones de endeudamiento respecto al ingreso declarado
    df["ratio_cuota_salario"] = df["cuota_pactada"] / (df["salario_cliente"] + 1)
    df["ratio_deuda_salario"] = df["total_otros_prestamos"] / (df["salario_cliente"] + 1)

    # Antigüedad del crédito respecto a la fecha más reciente del dataset (snapshot)
    snapshot_date = df["fecha_prestamo"].max() + pd.Timedelta(days=1)
    df["antiguedad_meses"] = (snapshot_date - df["fecha_prestamo"]).dt.days / 30

    # Bandera de mora
    df["flag_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)

    # Transformación logarítmica de variables monetarias muy asimétricas
    for col in ["capital_prestado", "salario_cliente", "total_otros_prestamos", "cuota_pactada"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # Bucket de edad (variable categórica derivada)
    df["edad_bucket"] = pd.cut(
        df["edad_cliente"],
        bins=[17, 25, 35, 45, 55, 65, 90],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66-90"],
    ).astype(object)

    # Los ratios pueden producir inf si hay divisiones por valores extremos; se normalizan a NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# --------------------------------------------------------------------------- #
# 4. Definición de listas de variables
# --------------------------------------------------------------------------- #
def get_feature_lists(include_leaky: bool = False) -> dict:
    """Retorna las listas de variables numéricas y categóricas a usar en el modelo.

    Parameters
    ----------
    include_leaky : bool
        Si es True, incluye `puntaje` en el set de variables numéricas.
        Por defecto se excluye por el riesgo de fuga de información detectado
        en el EDA (correlación ~0.92 con la variable objetivo) — ver README.
    """
    numeric_features = [
        "plazo_meses",
        "edad_cliente",
        "puntaje_datacredito",
        "cant_creditosvigentes",
        "huella_consulta",
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        "promedio_ingresos_datacredito",
        "ratio_cuota_salario",
        "ratio_deuda_salario",
        "antiguedad_meses",
        "flag_mora",
        "log_capital_prestado",
        "log_salario_cliente",
        "log_total_otros_prestamos",
        "log_cuota_pactada",
    ]
    categorical_features = ["tipo_credito", "tipo_laboral", "tendencia_ingresos", "edad_bucket"]

    if include_leaky:
        numeric_features = [LEAKY_FEATURE] + numeric_features

    return {"numeric": numeric_features, "categorical": categorical_features}


# --------------------------------------------------------------------------- #
# 5. Pipeline de preprocesamiento (pasos que SÍ deben ajustarse solo con train)
# --------------------------------------------------------------------------- #
def build_preprocessing_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Arma el pipeline de preprocesamiento con feature-engine + sklearn.

    Pasos: winsorización de outliers -> agrupación de categorías raras ->
    indicadores + imputación de nulos -> codificación one-hot -> escalado.
    Todos los pasos son transformadores compatibles con scikit-learn, por lo
    que se ajustan únicamente sobre el conjunto de entrenamiento.
    """
    numeric_null_vars = [v for v in NUMERIC_NULL_VARS if v in numeric_features]
    winsorize_vars = [v for v in WINSORIZE_VARS if v in numeric_features]

    steps = []

    if winsorize_vars:
        steps.append((
            "winsorizer",
            Winsorizer(capping_method="quantiles", tail="both", fold=0.01, variables=winsorize_vars),
        ))

    steps.append((
        "rare_label_encoder",
        RareLabelEncoder(tol=0.01, n_categories=2, variables=["tipo_credito"], replace_with="Otros"),
    ))

    steps.append((
        "categorical_imputer_sin_info",
        CategoricalImputer(imputation_method="missing", fill_value="Sin_informacion", variables=["tendencia_ingresos"]),
    ))

    if numeric_null_vars:
        steps.append((
            "missing_indicator",
            AddMissingIndicator(variables=numeric_null_vars),
        ))
        steps.append((
            "numeric_imputer",
            MeanMedianImputer(imputation_method="median", variables=numeric_null_vars),
        ))

    steps.append((
        "one_hot_encoder",
        FEOneHotEncoder(variables=categorical_features, drop_last=True),
    ))

    steps.append((
        "scaler",
        SklearnTransformerWrapper(StandardScaler(), variables=numeric_features),
    ))

    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# 6. Split
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.2):
    X = df[feature_cols]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# 7. Orquestador
# --------------------------------------------------------------------------- #
def prepare_datasets(data_path: Path | str = DATA_PATH, include_leaky: bool = False, test_size: float = 0.2):
    """Ejecuta el flujo completo: carga -> limpieza -> features -> split -> pipeline.

    Retorna X_train, X_test, y_train, y_test, preprocessing_pipeline (sin ajustar)
    y el diccionario de listas de variables usado.
    """
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)

    feature_lists = get_feature_lists(include_leaky=include_leaky)
    feature_cols = feature_lists["numeric"] + feature_lists["categorical"]

    X_train, X_test, y_train, y_test = split_data(df, feature_cols, test_size=test_size)
    pipeline = build_preprocessing_pipeline(feature_lists["numeric"], feature_lists["categorical"])

    return X_train, X_test, y_train, y_test, pipeline, feature_lists


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pipeline, feature_lists = prepare_datasets()
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Variables numéricas ({len(feature_lists['numeric'])}): {feature_lists['numeric']}")
    print(f"Variables categóricas ({len(feature_lists['categorical'])}): {feature_lists['categorical']}")

    X_train_t = pipeline.fit_transform(X_train, y_train)
    print(f"Dataset transformado: {X_train_t.shape[1]} columnas finales")

    out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(out_dir / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(out_dir / "test.csv", index=False)
    print(f"Conjuntos de entrenamiento/evaluación guardados en {out_dir}")
