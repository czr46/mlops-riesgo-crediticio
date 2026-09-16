"""
model_training_evaluation.py

Entrenamiento y evaluación de modelos de aprendizaje automático supervisado
para predecir `Pago_atiempo` (Avance 2 - Proyecto Integrador).

Provee las funciones reutilizables solicitadas (`build_model`,
`summarize_classification`), entrena varios modelos candidatos, los compara
con métricas apropiadas para un problema desbalanceado (precision, recall,
F1, ROC-AUC) y selecciona el de mejor desempeño. Guarda el modelo ganador en
`models/best_model.joblib`, la tabla comparativa en
`reports/model_comparison.csv` y el gráfico comparativo en
`reports/model_comparison.png`.

Uso:
    python model_training_evaluation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ft_engineering import RANDOM_STATE, prepare_datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"


# --------------------------------------------------------------------------- #
# Funciones reutilizables solicitadas en la consigna
# --------------------------------------------------------------------------- #
def build_model(estimator, preprocessing_pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Arma un Pipeline (preprocesamiento + estimador) y lo entrena."""
    model = Pipeline(steps=[*preprocessing_pipeline.steps, ("estimator", estimator)])
    model.fit(X_train, y_train)
    return model


def summarize_classification(model: Pipeline, X_test, y_test, model_name: str) -> dict:
    """Calcula y retorna las métricas de evaluación de un modelo ya entrenado."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "modelo": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


# --------------------------------------------------------------------------- #
# Comparación de modelos
# --------------------------------------------------------------------------- #
def get_candidate_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }


def train_and_compare(preprocessing_pipeline, X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict]:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    candidates = get_candidate_models(scale_pos_weight)

    fitted_models = {}
    rows = []
    for name, estimator in candidates.items():
        model = build_model(estimator, preprocessing_pipeline, X_train, y_train)
        metrics = summarize_classification(model, X_test, y_test, name)
        rows.append(metrics)
        fitted_models[name] = model
        print(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "modelo"))

    results_df = pd.DataFrame(rows).set_index("modelo").sort_values("roc_auc", ascending=False)
    return results_df, fitted_models


def plot_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    results_df[metrics].plot(kind="bar", ax=ax)
    ax.set_title("Comparación de modelos - Pago_atiempo")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_curves(fitted_models: dict, X_test, y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="azar")
    ax.set_title("Curvas ROC - comparación de modelos")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def select_best_model(results_df: pd.DataFrame, fitted_models: dict, metric: str = "roc_auc"):
    best_name = results_df[metric].idxmax()
    return best_name, fitted_models[best_name]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_experiment(include_leaky: bool) -> pd.DataFrame:
    tag = "con_puntaje" if include_leaky else "sin_puntaje"
    print(f"\n=== Entrenamiento {tag} ===")

    X_train, X_test, y_train, y_test, pipeline, feature_lists = prepare_datasets(include_leaky=include_leaky)
    results_df, fitted_models = train_and_compare(pipeline, X_train, X_test, y_train, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / f"model_comparison_{tag}.csv")
    plot_comparison(results_df, REPORTS_DIR / f"model_comparison_{tag}.png")
    plot_roc_curves(fitted_models, X_test, y_test, REPORTS_DIR / f"roc_curves_{tag}.png")

    if not include_leaky:
        best_name, best_model = select_best_model(results_df, fitted_models, metric="roc_auc")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

        metadata = {
            "modelo_seleccionado": best_name,
            "metricas": results_df.loc[best_name].to_dict(),
            "variables_numericas": feature_lists["numeric"],
            "variables_categoricas": feature_lists["categorical"],
            "incluye_puntaje": include_leaky,
            "nota": (
                "Se excluye 'puntaje' del modelo de producción por riesgo de fuga de "
                "informacion (correlacion ~0.92 con la variable objetivo, ver EDA). "
                "Ver reports/model_comparison_con_puntaje.csv para el analisis de "
                "sensibilidad incluyendo esa variable."
            ),
        }
        with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nMejor modelo (sin 'puntaje'): {best_name}")
        print(results_df)
        print(f"\nGuardado en {MODELS_DIR / 'best_model.joblib'}")

    return results_df


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    results_sin_puntaje = run_experiment(include_leaky=False)
    results_con_puntaje = run_experiment(include_leaky=True)

    print("\n=== Resumen: efecto de incluir 'puntaje' (posible leakage) ===")
    comparison = pd.concat(
        [results_sin_puntaje["roc_auc"].rename("roc_auc_sin_puntaje"),
         results_con_puntaje["roc_auc"].rename("roc_auc_con_puntaje")],
        axis=1,
    )
    print(comparison)
