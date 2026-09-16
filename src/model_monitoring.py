"""
model_monitoring.py

Trabajo de monitoreo del modelo en producción (Avance 3 - Proyecto Integrador).

Compara periódicamente la distribución de los datos "actuales" (nuevas
solicitudes de crédito) contra la distribución de referencia (los datos con
los que se entrenó el modelo) para detectar *data drift*: cambios en la
población que puedan degradar el desempeño del modelo entrenado en
`model_training_evaluation.py`.

Nota importante: como este es un ejercicio académico y no contamos con un
flujo real de datos nuevos llegando en producción, este script **simula**
varios períodos de datos "actuales" a partir del conjunto de evaluación,
introduciendo perturbaciones crecientes de forma controlada
(`simulate_periods`). En un entorno productivo, esos períodos vendrían de
consultas reales al DWH/Datalake con una periodicidad definida (diaria,
semanal, etc.) — la lógica de cálculo de métricas de drift y generación de
alertas es la misma en ambos casos.

Métricas calculadas:
- Kolmogorov-Smirnov (KS test) — variables numéricas
- Population Stability Index (PSI) — variables numéricas
- Jensen-Shannon divergence — variables numéricas
- Chi-cuadrado — variables categóricas

Uso:
    python model_monitoring.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import RANDOM_STATE, prepare_datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIFT_DIR = REPO_ROOT / "reports" / "drift"

PSI_BUCKETS = 10

# Umbrales de alerta (convención estándar de la industria para PSI;
# p-value estándar de 0.05 para KS y Chi-cuadrado)
PSI_THRESHOLDS = {"moderado": 0.10, "severo": 0.25}
P_VALUE_THRESHOLD = 0.05


# --------------------------------------------------------------------------- #
# 1. Métricas de data drift
# --------------------------------------------------------------------------- #
def compute_psi(reference: pd.Series, current: pd.Series, buckets: int = PSI_BUCKETS) -> float:
    """Population Stability Index entre dos muestras numéricas."""
    ref = reference.dropna()
    cur = current.dropna()
    breakpoints = np.unique(ref.quantile(np.linspace(0, 1, buckets + 1)).values)
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(ref_counts.sum(), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(cur_counts.sum(), 1))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_ks(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Test de Kolmogorov-Smirnov de dos muestras. Retorna (estadístico, p-value)."""
    stat, p_value = ks_2samp(reference.dropna(), current.dropna())
    return float(stat), float(p_value)


def compute_jensen_shannon(reference: pd.Series, current: pd.Series, buckets: int = PSI_BUCKETS) -> float:
    """Divergencia de Jensen-Shannon entre los histogramas de dos muestras numéricas."""
    ref = reference.dropna()
    cur = current.dropna()
    breakpoints = np.unique(ref.quantile(np.linspace(0, 1, buckets + 1)).values)
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    return float(jensenshannon(ref_pct, cur_pct, base=2))


def compute_chi_square(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Test de Chi-cuadrado de independencia entre dos muestras categóricas.
    Retorna (estadístico, p-value)."""
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))

    table = pd.DataFrame(
        {
            "reference": [ref_counts.get(c, 0) for c in categories],
            "current": [cur_counts.get(c, 0) for c in categories],
        },
        index=categories,
    ).T
    table = table.loc[:, table.sum(axis=0) > 0]

    if table.shape[1] < 2:
        return 0.0, 1.0

    stat, p_value, _, _ = chi2_contingency(table)
    return float(stat), float(p_value)


def classify_psi(psi_value: float) -> str:
    if psi_value >= PSI_THRESHOLDS["severo"]:
        return "severo"
    if psi_value >= PSI_THRESHOLDS["moderado"]:
        return "moderado"
    return "sin_drift"


def classify_p_value(p_value: float) -> str:
    return "drift_detectado" if p_value < P_VALUE_THRESHOLD else "sin_drift"


# --------------------------------------------------------------------------- #
# 2. Simulación de muestreo periódico (ver nota al inicio del módulo)
# --------------------------------------------------------------------------- #
def simulate_periods(X_test: pd.DataFrame, n_periods: int = 5, seed: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    """Simula lotes 'actuales' con drift creciente a partir del set de evaluación.

    P0 no tiene perturbación (sirve como control: debería marcar 'sin_drift'
    en la mayoría de variables). P1..Pn introducen corrimientos progresivos en
    el ingreso declarado, la relación deuda/salario, la edad y la proporción
    de tipo_laboral — simulando, por ejemplo, la entrada de una campaña de
    crédito dirigida a un segmento distinto de clientes.
    """
    rng = np.random.default_rng(seed)
    periods = {"P0_sin_drift": X_test.copy()}

    for i in range(1, n_periods + 1):
        shift = i / n_periods
        batch = X_test.copy()

        # Corrimiento creciente en variables monetarias (ingreso declarado sube)
        noise = rng.normal(loc=1 + 0.35 * shift, scale=0.05, size=len(batch))
        batch["log_salario_cliente"] = batch["log_salario_cliente"] * noise

        # La relación deuda/salario se reduce (clientes "aparentan" menor riesgo)
        batch["ratio_deuda_salario"] = batch["ratio_deuda_salario"] * (1 - 0.3 * shift)

        # La población se vuelve más joven
        batch["edad_cliente"] = (batch["edad_cliente"] - 8 * shift).clip(lower=18)

        # Más clientes independientes (cambio en la mezcla de canal/segmento)
        flip_mask = rng.random(len(batch)) < (0.25 * shift)
        batch.loc[flip_mask, "tipo_laboral"] = "Independiente"

        periods[f"P{i}_shift_{shift:.1f}"] = batch

    return periods


# --------------------------------------------------------------------------- #
# 3. Reporte de drift
# --------------------------------------------------------------------------- #
def compute_drift_report(
    reference: pd.DataFrame, periods: dict[str, pd.DataFrame], numeric_features: list[str], categorical_features: list[str]
) -> pd.DataFrame:
    rows = []
    for period_name, current in periods.items():
        for col in numeric_features:
            psi = compute_psi(reference[col], current[col])
            ks_stat, ks_p = compute_ks(reference[col], current[col])
            js = compute_jensen_shannon(reference[col], current[col])
            rows.append({"periodo": period_name, "variable": col, "tipo": "numerica", "metrica": "PSI", "valor": psi, "alerta": classify_psi(psi)})
            rows.append({"periodo": period_name, "variable": col, "tipo": "numerica", "metrica": "KS_statistic", "valor": ks_stat, "alerta": classify_p_value(ks_p)})
            rows.append({"periodo": period_name, "variable": col, "tipo": "numerica", "metrica": "Jensen_Shannon", "valor": js, "alerta": classify_psi(js)})

        for col in categorical_features:
            chi2_stat, chi2_p = compute_chi_square(reference[col], current[col])
            rows.append({"periodo": period_name, "variable": col, "tipo": "categorica", "metrica": "Chi2_statistic", "valor": chi2_stat, "alerta": classify_p_value(chi2_p)})

    return pd.DataFrame(rows)


def generate_alerts(drift_report: pd.DataFrame) -> list[dict]:
    """Genera mensajes de alerta para variables con drift moderado/severo o
    detectado, con una recomendación accionable."""
    alerts = []
    flagged = drift_report[drift_report["alerta"].isin(["severo", "drift_detectado"])]

    for _, row in flagged.iterrows():
        if row["alerta"] == "severo":
            mensaje = (
                f"[CRÍTICO] Drift severo en '{row['variable']}' durante {row['periodo']} "
                f"({row['metrica']}={row['valor']:.3f}). Revisar la variable y considerar "
                f"reentrenamiento del modelo."
            )
        else:
            mensaje = (
                f"[ALERTA] Se detectó drift estadísticamente significativo en '{row['variable']}' "
                f"durante {row['periodo']} ({row['metrica']}={row['valor']:.3f})."
            )
        alerts.append({
            "periodo": row["periodo"],
            "variable": row["variable"],
            "metrica": row["metrica"],
            "valor": row["valor"],
            "nivel": row["alerta"],
            "mensaje": mensaje,
        })

    return alerts


# --------------------------------------------------------------------------- #
# 4. Visualizaciones
# --------------------------------------------------------------------------- #
def plot_psi_evolution(drift_report: pd.DataFrame, out_path: Path) -> None:
    psi_df = drift_report[drift_report["metrica"] == "PSI"].copy()
    pivot = psi_df.pivot(index="periodo", columns="variable", values="valor")

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(ax=ax, marker="o")
    ax.axhline(PSI_THRESHOLDS["moderado"], color="orange", linestyle="--", label="umbral moderado (0.10)")
    ax.axhline(PSI_THRESHOLDS["severo"], color="red", linestyle="--", label="umbral severo (0.25)")
    ax.set_title("Evolución del PSI por variable y período")
    ax.set_ylabel("PSI")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_psi_heatmap(drift_report: pd.DataFrame, out_path: Path) -> None:
    psi_df = drift_report[drift_report["metrica"] == "PSI"].copy()
    pivot = psi_df.pivot(index="variable", columns="periodo", values="valor")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "PSI"})
    ax.set_title("Mapa de calor de PSI (variable x período)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_distribution_comparison(reference: pd.Series, current: pd.Series, variable: str, period_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(reference.dropna(), label="Histórico (referencia)", fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(current.dropna(), label=f"Actual ({period_name})", fill=True, alpha=0.3, ax=ax)
    ax.set_title(f"Distribución histórica vs. actual — {variable}")
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_monitoring(n_periods: int = 5) -> pd.DataFrame:
    sns.set_theme(style="whitegrid")

    X_train, X_test, y_train, y_test, _, feature_lists = prepare_datasets(include_leaky=False)
    reference = X_train
    periods = simulate_periods(X_test, n_periods=n_periods)

    drift_report = compute_drift_report(reference, periods, feature_lists["numeric"], feature_lists["categorical"])

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    drift_report.to_csv(DRIFT_DIR / "drift_metrics.csv", index=False)

    alerts = generate_alerts(drift_report)
    with open(DRIFT_DIR / "alerts.json", "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)

    plot_psi_evolution(drift_report, DRIFT_DIR / "psi_evolution.png")
    plot_psi_heatmap(drift_report, DRIFT_DIR / "psi_heatmap.png")

    last_period_name = list(periods.keys())[-1]
    plot_distribution_comparison(
        reference["log_salario_cliente"], periods[last_period_name]["log_salario_cliente"],
        "log_salario_cliente", last_period_name, DRIFT_DIR / "distribucion_log_salario_cliente.png",
    )
    plot_distribution_comparison(
        reference["edad_cliente"], periods[last_period_name]["edad_cliente"],
        "edad_cliente", last_period_name, DRIFT_DIR / "distribucion_edad_cliente.png",
    )

    print(f"Períodos simulados: {list(periods.keys())}")
    print(f"Alertas generadas: {len(alerts)}")
    for a in alerts[:10]:
        print(" -", a["mensaje"])
    print(f"\nReporte completo guardado en {DRIFT_DIR}")

    return drift_report


if __name__ == "__main__":
    run_monitoring()
