"""
streamlit_app.py

Aplicación en Streamlit para visualizar el monitoreo de data drift del modelo
de riesgo crediticio (Avance 3 - Proyecto Integrador). Consume la lógica de
`model_monitoring.py` (cálculo de métricas de drift sobre períodos simulados)
y la presenta de forma interactiva: comparación de distribuciones histórica
vs. actual, tabla de métricas por variable, evolución temporal del drift, e
indicadores de alerta con recomendaciones.

Uso:
    streamlit run streamlit_app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from ft_engineering import prepare_datasets
from model_monitoring import (
    P_VALUE_THRESHOLD,
    PSI_THRESHOLDS,
    compute_drift_report,
    generate_alerts,
    simulate_periods,
)

st.set_page_config(page_title="Monitoreo de Data Drift - Riesgo Crediticio", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner="Cargando datos y calculando métricas de drift...")
def load_data():
    X_train, X_test, _, _, _, feature_lists = prepare_datasets(include_leaky=False)
    periods = simulate_periods(X_test, n_periods=5)
    drift_report = compute_drift_report(X_train, periods, feature_lists["numeric"], feature_lists["categorical"])
    alerts = generate_alerts(drift_report)
    return X_train, periods, drift_report, alerts, feature_lists


reference, periods, drift_report, alerts, feature_lists = load_data()
period_names = list(periods.keys())

st.title("📊 Monitoreo de Data Drift — Modelo de Riesgo Crediticio")
st.caption(
    "Los períodos P1-P5 se simulan con drift creciente a partir del conjunto de evaluación, "
    "ya que este proyecto académico no cuenta con un flujo real de datos de producción "
    "(ver docstring de `model_monitoring.py`). La lógica de cálculo de métricas y alertas "
    "es la misma que se usaría con datos reales."
)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.header("Filtros")
selected_period = st.sidebar.selectbox("Período a analizar", period_names, index=len(period_names) - 1)
selected_var_numeric = st.sidebar.selectbox("Variable numérica", feature_lists["numeric"], index=feature_lists["numeric"].index("log_salario_cliente"))

# --------------------------------------------------------------------------- #
# Indicadores tipo semáforo
# --------------------------------------------------------------------------- #
period_report = drift_report[drift_report["periodo"] == selected_period]
n_severo = (period_report["alerta"] == "severo").sum()
n_moderado = period_report["alerta"].isin(["moderado", "drift_detectado"]).sum()
n_ok = (period_report["alerta"] == "sin_drift").sum()

col1, col2, col3 = st.columns(3)
col1.metric("🔴 Drift severo", int(n_severo))
col2.metric("🟡 Drift moderado / detectado", int(n_moderado))
col3.metric("🟢 Sin drift", int(n_ok))

st.divider()

# --------------------------------------------------------------------------- #
# Comparación de distribuciones histórica vs. actual
# --------------------------------------------------------------------------- #
st.subheader(f"Distribución histórica vs. actual — `{selected_var_numeric}` ({selected_period})")
fig, ax = plt.subplots(figsize=(9, 4))
sns.kdeplot(reference[selected_var_numeric].dropna(), label="Histórico (referencia)", fill=True, alpha=0.35, ax=ax)
sns.kdeplot(periods[selected_period][selected_var_numeric].dropna(), label=f"Actual ({selected_period})", fill=True, alpha=0.35, ax=ax)
ax.legend()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Tabla de métricas de drift
# --------------------------------------------------------------------------- #
st.subheader(f"Tabla de métricas de drift por variable — {selected_period}")


def _highlight_alert(val: str) -> str:
    colors = {
        "severo": "background-color: #f8d7da",
        "moderado": "background-color: #fff3cd",
        "drift_detectado": "background-color: #fff3cd",
        "sin_drift": "background-color: #d4edda",
    }
    return colors.get(val, "")


st.dataframe(
    period_report[["variable", "tipo", "metrica", "valor", "alerta"]]
    .sort_values(["alerta", "variable"])
    .style.map(_highlight_alert, subset=["alerta"])
    .format({"valor": "{:.4f}"}),
    width="stretch",
    height=400,
)

st.divider()

# --------------------------------------------------------------------------- #
# Evolución temporal del drift
# --------------------------------------------------------------------------- #
st.subheader(f"Evolución del drift en el tiempo (PSI) — `{selected_var_numeric}`")
psi_evo = drift_report[(drift_report["metrica"] == "PSI") & (drift_report["variable"] == selected_var_numeric)]

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(psi_evo["periodo"], psi_evo["valor"], marker="o", color="#4C72B0")
ax2.axhline(PSI_THRESHOLDS["moderado"], color="orange", linestyle="--", label="umbral moderado (0.10)")
ax2.axhline(PSI_THRESHOLDS["severo"], color="red", linestyle="--", label="umbral severo (0.25)")
ax2.set_ylabel("PSI")
ax2.legend()
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

vals = psi_evo["valor"].to_numpy()
if len(vals) >= 2:
    deltas = np.diff(vals)
    if deltas.max(initial=0) > PSI_THRESHOLDS["moderado"]:
        idx = int(np.argmax(deltas))
        st.warning(
            f"⚠️ Cambio abrupto detectado: el PSI de `{selected_var_numeric}` subió "
            f"{deltas[idx]:.2f} entre `{psi_evo['periodo'].iloc[idx]}` y "
            f"`{psi_evo['periodo'].iloc[idx + 1]}`. Esto sugiere un cambio repentino en la "
            f"población, no una deriva gradual."
        )

st.divider()

# --------------------------------------------------------------------------- #
# Alertas y recomendaciones
# --------------------------------------------------------------------------- #
st.subheader("🔔 Alertas y recomendaciones")
period_alerts = [a for a in alerts if a["periodo"] == selected_period]

if not period_alerts:
    st.success("No se detectaron alertas para este período: la distribución de los datos se mantiene estable respecto al histórico.")
else:
    for a in sorted(period_alerts, key=lambda x: x["nivel"] != "severo"):
        if a["nivel"] == "severo":
            st.error(f"{a['mensaje']}\n\n**Recomendación:** iniciar proceso de reentrenamiento del modelo y validar con negocio el cambio en la población.")
        else:
            st.warning(f"{a['mensaje']}\n\n**Recomendación:** monitorear de cerca en el siguiente período; si la tendencia continúa, evaluar reentrenamiento.")

with st.expander("¿Qué significa cada métrica?"):
    st.markdown(
        """
- **PSI (Population Stability Index):** mide qué tanto cambió la distribución de una
  variable numérica. `< 0.10` sin drift relevante, `0.10 - 0.25` drift moderado,
  `> 0.25` drift severo (regla estándar de la industria).
- **KS (Kolmogorov-Smirnov):** compara las distribuciones acumuladas de dos muestras
  numéricas; se reporta como drift si el p-value < 0.05. Con muestras grandes puede
  marcar diferencias mínimas como "significativas" — por eso se usa junto con PSI,
  que sí refleja la magnitud del cambio.
- **Jensen-Shannon:** divergencia entre dos distribuciones de probabilidad (basada en
  histogramas), acotada entre 0 y 1; valores más altos indican mayor separación.
- **Chi-cuadrado:** equivalente a KS pero para variables categóricas — compara la
  proporción de cada categoría entre el histórico y el período actual.
        """
    )
