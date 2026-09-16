# Modelo de Riesgo Crediticio — Proyecto Integrador (Henry, PIM5 - Data Science)

## Caso de negocio

El equipo de Datos y Analítica de una empresa financiera necesita anticipar el
comportamiento de pago de nuevos solicitantes de crédito. El objetivo de este
proyecto es construir, versionar y desplegar un modelo de aprendizaje
automático supervisado que, a partir del historial de créditos, prediga la
variable `Pago_atiempo` (si el cliente pagará su crédito a tiempo o no),
siguiendo prácticas de MLOps (versionamiento, reproducibilidad, monitoreo y
despliegue como servicio).

Este repositorio es el entregable final del Proyecto Integrador y documenta
cada avance del proceso: desde la carga y comprensión de los datos, pasando
por la ingeniería de características y el modelado, hasta el despliegue del
modelo como API y su monitoreo en producción.

## Estructura del repositorio

La estructura de carpetas fue definida por el equipo de MLOps de la empresa y
**no debe modificarse**, ya que el pipeline de validación en Jenkins depende
de ella:

```
mlops_pipeline/
├── src/
│   ├── Cargar_datos.ipynb            # Carga del dataset de ejemplo (.csv)
│   ├── comprension_eda.ipynb         # Análisis exploratorio de datos (EDA)
│   ├── ft_engineering.py             # Ingeniería de características
│   ├── model_training_evaluation.py  # Entrenamiento y evaluación de modelos
│   ├── model_deploy.py               # Despliegue del modelo como API
│   ├── model_monitoring.py           # Monitoreo y detección de data drift
│   └── streamlit_app.py              # Dashboard interactivo de monitoreo
├── Base_de_datos.csv                 # Dataset de ejemplo (no productivo)
├── requirements.txt                  # Dependencias del proyecto
├── .gitignore
└── readme.md
```

Además de estos archivos fijos, el pipeline genera dos carpetas de salida al
ejecutar `ft_engineering.py` y `model_training_evaluation.py`:

- `models/` — modelo ganador (`best_model.joblib`) y sus metadatos
  (`model_metadata.json`). Se versiona porque `model_deploy.py` lo necesita.
- `reports/` — tablas comparativas y gráficos de evaluación de modelos, y
  `reports/drift/` con las métricas, alertas y gráficos de data drift
  generados por `model_monitoring.py`.
- `data/processed/` — `train.csv` / `test.csv` intermedios (no se versionan,
  se regeneran corriendo `ft_engineering.py`).

> En un entorno productivo, `Base_de_datos.csv` no se versionaría en el
> repositorio: la información llegaría desde el DWH/Datalake de la empresa a
> través de otro proceso. Para este ejercicio académico se utiliza un dataset
> de ejemplo no productivo.

## Flujo de ramas y versiones

El repositorio sigue un flujo de tres ramas: `developer`, `certification` y
`main`. Cada rama parte de una estructura de carpetas idéntica (`V1.0.0`) y
evoluciona mediante *pull requests* revisados por un compañero antes de
integrarse a `main`.

| Versión | Rama de trabajo | Contenido |
|---|---|---|
| V1.0.0 | main / certification / developer | Estructura de carpetas inicial (punto de partida) |
| V1.0.1 | developer → main | `Cargar_datos.ipynb` + `comprension_eda.ipynb` |
| V1.1.0 | developer → main | Ingeniería de características y modelado (`ft_engineering.py`) |
| V1.1.1 | developer → main | Monitoreo de data drift y app en Streamlit |
| ... | developer → main | Despliegue con FastAPI + Docker |

## Instalación y entorno local

```bash
# Clonar el repositorio
git clone <URL-del-repositorio>
cd mlops_pipeline

# Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Registrar el kernel para Jupyter (opcional, para trabajar los notebooks)
python -m ipykernel install --user --name=mlops_pipeline --display-name="mlops_pipeline"
```

### Ejecutar el pipeline y el dashboard de monitoreo

```bash
cd src
python ft_engineering.py             # genera data/processed/{train,test}.csv
python model_training_evaluation.py  # entrena y guarda models/best_model.joblib
python model_monitoring.py           # calcula métricas de drift -> reports/drift/
streamlit run streamlit_app.py       # dashboard interactivo de monitoreo
```

## Estado del proyecto

- [x] **Avance 1 — Versionamiento y colaboración:** estructura de carpetas,
  ramas, `requirements.txt`, `Cargar_datos.ipynb` y `comprension_eda.ipynb`.
- [x] **Avance 2 — Ingeniería de características y modelado:** `ft_engineering.py`,
  entrenamiento y evaluación de modelos supervisados.
- [x] **Avance 3 — Monitoreo y aplicación:** `model_monitoring.py`, data drift,
  app en Streamlit.
- [ ] **Avance 4 — Despliegue:** `model_deploy.py`, API con FastAPI, imagen Docker.

## Hallazgos y decisiones clave

- **`puntaje` presenta una correlación ~0.92 con `Pago_atiempo`** y, al
  incluirla como feature, cualquier modelo alcanza ROC-AUC ≈ 1.0 — una
  confirmación empírica de fuga de información (ver
  `reports/model_comparison_con_puntaje.csv`). Por esto **se excluye del
  modelo de producción**.
- **Sin `puntaje`**, los tres modelos candidatos (regresión logística, random
  forest, XGBoost) alcanzan un ROC-AUC realista de **~0.66**. Se seleccionó
  **random forest** como modelo ganador por tener el mejor F1/ROC-AUC de
  forma consistente (ver `reports/model_comparison_sin_puntaje.csv` y
  `models/model_metadata.json`).
- El desbalance de clases (~95%/5%) se maneja con `class_weight="balanced"`
  (regresión logística / random forest) y `scale_pos_weight` (XGBoost), y se
  evalúa con precision, recall, F1 y ROC-AUC en vez de accuracy.
- Variables derivadas: relación cuota/salario, relación deuda/salario,
  antigüedad del crédito, bandera de mora, transformación logarítmica de
  variables monetarias y bucket de edad.
- Preprocesamiento (feature-engine): winsorización de outliers en
  `salario_cliente`/`total_otros_prestamos`, agrupación de categorías raras
  en `tipo_credito`, imputación explícita de `tendencia_ingresos` como
  "Sin_informacion", indicadores + imputación de mediana para las variables
  de saldo con nulos, one-hot encoding y escalado — todo ajustado únicamente
  sobre el conjunto de entrenamiento para evitar fuga de información desde el
  test set.

- **Monitoreo de data drift** (`model_monitoring.py`): compara la
  distribución de nuevos lotes de datos contra la distribución de
  entrenamiento usando PSI, KS, Jensen-Shannon (numéricas) y Chi-cuadrado
  (categóricas), con umbrales estándar de industria (PSI < 0.10 sin drift,
  0.10-0.25 moderado, > 0.25 severo). Como el proyecto no cuenta con datos
  reales de producción, se simulan períodos con drift creciente
  (`simulate_periods`) para validar que la lógica de detección funciona: por
  ejemplo, un corrimiento simulado en el ingreso declarado (`log_salario_cliente`)
  se detecta correctamente como drift severo (PSI > 1) a partir del primer
  período simulado (ver `reports/drift/psi_heatmap.png`).
- El test KS puede marcar drift "significativo" en variables con diferencias
  mínimas cuando el tamaño de muestra es grande (p. ej. `huella_consulta`
  incluso sin ninguna perturbación introducida) — por eso las alertas de
  severidad se basan principalmente en PSI, que sí refleja la magnitud real
  del cambio, y KS/Chi-cuadrado se usan como corroboración estadística.
- El dashboard en Streamlit (`streamlit_app.py`) permite elegir período y
  variable, muestra indicadores tipo semáforo, la tabla de métricas, la
  evolución del PSI en el tiempo (con detección de cambios abruptos) y las
  alertas con su recomendación (reentrenar vs. monitorear).

## Próximos pasos

El Avance 4 (`model_deploy.py`: API con FastAPI + imagen Docker) se
documentará aquí una vez completado.
