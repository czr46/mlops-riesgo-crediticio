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
│   └── model_monitoring.py           # Monitoreo y detección de data drift
├── Base_de_datos.csv                 # Dataset de ejemplo (no productivo)
├── requirements.txt                  # Dependencias del proyecto
├── .gitignore
└── readme.md
```

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

## Estado del proyecto

- [x] **Avance 1 — Versionamiento y colaboración:** estructura de carpetas,
  ramas, `requirements.txt`, `Cargar_datos.ipynb` y `comprension_eda.ipynb`.
- [ ] **Avance 2 — Ingeniería de características y modelado:** `ft_engineering.py`,
  entrenamiento y evaluación de modelos supervisados.
- [ ] **Avance 3 — Monitoreo y aplicación:** `model_monitoring.py`, data drift,
  app en Streamlit.
- [ ] **Avance 4 — Despliegue:** `model_deploy.py`, API con FastAPI, imagen Docker.

Los principales hallazgos del análisis exploratorio y las decisiones de
modelado se documentarán en esta sección a medida que avance el proyecto.
