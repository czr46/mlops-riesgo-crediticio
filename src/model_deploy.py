"""
model_deploy.py

Despliegue del modelo de riesgo crediticio como servicio HTTP (Avance 4 -
Proyecto Integrador). Carga el modelo ganador (`models/best_model.joblib`,
un `Pipeline` de preprocesamiento + `random_forest`) y lo expone con FastAPI:

- `POST /predict`       — una sola solicitud de crédito (JSON).
- `POST /predict/batch` — varias solicitudes en un solo request (JSON, lista).
- `POST /predict/csv`   — un archivo CSV con una o más solicitudes.
- `GET  /health`        — estado del servicio y metadatos del modelo cargado.

Los campos de entrada son las columnas **crudas** del dataset original (las
mismas que trae `Base_de_datos.csv`, sin `Pago_atiempo` ni `puntaje`, que se
excluye por fuga de información — ver `reports/model_comparison_con_puntaje.csv`).
Internamente se aplican las mismas reglas de limpieza e ingeniería de
características que en el entrenamiento (`ft_engineering.py`) antes de
invocar al modelo.

Uso local:
    uvicorn model_deploy:app --reload --port 8000

Uso con Docker (ver Dockerfile en la raíz del repo):
    docker build -t riesgo-crediticio-api .
    docker run -p 8000:8000 riesgo-crediticio-api

Documentación interactiva una vez levantado el servicio: http://localhost:8000/docs
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ft_engineering import clean_data, get_feature_lists

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "best_model.joblib"
DATA_PATH = REPO_ROOT / "Base_de_datos.csv"

FEATURE_LISTS = get_feature_lists(include_leaky=False)
NUMERIC_FEATURES = FEATURE_LISTS["numeric"]
CATEGORICAL_FEATURES = FEATURE_LISTS["categorical"]

# Umbrales para traducir la probabilidad de pago a tiempo en una etiqueta de
# riesgo legible para negocio (criterio propio del proyecto, documentado aquí).
RISK_THRESHOLDS = {"bajo": 0.70, "medio": 0.40}

# Estado compartido, se llena en el lifespan de la app (evita recargar el
# modelo y el csv en cada request).
state: dict = {"model": None, "reference_date": None, "model_version": None}


# --------------------------------------------------------------------------- #
# Ingeniería de características para requests nuevos
# --------------------------------------------------------------------------- #
def _load_reference_date() -> pd.Timestamp:
    """Calcula la fecha de referencia (snapshot) usada para `antiguedad_meses`.

    En `ft_engineering.engineer_features` esa fecha se calcula como el máximo
    de `fecha_prestamo` del propio dataset que se está transformando. Eso
    funciona para el entrenamiento (se transforma todo el dataset a la vez),
    pero no tiene sentido para una solicitud nueva de 1 o pocas filas: el
    "máximo" sería la propia fecha de la solicitud, dando `antiguedad_meses`
    ~0 siempre. Por eso, para servir el modelo se reutiliza la MISMA fecha de
    referencia que se usó al entrenar (el snapshot del dataset de
    entrenamiento), para que la variable signifique lo mismo en producción
    que en entrenamiento.
    """
    df = pd.read_csv(DATA_PATH, usecols=["fecha_prestamo"], parse_dates=["fecha_prestamo"])
    return df["fecha_prestamo"].max() + pd.Timedelta(days=1)


def _engineer_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Réplica de `ft_engineering.engineer_features`, con fecha de referencia fija.

    Debe mantenerse alineada con esa función si se modifica la ingeniería de
    características en `ft_engineering.py`.
    """
    df = df.copy()

    df["ratio_cuota_salario"] = df["cuota_pactada"] / (df["salario_cliente"] + 1)
    df["ratio_deuda_salario"] = df["total_otros_prestamos"] / (df["salario_cliente"] + 1)

    df["antiguedad_meses"] = (reference_date - df["fecha_prestamo"]).dt.days / 30

    df["flag_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)

    for col in ["capital_prestado", "salario_cliente", "total_otros_prestamos", "cuota_pactada"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    df["edad_bucket"] = pd.cut(
        df["edad_cliente"],
        bins=[17, 25, 35, 45, 55, 65, 90],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66-90"],
    ).astype(object)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _risk_label(prob_pago_atiempo: float) -> str:
    if prob_pago_atiempo >= RISK_THRESHOLDS["bajo"]:
        return "bajo"
    if prob_pago_atiempo >= RISK_THRESHOLDS["medio"]:
        return "medio"
    return "alto"


def run_predictions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Corre el pipeline completo (limpieza -> features -> modelo) sobre un
    DataFrame de solicitudes crudas y retorna las predicciones."""
    if state["model"] is None:
        raise RuntimeError("El modelo no está cargado.")

    missing = set(RAW_INPUT_COLUMNS) - set(df_raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df_raw.copy()
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

    df = clean_data(df)
    df = _engineer_features(df, state["reference_date"])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba_pago_atiempo = state["model"].predict_proba(X)[:, 1]
    pred = (proba_pago_atiempo >= 0.5).astype(int)

    return pd.DataFrame({
        "prediccion_pago_atiempo": pred,
        "probabilidad_pago_atiempo": proba_pago_atiempo.round(4),
        "nivel_riesgo": [_risk_label(p) for p in proba_pago_atiempo],
    })


# --------------------------------------------------------------------------- #
# Esquema de entrada (columnas crudas del dataset, sin `puntaje` ni el target)
# --------------------------------------------------------------------------- #
class SolicitudCredito(BaseModel):
    tipo_credito: str = Field(..., description="Código del tipo de crédito solicitado")
    fecha_prestamo: date = Field(..., description="Fecha de desembolso del crédito")
    capital_prestado: float = Field(..., ge=0, description="Monto solicitado")
    plazo_meses: int = Field(..., gt=0, description="Plazo del crédito en meses")
    edad_cliente: float = Field(..., ge=0, description="Edad del solicitante")
    tipo_laboral: str = Field(..., description="Situación laboral del solicitante")
    salario_cliente: float = Field(..., ge=0, description="Salario declarado")
    total_otros_prestamos: float = Field(..., ge=0, description="Saldo de otras deudas vigentes")
    cuota_pactada: float = Field(..., ge=0, description="Cuota mensual pactada del crédito")
    puntaje_datacredito: Optional[float] = Field(None, description="Score de la central de riesgo")
    cant_creditosvigentes: int = Field(..., ge=0, description="Cantidad de créditos vigentes")
    huella_consulta: int = Field(..., ge=0, description="Cantidad de consultas recientes a la central de riesgo")
    saldo_mora: Optional[float] = Field(None, ge=0, description="Saldo actualmente en mora")
    saldo_total: Optional[float] = Field(None, ge=0, description="Saldo total de deuda vigente")
    saldo_principal: Optional[float] = Field(None, ge=0, description="Saldo principal de deuda vigente")
    saldo_mora_codeudor: Optional[float] = Field(None, ge=0, description="Saldo en mora asociado a codeudores")
    creditos_sectorFinanciero: int = Field(..., ge=0)
    creditos_sectorCooperativo: int = Field(..., ge=0)
    creditos_sectorReal: int = Field(..., ge=0)
    promedio_ingresos_datacredito: Optional[float] = Field(None, description="Ingreso promedio reportado a la central de riesgo")
    tendencia_ingresos: Optional[Literal["Creciente", "Estable", "Decreciente"]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "tipo_credito": "12",
                "fecha_prestamo": "2024-03-15",
                "capital_prestado": 5000000,
                "plazo_meses": 24,
                "edad_cliente": 35,
                "tipo_laboral": "Empleado",
                "salario_cliente": 2500000,
                "total_otros_prestamos": 800000,
                "cuota_pactada": 250000,
                "puntaje_datacredito": 650,
                "cant_creditosvigentes": 2,
                "huella_consulta": 1,
                "saldo_mora": 0,
                "saldo_total": 1200000,
                "saldo_principal": 1000000,
                "saldo_mora_codeudor": 0,
                "creditos_sectorFinanciero": 1,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 1,
                "promedio_ingresos_datacredito": 2400000,
                "tendencia_ingresos": "Estable",
            }
        }


RAW_INPUT_COLUMNS = list(SolicitudCredito.model_fields.keys())


class PrediccionResponse(BaseModel):
    prediccion_pago_atiempo: int
    probabilidad_pago_atiempo: float
    nivel_riesgo: str


class BatchPrediccionResponse(BaseModel):
    resultados: list[PrediccionResponse]


class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    modelo: Optional[str] = None
    n_variables_numericas: int
    n_variables_categoricas: int


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta antes `python ft_engineering.py` y `python model_training_evaluation.py`."
        )
    state["model"] = joblib.load(MODEL_PATH)
    state["reference_date"] = _load_reference_date()
    state["model_version"] = type(state["model"].named_steps["estimator"]).__name__
    yield
    state["model"] = None


app = FastAPI(
    title="API de Riesgo Crediticio",
    description=(
        "Predice si un nuevo solicitante de crédito pagará a tiempo "
        "(`Pago_atiempo`), a partir de sus datos de solicitud y de central "
        "de riesgo. Proyecto Integrador PIM5 (Henry, Data Science)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state["model"] is not None else "modelo no cargado",
        modelo_cargado=state["model"] is not None,
        modelo=state["model_version"],
        n_variables_numericas=len(NUMERIC_FEATURES),
        n_variables_categoricas=len(CATEGORICAL_FEATURES),
    )


@app.post("/predict", response_model=PrediccionResponse)
def predict(solicitud: SolicitudCredito) -> PrediccionResponse:
    df_raw = pd.DataFrame([solicitud.model_dump()])
    try:
        resultado = run_predictions(df_raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PrediccionResponse(**resultado.iloc[0].to_dict())


@app.post("/predict/batch", response_model=BatchPrediccionResponse)
def predict_batch(solicitudes: list[SolicitudCredito]) -> BatchPrediccionResponse:
    if not solicitudes:
        raise HTTPException(status_code=422, detail="La lista de solicitudes está vacía.")
    df_raw = pd.DataFrame([s.model_dump() for s in solicitudes])
    try:
        resultado = run_predictions(df_raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return BatchPrediccionResponse(resultados=[PrediccionResponse(**row) for row in resultado.to_dict("records")])


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(..., description="CSV con las mismas columnas que /predict")):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="El archivo debe ser un .csv")

    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"No se pudo leer el CSV: {e}")

    try:
        resultado = run_predictions(df_raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    salida = pd.concat([df_raw.reset_index(drop=True), resultado], axis=1)
    buffer = io.StringIO()
    salida.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predicciones.csv"},
    )


if __name__ == "__main__":
    import os

    import uvicorn

    # Por defecto solo escucha en localhost (127.0.0.1): correr esto
    # directamente en tu maquina no deberia exponer la API a toda tu red.
    # Dentro de un contenedor Docker SI hace falta escuchar en todas las
    # interfaces para que el puerto publicado (`docker run -p ...`) llegue
    # al proceso -- eso se resuelve aparte, con el CMD del Dockerfile
    # (`uvicorn model_deploy:app --host 0.0.0.0 ...`), no aqui.
    host = os.getenv("API_HOST", "127.0.0.1")
    uvicorn.run("model_deploy:app", host=host, port=8000, reload=True)
