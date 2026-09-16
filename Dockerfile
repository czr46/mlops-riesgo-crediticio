# Imagen para servir el modelo de riesgo crediticio como API (Avance 4).
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas para compilar algunos paquetes de ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python primero (aprovecha el cache de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código, el dataset (necesario para la fecha de referencia de
# `antiguedad_meses`, ver model_deploy.py) y el modelo ya entrenado
COPY src/ ./src/
COPY Base_de_datos.csv .
COPY models/ ./models/

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
