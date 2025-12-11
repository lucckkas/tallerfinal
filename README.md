# MHealth HAR - Fin de curso

Sistema completo de reconocimiento de actividad humana con un único modelo entrenado sobre el dataset MHealth. Incluye pipeline ML reproducible, API FastAPI lista para producción y frontend React minimalista.

## Arquitectura

- `ml/`: pipeline de datos (descarga, ventanas, extracción de características, normalización) y scripts `train.py`, `evaluate.py`, `infer.py`. Modelo: RandomForest sobre features estadísticas por ventana.
- `backend/`: servicio FastAPI que carga los artefactos y expone `/health`, `/model-info`, `/predict`, `/evaluate-log`.
- `frontend/`: UI Vite + React + TypeScript para subir `.log`, ver predicciones por ventana y métricas con matriz de confusión.
- `config/config.yaml`: parámetros de ventana, solapamiento, sujetos excluidos demo, rutas de artefactos, semilla y versión.
- `docker-compose.yml`: orquesta backend + frontend. `prompts/`: trazabilidad de uso de IA.

## Supuestos clave

- Sujetos excluidos para demo (nunca usados en entrenamiento/normalización): **9 y 10** (regla determinística: los dos IDs más altos del dataset).
- Frecuencia de muestreo: 50 Hz. Ventana: 5s con solapamiento 2.5s. Estadísticas por sensor: mean, std, min, max, median, mad, energy.
- Modelo único: `RandomForestClassifier` con `n_estimators=200`, `class_weight=balanced`. Semilla global: 42.

## Estructura

```
config/config.yaml         # Configuración central
ml/src/mhealth/            # Código del pipeline
ml/train.py                # Entrenamiento completo
ml/evaluate.py             # Evaluación (split o archivo .log)
ml/infer.py                # Inferencia rápida sobre un .log
backend/app/               # FastAPI app, esquemas y servicio
frontend/                  # Vite + React UI
docker-compose.yml         # Levanta backend y frontend
prompts/                   # Prompts usados
```

## Entrenamiento y artefactos

1) Crear entorno:
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r ml/requirements.txt
```
2) Ejecutar entrenamiento (descarga automática del dataset desde UCI):
```bash
PYTHONPATH=ml/src python ml/train.py --config config/config.yaml
```
Artefactos generados en `ml/artifacts/`:
- `model.joblib` (pipeline scaler + RF)
- `features.json` (columnas de features)
- `metrics.json` (accuracy, macro F1, matriz de confusión para train/val/test/demo)
- `model_info.json` (versión, hiperparámetros, semillas, splits usados y sujetos demo)

> Si quieres probar el flujo end-to-end sin reentrenar, copia `mHealth_subject9.log` y `mHealth_subject10.log` desde el dataset a `ml/demo_logs/`.

## Evaluación e inferencia standalone

- Evaluar split guardado (por defecto test):
```bash
PYTHONPATH=ml/src python ml/evaluate.py --config config/config.yaml --split test
```
- Evaluar o predecir un archivo `.log`:
```bash
PYTHONPATH=ml/src python ml/evaluate.py --config config/config.yaml --log path/al/archivo.log --subject-id 99
PYTHONPATH=ml/src python ml/infer.py path/al/archivo.log --config config/config.yaml --subject-id 99
```

## Backend FastAPI

Instalación local:
```bash
pip install -r backend/requirements.txt -r ml/requirements.txt
export PYTHONPATH=ml/src
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
Endpoints:
- `GET /health`
- `GET /model-info`
- `POST /predict` (archivo `.log`, devuelve predicción por ventana y resumen agregado)
- `POST /evaluate-log` (archivo `.log` con etiqueta en última columna, devuelve métricas y matriz de confusión)

Tests API:
```bash
pytest backend/tests
```

## Frontend React (Vite + TS)

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173 (requiere backend en 8000)
npm run lint
npm run build
```
La UI permite:
- Subir `.log` para predicción: muestra distribución por actividad y detalle por ventana.
- Subir `.log` etiquetado para evaluación: muestra accuracy, macro F1 y matriz de confusión.
- Consultar parámetros del modelo (versión, ventana, sujetos demo, splits).

## Docker / Compose

Construir imágenes:
```bash
docker-compose build
```
Levantar stack:
```bash
docker-compose up
```
Accesos: backend `http://localhost:8000`, frontend `http://localhost:5173`.

## Calidad y mantenimiento

- Código modular, reproducible con semilla fija.
- Split por sujeto para evitar fugas; sujetos 9 y 10 reservados como demo/test UI.
- Config centralizada en `config/config.yaml` y `.env` (ver `.env.example`).
- Lint frontend (`npm run lint`) y tests backend (`pytest backend/tests`).

## Mejoras futuras

- Persistir y versionar artefactos (DVC/MLflow) y publicar contenedores en un registry.
- Añadir monitoreo (logs estructurados, métricas Prometheus) y tracing en FastAPI.
- Extender UI con gráficos de probabilidades y soporte drag&drop para múltiples archivos.
- Evaluar modelos livianos (LightGBM / XGBoost) y poda de features para despliegues edge.
