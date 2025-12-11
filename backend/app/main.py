from __future__ import annotations

import pathlib
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import load_settings
from .schemas import (
    AggregatePrediction,
    EvaluateResponse,
    HealthResponse,
    ModelInfo,
    PredictResponse,
    WindowPrediction,
)
from .service import ModelService

settings = load_settings()
try:
    service = ModelService(settings)
except FileNotFoundError:
    service = None  # Lazy-loaded in dependency to allow tests with overrides

app = FastAPI(
    title="MHealth HAR API",
    version="1.0.0",
    description="API de reconocimiento de actividad humana usando MHealth.",
)

origins = [o.strip() for o in settings.allowed_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfo)
def _get_service() -> ModelService:
    if service is None:
        raise HTTPException(
            status_code=500, detail="Modelo no disponible. Entrene primero."
        )
    return service


@app.get("/model-info", response_model=ModelInfo)
def model_info(svc: ModelService = Depends(_get_service)) -> ModelInfo:
    payload = svc.model_info_payload()
    return ModelInfo(**payload)


def _validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo no proporcionado.")
    if not file.filename.endswith(".log"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .log.")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...), svc: ModelService = Depends(_get_service)
) -> PredictResponse:
    _validate_file(file)
    result = svc.predict(file)
    return PredictResponse(**result)


@app.post("/evaluate-log", response_model=EvaluateResponse)
async def evaluate_log(
    file: UploadFile = File(...), svc: ModelService = Depends(_get_service)
) -> EvaluateResponse:
    _validate_file(file)
    result = svc.evaluate(file)
    return EvaluateResponse(
        metrics=result["metrics"],
        predictions=result["predictions"],
        ground_truth=result["ground_truth"],
    )
