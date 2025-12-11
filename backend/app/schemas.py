from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class WindowPrediction(BaseModel):
    window_index: int
    prediction: int
    activity: str
    proba: Dict[str, float]


class AggregatePrediction(BaseModel):
    fraction_per_activity: Dict[str, float]
    mean_proba: Dict[str, float]


class PredictResponse(BaseModel):
    per_window: List[WindowPrediction]
    aggregate: AggregatePrediction


class ModelInfo(BaseModel):
    version: str
    model_type: str
    random_seed: int
    window_seconds: float
    window_overlap_seconds: float
    sample_rate_hz: int
    excluded_subjects_demo: List[int]
    splits: dict
    feature_columns: List[str]
    metrics: Optional[dict]


class EvaluationMetrics(BaseModel):
    accuracy: Optional[float]
    macro_f1: Optional[float]
    confusion_matrix: List[List[int]]


class EvaluateResponse(BaseModel):
    metrics: EvaluationMetrics
    predictions: Optional[List[int]] = None
    ground_truth: Optional[List[int]] = None
