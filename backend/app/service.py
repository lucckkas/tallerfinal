from __future__ import annotations

import pathlib
import sys
import tempfile
from typing import Any, Dict

import numpy as np
from fastapi import HTTPException, UploadFile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Ensure mhealth package is importable
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "ml" / "src"))

from mhealth.config import load_config
from mhealth.constants import LABEL_COLUMN, SUBJECT_COLUMN
from mhealth.inference import (
    ensure_feature_order,
    load_artifacts,
    predict_windows,
    prepare_features_from_log,
)
from mhealth.utils import load_json

from .config import Settings


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = load_config(settings.config_yaml)
        self.model, self.feature_columns, self.model_info = load_artifacts(self.config)
        try:
            self.metrics = load_json(self.settings.metrics_artifact)
        except FileNotFoundError:
            self.metrics = None

    def _save_upload(self, file: UploadFile) -> pathlib.Path:
        suffix = pathlib.Path(file.filename or "upload.log").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = file.file.read()
            tmp.write(content)
            return pathlib.Path(tmp.name)

    def predict(self, file: UploadFile) -> Dict[str, Any]:
        path = self._save_upload(file)
        try:
            windows = prepare_features_from_log(path, self.config, subject_id=0)
            feature_df = windows.drop(columns=[LABEL_COLUMN, SUBJECT_COLUMN])
            return predict_windows(self.model, feature_df, self.feature_columns)
        except Exception as exc:  # pragma: no cover - safety net
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            path.unlink(missing_ok=True)

    def evaluate(self, file: UploadFile) -> Dict[str, Any]:
        path = self._save_upload(file)
        try:
            windows = prepare_features_from_log(path, self.config, subject_id=0)
            feature_df = windows.drop(columns=[LABEL_COLUMN, SUBJECT_COLUMN])
            feature_df = ensure_feature_order(feature_df, self.feature_columns)
            preds = self.model.predict(feature_df)
            metrics = {}
            if (
                windows[LABEL_COLUMN].nunique() > 1
                or windows[LABEL_COLUMN].unique()[0] != -1
            ):
                y_true = windows[LABEL_COLUMN]
                metrics = {
                    "accuracy": accuracy_score(y_true, preds),
                    "macro_f1": f1_score(y_true, preds, average="macro"),
                    "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="El archivo no contiene etiquetas en la última columna para evaluar.",
                )
            return {
                "metrics": metrics,
                "predictions": preds.tolist(),
                "ground_truth": y_true.tolist(),
            }
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            path.unlink(missing_ok=True)

    def model_info_payload(self) -> Dict[str, Any]:
        info = self.model_info
        info["metrics"] = self.metrics
        return info
