from __future__ import annotations

import pathlib
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .constants import ACTIVITY_MAP, LABEL_COLUMN, SENSOR_COLUMNS, SUBJECT_COLUMN
from .preprocess import create_windows, filter_unlabeled_activity
from .utils import load_json


def load_artifacts(config: Config):
    model = joblib.load(config.artifacts["model_path"])
    feature_meta = load_json(config.artifacts["feature_metadata"])
    feature_columns = feature_meta["feature_columns"]
    model_info = load_json(config.artifacts["model_info"])
    return model, feature_columns, model_info


def prepare_features_from_log(
    log_path: pathlib.Path, config: Config, subject_id: int = 0
) -> pd.DataFrame:
    df = pd.read_csv(
        log_path,
        sep=r"\s+",
        header=None,
    )
    # Assume last column is label if present, otherwise fill with -1
    if df.shape[1] == len(SENSOR_COLUMNS):
        df[len(SENSOR_COLUMNS)] = -1
    elif df.shape[1] != len(SENSOR_COLUMNS) + 1:
        raise ValueError("Unexpected log format; expected 23 or 24 columns.")

    df["timestamp"] = np.arange(len(df)) / float(config.sample_rate_hz)
    df[SUBJECT_COLUMN] = subject_id
    # Re-assign sensor columns and label names
    data = df.copy()
    col_names = SENSOR_COLUMNS + [LABEL_COLUMN]
    data.columns = col_names + ["timestamp", SUBJECT_COLUMN]

    windows = create_windows(
        data,
        config.window_seconds,
        config.window_overlap_seconds,
        config.sample_rate_hz,
        feature_stats=config.features.get("stats"),
    )

    # Filter out activity 0 (unlabeled) to match training data
    # Only filter if we have ground truth labels (not -1)
    if (windows[LABEL_COLUMN] != -1).any():
        windows = windows[windows[LABEL_COLUMN] != 0].copy()

    return windows


def ensure_feature_order(
    features: pd.DataFrame, feature_columns: List[str]
) -> pd.DataFrame:
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0.0
    return features[feature_columns]


def predict_windows(
    model,
    features: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, object]:
    ordered = ensure_feature_order(features, feature_columns)
    proba = model.predict_proba(ordered)
    preds = model.predict(ordered)
    classes = list(model.classes_)
    per_window = []
    for idx, (pred, probs) in enumerate(zip(preds, proba)):
        per_window.append(
            {
                "window_index": idx,
                "prediction": int(pred),
                "activity": ACTIVITY_MAP.get(int(pred), str(pred)),
                "proba": {
                    ACTIVITY_MAP.get(int(cls), str(cls)): float(p)
                    for cls, p in zip(classes, probs)
                },
            }
        )
    agg = aggregate_predictions(preds, proba, classes)
    return {"per_window": per_window, "aggregate": agg}


def aggregate_predictions(
    preds: np.ndarray, proba: np.ndarray, classes: List[int]
) -> Dict[str, object]:
    df = pd.DataFrame({"pred": preds})
    agg = df["pred"].value_counts(normalize=True)
    summary = {
        ACTIVITY_MAP.get(int(label), str(label)): float(freq)
        for label, freq in agg.items()
    }
    mean_proba = proba.mean(axis=0)
    proba_summary = {
        ACTIVITY_MAP.get(int(cls), str(cls)): float(p)
        for cls, p in zip(classes, mean_proba)
    }
    return {"fraction_per_activity": summary, "mean_proba": proba_summary}
