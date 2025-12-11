from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .constants import ACTIVITY_MAP, LABEL_COLUMN, SUBJECT_COLUMN
from .preprocess import (
    build_feature_matrix,
    create_windows,
    filter_unlabeled_activity,
    split_by_subject,
)
from .utils import ensure_dir, save_json, set_global_seed


def train_model(
    df: pd.DataFrame,
    config: Config,
    demo_df: pd.DataFrame = None,
) -> Dict[str, object]:
    """
    Train model on provided dataframe.

    Args:
        df: Training data (should NOT contain demo subjects)
        config: Configuration
        demo_df: Optional demo subjects data for evaluation only
    """
    set_global_seed(config.random_seed)

    # Remove activity 0 (unlabeled) to prevent class imbalance
    df = filter_unlabeled_activity(df)

    # Verify no demo subjects leaked into training data
    training_subjects = set(df[SUBJECT_COLUMN].unique())
    demo_subjects = set(config.excluded_subjects_demo)
    leaked = training_subjects & demo_subjects
    if leaked:
        raise ValueError(
            f"FUGA DE DATOS DETECTADA: Sujetos demo {leaked} encontrados en datos de entrenamiento!"
        )

    print(f"[SEGURIDAD] Sujetos en entrenamiento: {sorted(training_subjects)}")
    print(f"[SEGURIDAD] Sujetos excluidos (demo): {sorted(demo_subjects)}")
    print(f"[SEGURIDAD] Verificación OK: No hay fuga de datos")

    train_df_raw, val_df_raw, test_df_raw = split_by_subject(df, config)

    feature_stats = config.features.get("stats")
    train_windows = create_windows(
        train_df_raw,
        config.window_seconds,
        config.window_overlap_seconds,
        config.sample_rate_hz,
        feature_stats=feature_stats,
    )
    val_windows = create_windows(
        val_df_raw,
        config.window_seconds,
        config.window_overlap_seconds,
        config.sample_rate_hz,
        feature_stats=feature_stats,
    )
    test_windows = create_windows(
        test_df_raw,
        config.window_seconds,
        config.window_overlap_seconds,
        config.sample_rate_hz,
        feature_stats=feature_stats,
    )

    # Process demo data if provided
    if demo_df is not None and len(demo_df) > 0:
        demo_df_filtered = filter_unlabeled_activity(demo_df)
        demo_windows = create_windows(
            demo_df_filtered,
            config.window_seconds,
            config.window_overlap_seconds,
            config.sample_rate_hz,
            feature_stats=feature_stats,
        )
    else:
        demo_windows = pd.DataFrame()

    X_train, y_train = build_feature_matrix(train_windows)
    X_val, y_val = build_feature_matrix(val_windows)
    X_test, y_test = build_feature_matrix(test_windows)
    X_demo, y_demo = build_feature_matrix(demo_windows)

    scaler = StandardScaler()
    clf = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        random_state=config.random_seed,
        class_weight=config.model.class_weight,
        n_jobs=-1,
    )

    pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    metrics = {
        "val": compute_metrics(pipeline, X_val, y_val),
        "test": compute_metrics(pipeline, X_test, y_test),
        "demo": compute_metrics(pipeline, X_demo, y_demo),
        "train": compute_metrics(pipeline, X_train, y_train),
    }

    artifacts = {
        "pipeline": pipeline,
        "feature_columns": list(X_train.columns),
        "metrics": metrics,
        "splits": {
            "train_subjects": sorted(train_df_raw[SUBJECT_COLUMN].unique().tolist()),
            "val_subjects": sorted(val_df_raw[SUBJECT_COLUMN].unique().tolist()),
            "test_subjects": sorted(test_df_raw[SUBJECT_COLUMN].unique().tolist()),
            "demo_subjects": sorted(demo_df[SUBJECT_COLUMN].unique().tolist())
            if demo_df is not None and len(demo_df) > 0
            else config.excluded_subjects_demo,
        },
    }
    return artifacts


def compute_metrics(
    model: Pipeline, X: pd.DataFrame, y_true: pd.Series
) -> Dict[str, object]:
    if X.empty or y_true.empty:
        return {"accuracy": None, "macro_f1": None, "confusion_matrix": []}
    preds = model.predict(X)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    cm = confusion_matrix(y_true, preds)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def save_artifacts(artifacts: Dict[str, object], config: Config) -> None:
    ensure_dir(config.artifacts["dir"])
    pipeline = artifacts["pipeline"]
    joblib.dump(pipeline, config.artifacts["model_path"])

    # Store metrics
    metrics = artifacts["metrics"]
    save_json(config.artifacts["metrics"], metrics)  # type: ignore[arg-type]

    info = {
        "version": config.version,
        "model_type": config.model.type,
        "random_seed": config.random_seed,
        "window_seconds": config.window_seconds,
        "window_overlap_seconds": config.window_overlap_seconds,
        "sample_rate_hz": config.sample_rate_hz,
        "excluded_subjects_demo": config.excluded_subjects_demo,
        "splits": artifacts["splits"],
        "feature_columns": artifacts["feature_columns"],
        "activity_labels": ACTIVITY_MAP,
    }
    save_json(config.artifacts["model_info"], info)  # type: ignore[arg-type]

    feature_cols = artifacts["feature_columns"]
    save_json(
        config.artifacts["feature_metadata"],  # type: ignore[arg-type]
        {"feature_columns": feature_cols},
    )
