from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from mhealth.config import load_config
from mhealth.constants import LABEL_COLUMN, SUBJECT_COLUMN
from mhealth.data import load_dataset
from mhealth.inference import ensure_feature_order, load_artifacts, prepare_features_from_log
from mhealth.preprocess import create_windows, filter_demo_subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MHealth model.")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML.")
    parser.add_argument("--log", help="Optional path to a single .log file for evaluation.")
    parser.add_argument("--subject-id", type=int, default=0, help="Subject id for single log.")
    parser.add_argument("--split", choices=["val", "test", "demo", "train"], default="test")
    return parser.parse_args()


def evaluate_split(df, model, feature_cols, config, split_name: str, splits: dict) -> dict:
    if split_name == "demo":
        subset = df[df[SUBJECT_COLUMN].isin(config.excluded_subjects_demo)].copy()
    else:
        subset_subjects = splits.get(f"{split_name}_subjects")
        subset = df[df[SUBJECT_COLUMN].isin(subset_subjects)].copy()
    windows = create_windows(
        subset,
        config.window_seconds,
        config.window_overlap_seconds,
        config.sample_rate_hz,
        feature_stats=config.features.get("stats"),
    )
    feature_df = windows.drop(columns=[LABEL_COLUMN, SUBJECT_COLUMN])
    feature_df = ensure_feature_order(feature_df, feature_cols)
    y_true = windows[LABEL_COLUMN]
    preds = model.predict(feature_df)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    cm = confusion_matrix(y_true, preds)
    return {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist()}


def evaluate_log(log_path: pathlib.Path, model, feature_cols, config, subject_id: int) -> dict:
    windows = prepare_features_from_log(log_path, config, subject_id)
    feature_df = windows.drop(columns=[LABEL_COLUMN, SUBJECT_COLUMN])
    feature_df = ensure_feature_order(feature_df, feature_cols)
    preds = model.predict(feature_df)
    proba = model.predict_proba(feature_df)
    metrics = {}
    if windows[LABEL_COLUMN].nunique() > 1 or windows[LABEL_COLUMN].unique()[0] != -1:
        y_true = windows[LABEL_COLUMN]
        metrics["accuracy"] = accuracy_score(y_true, preds)
        metrics["macro_f1"] = f1_score(y_true, preds, average="macro")
        metrics["confusion_matrix"] = confusion_matrix(y_true, preds).tolist()
    return {"predictions": preds.tolist(), "proba": proba.tolist(), "metrics": metrics}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model, feature_cols, model_info = load_artifacts(config)

    if args.log:
        result = evaluate_log(pathlib.Path(args.log), model, feature_cols, config, args.subject_id)
        print(result)
        return

    df = load_dataset(config)
    # Remove demo subjects from main splits
    remaining, _ = filter_demo_subjects(df, config.excluded_subjects_demo)
    metrics = evaluate_split(remaining, model, feature_cols, config, args.split, model_info["splits"])
    print(metrics)


if __name__ == "__main__":
    main()
