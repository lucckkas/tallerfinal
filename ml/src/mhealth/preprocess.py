from __future__ import annotations

import collections
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config
from .constants import ACTIVITY_MAP, LABEL_COLUMN, SENSOR_COLUMNS, SUBJECT_COLUMN


def filter_demo_subjects(
    df: pd.DataFrame, excluded: Sequence[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = df[SUBJECT_COLUMN].isin(excluded)
    demo = df[mask].copy()
    remaining = df[~mask].copy()
    return remaining, demo


def filter_unlabeled_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove activity 0 (unlabeled/null activity) from dataset.
    This prevents extreme class imbalance issues during training.
    """
    return df[df[LABEL_COLUMN] != 0].copy()


def split_by_subject(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = sorted(df[SUBJECT_COLUMN].unique())
    rng = np.random.default_rng(config.random_seed)
    rng.shuffle(subjects)
    total = len(subjects)
    val_n = max(1, int(round(total * config.train_val_test_split.val_ratio)))
    test_n = max(1, int(round(total * config.train_val_test_split.test_ratio)))
    train_n = max(1, total - val_n - test_n)

    train_subj = subjects[:train_n]
    val_subj = subjects[train_n : train_n + val_n]
    test_subj = subjects[train_n + val_n : train_n + val_n + test_n]

    train_df = df[df[SUBJECT_COLUMN].isin(train_subj)].copy()
    val_df = df[df[SUBJECT_COLUMN].isin(val_subj)].copy()
    test_df = df[df[SUBJECT_COLUMN].isin(test_subj)].copy()

    return train_df, val_df, test_df


def _window_indices(
    n_samples: int, window_size: int, step_size: int
) -> List[Tuple[int, int]]:
    indices = []
    start = 0
    while start + window_size <= n_samples:
        end = start + window_size
        indices.append((start, end))
        start += step_size
    return indices


def create_windows(
    df: pd.DataFrame,
    window_seconds: float,
    overlap_seconds: float,
    sample_rate_hz: int,
    feature_stats: Sequence[str] | None = None,
) -> pd.DataFrame:
    window_size = int(window_seconds * sample_rate_hz)
    overlap = int(overlap_seconds * sample_rate_hz)
    step = max(1, window_size - overlap)
    rows: List[dict] = []

    for subject_id, group in df.groupby(df[SUBJECT_COLUMN]):
        group = group.sort_values("timestamp")
        idxs = _window_indices(len(group), window_size, step)
        for start, end in idxs:
            window = group.iloc[start:end]
            label_mode = window[LABEL_COLUMN].mode()
            label = int(label_mode.iloc[0]) if not label_mode.empty else None
            feature_row = extract_features(
                window[SENSOR_COLUMNS], feature_stats=feature_stats
            )
            feature_row[LABEL_COLUMN] = label
            feature_row[SUBJECT_COLUMN] = subject_id
            rows.append(feature_row)

    return pd.DataFrame(rows)


def extract_features(
    window_df: pd.DataFrame, feature_stats: Sequence[str] | None = None
) -> Dict[str, float]:
    stats = feature_stats or ["mean", "std", "min", "max", "median", "mad", "energy"]
    features: Dict[str, float] = {}
    for col in window_df.columns:
        values = window_df[col].values
        if "mean" in stats:
            features[f"{col}__mean"] = float(np.mean(values))
        if "std" in stats:
            features[f"{col}__std"] = float(np.std(values))
        if "min" in stats:
            features[f"{col}__min"] = float(np.min(values))
        if "max" in stats:
            features[f"{col}__max"] = float(np.max(values))
        if "median" in stats:
            features[f"{col}__median"] = float(np.median(values))
        if "mad" in stats:
            mad = float(np.median(np.abs(values - np.median(values))))
            features[f"{col}__mad"] = mad
        if "energy" in stats:
            energy = float(np.sum(values**2) / len(values))
            features[f"{col}__energy"] = energy
    return features


def build_feature_matrix(
    df_windows: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        c for c in df_windows.columns if c not in (LABEL_COLUMN, SUBJECT_COLUMN)
    ]
    X = df_windows[feature_cols].copy()
    y = df_windows[LABEL_COLUMN].astype(int).copy()
    return X, y


def fit_scaler(train_features: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def transform_features(scaler: StandardScaler, features: pd.DataFrame) -> np.ndarray:
    return scaler.transform(features)


def activity_distribution(preds: Sequence[int]) -> Dict[str, int]:
    counter = collections.Counter(preds)
    return {ACTIVITY_MAP.get(k, str(k)): int(v) for k, v in counter.items()}
