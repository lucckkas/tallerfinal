"""Analyze a specific subject's data to understand prediction issues."""

import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from mhealth.config import load_config
from mhealth.data import load_dataset
from mhealth.constants import LABEL_COLUMN, SUBJECT_COLUMN


def analyze_subject(subject_id: int):
    config = load_config("config/config.yaml")
    print(f"Loading dataset...")
    df = load_dataset(config)

    # Filter for specific subject
    subj_df = df[df[SUBJECT_COLUMN] == subject_id].copy()

    if len(subj_df) == 0:
        print(f"Subject {subject_id} not found in dataset!")
        return

    print(f"\n{'=' * 60}")
    print(f"Subject {subject_id} Analysis")
    print(f"{'=' * 60}")

    # Label distribution
    print(f"\nTotal samples: {len(subj_df)}")
    print(f"\nLabel distribution:")
    label_counts = subj_df[LABEL_COLUMN].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = (count / len(subj_df)) * 100
        print(f"  Activity {label}: {count:5d} samples ({pct:5.1f}%)")

    # Check for activity 0
    zero_count = (subj_df[LABEL_COLUMN] == 0).sum()
    zero_pct = (zero_count / len(subj_df)) * 100
    print(f"\n⚠️  Activity 0 (unlabeled): {zero_count} samples ({zero_pct:.1f}%)")

    if zero_pct > 50:
        print(f"   WARNING: Subject {subject_id} has >50% unlabeled data!")
        print(f"   This explains poor performance after filtering activity 0.")

    # Sensor statistics
    sensor_cols = [
        c
        for c in subj_df.columns
        if c not in [LABEL_COLUMN, SUBJECT_COLUMN, "timestamp"]
    ]
    print(f"\nSensor data quality:")
    nan_count = subj_df[sensor_cols].isnull().sum().sum()
    zero_rows = (subj_df[sensor_cols] == 0).all(axis=1).sum()
    print(f"  NaN values: {nan_count}")
    print(f"  All-zero rows: {zero_rows}")

    # Activity transitions
    transitions = (subj_df[LABEL_COLUMN].diff() != 0).sum()
    print(f"\nActivity transitions: {transitions}")

    return subj_df


if __name__ == "__main__":
    subject_id = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    analyze_subject(subject_id)
