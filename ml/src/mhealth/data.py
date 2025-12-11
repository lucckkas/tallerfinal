from __future__ import annotations

import pathlib
import re
import zipfile
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import requests

from .config import Config
from .constants import (
    ACTIVITY_MAP,
    ALL_COLUMNS,
    DATASET_URL,
    LABEL_COLUMN,
    RAW_DIR,
    SUBJECT_COLUMN,
)
from .utils import ensure_dir


def fetch_dataset(url: str = None) -> pathlib.Path:
    """
    Download dataset zip to RAW_DIR if not present.
    """
    ensure_dir(RAW_DIR)
    zip_path = RAW_DIR / "mhealth_dataset.zip"
    if zip_path.exists():
        return zip_path
    dataset_url = url or DATASET_URL
    resp = requests.get(dataset_url, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    return zip_path


def extract_dataset(zip_path: pathlib.Path) -> pathlib.Path:
    target_dir = RAW_DIR / "mhealth_dataset"
    if target_dir.exists():
        return target_dir
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    # Some zips nest the folder; try to find the extracted root.
    possible = list(RAW_DIR.glob("**/mHealth_subject1.log"))
    if possible:
        target_dir = possible[0].parent
    return target_dir


def iter_subject_files(dataset_dir: pathlib.Path) -> Iterable[Tuple[int, pathlib.Path]]:
    for path in sorted(dataset_dir.glob("mHealth_subject*.log")):
        match = re.search(r"subject(\d+)", path.name)
        if not match:
            continue
        yield int(match.group(1)), path


def load_subject_log(
    path: pathlib.Path, subject_id: int, sample_rate_hz: int
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
    )
    # Add synthetic timestamp based on sample rate to keep ordering.
    df["timestamp"] = np.arange(len(df)) / float(sample_rate_hz)
    df[SUBJECT_COLUMN] = subject_id
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def load_dataset(config: Config, exclude_demo: bool = True) -> pd.DataFrame:
    """
    Load MHealth dataset.

    Args:
        config: Configuration object
        exclude_demo: If True, completely excludes demo subjects (9, 10) from loading.
                     This prevents any possibility of data leakage.
    """
    zip_path = fetch_dataset()
    dataset_dir = extract_dataset(zip_path)
    frames = []
    excluded_subjects = set(config.excluded_subjects_demo) if exclude_demo else set()

    for subject_id, path in iter_subject_files(dataset_dir):
        if subject_id in excluded_subjects:
            print(
                f"[INFO] Excluyendo sujeto {subject_id} del dataset de entrenamiento (demo)"
            )
            continue
        frames.append(load_subject_log(path, subject_id, config.sample_rate_hz))

    data = pd.concat(frames, ignore_index=True)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].map(lambda x: int(x))
    return data


def load_demo_subjects(config: Config) -> pd.DataFrame:
    """
    Load ONLY the demo subjects (9, 10) for evaluation purposes.
    These subjects are never used in training.
    """
    zip_path = fetch_dataset()
    dataset_dir = extract_dataset(zip_path)
    frames = []
    demo_subjects = set(config.excluded_subjects_demo)

    for subject_id, path in iter_subject_files(dataset_dir):
        if subject_id in demo_subjects:
            print(f"[INFO] Cargando sujeto demo {subject_id} para evaluación")
            frames.append(load_subject_log(path, subject_id, config.sample_rate_hz))

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].map(lambda x: int(x))
    return data


def activity_name(label: int) -> str:
    return ACTIVITY_MAP.get(label, f"unknown_{label}")
