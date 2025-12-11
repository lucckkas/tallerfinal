from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class ModelConfig:
    type: str
    n_estimators: int
    max_depth: Optional[int]
    class_weight: Optional[str]


@dataclass
class Config:
    version: str
    random_seed: int
    sample_rate_hz: int
    window_seconds: float
    window_overlap_seconds: float
    excluded_subjects_demo: List[int]
    train_val_test_split: SplitConfig
    features: dict
    model: ModelConfig
    artifacts: dict


def load_config(path: str | pathlib.Path = "config/config.yaml") -> Config:
    config_path = pathlib.Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        version=str(raw["version"]),
        random_seed=int(raw["random_seed"]),
        sample_rate_hz=int(raw["sample_rate_hz"]),
        window_seconds=float(raw["window_seconds"]),
        window_overlap_seconds=float(raw["window_overlap_seconds"]),
        excluded_subjects_demo=list(raw["excluded_subjects_demo"]),
        train_val_test_split=SplitConfig(
            train_ratio=float(raw["train_val_test_split"]["train_ratio"]),
            val_ratio=float(raw["train_val_test_split"]["val_ratio"]),
            test_ratio=float(raw["train_val_test_split"]["test_ratio"]),
        ),
        features=raw["features"],
        model=ModelConfig(
            type=str(raw["model"]["type"]),
            n_estimators=int(raw["model"]["n_estimators"]),
            max_depth=(
                None if raw["model"]["max_depth"] is None else int(raw["model"]["max_depth"])
            ),
            class_weight=raw["model"]["class_weight"],
        ),
        artifacts=raw["artifacts"],
    )
