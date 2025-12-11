from __future__ import annotations

import json
import os
import pathlib
import random
from typing import Any

import numpy as np


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    # Use Generator for numpy 2.x compatibility
    # np.random.seed(seed)  # Deprecated in numpy 2.x
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    path = pathlib.Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
