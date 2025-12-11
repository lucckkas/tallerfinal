from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from mhealth.config import load_config
from mhealth.inference import load_artifacts, predict_windows, prepare_features_from_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single .log file.")
    parser.add_argument("log_path", help="Path to mHealth .log file.")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML.")
    parser.add_argument("--subject-id", type=int, default=0, help="Subject id placeholder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model, feature_cols, _ = load_artifacts(config)
    windows = prepare_features_from_log(pathlib.Path(args.log_path), config, args.subject_id)
    feature_df = windows.drop(columns=["activity", "subject"])
    result = predict_windows(model, feature_df, feature_cols)
    print(result)


if __name__ == "__main__":
    main()
