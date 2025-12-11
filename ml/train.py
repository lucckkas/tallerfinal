from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from mhealth.config import load_config
from mhealth.data import load_dataset, load_demo_subjects
from mhealth.modeling import save_artifacts, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MHealth HAR model.")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("CARGANDO DATOS CON EXCLUSIÓN DE SUJETOS DEMO")
    print("=" * 60)

    # Cargar dataset SIN sujetos demo (9, 10)
    print("\nCargando dataset de entrenamiento (excluyendo sujetos demo)...")
    df = load_dataset(config, exclude_demo=True)

    # Cargar sujetos demo por separado (solo para evaluación)
    print("\nCargando sujetos demo para evaluación...")
    demo_df = load_demo_subjects(config)

    print("\n" + "=" * 60)
    print("ENTRENANDO MODELO")
    print("=" * 60)
    artifacts = train_model(df, config, demo_df=demo_df)

    print("\nSaving artifacts...")
    save_artifacts(artifacts, config)

    print("\n" + "=" * 60)
    print("MÉTRICAS FINALES")
    print("=" * 60)
    for split, metrics in artifacts["metrics"].items():
        if metrics["accuracy"] is not None:
            print(
                f"{split}: acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
