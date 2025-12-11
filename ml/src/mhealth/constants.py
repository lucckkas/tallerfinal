from __future__ import annotations

import pathlib

DATASET_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"

RAW_DIR = pathlib.Path("ml/data/raw")
PROCESSED_DIR = pathlib.Path("ml/data/processed")

# Column names from the official MHEALTH dataset description.
SENSOR_COLUMNS = [
    # Chest Accelerometer
    "acc_chest_x",
    "acc_chest_y",
    "acc_chest_z",
    # Two ECG channels
    "ecg_1",
    "ecg_2",
    # Left ankle accelerometer
    "acc_ankle_x",
    "acc_ankle_y",
    "acc_ankle_z",
    # Left ankle gyroscope
    "gyro_ankle_x",
    "gyro_ankle_y",
    "gyro_ankle_z",
    # Left ankle magnetometer
    "mag_ankle_x",
    "mag_ankle_y",
    "mag_ankle_z",
    # Right arm accelerometer
    "acc_arm_x",
    "acc_arm_y",
    "acc_arm_z",
    # Right arm gyroscope
    "gyro_arm_x",
    "gyro_arm_y",
    "gyro_arm_z",
    # Right arm magnetometer
    "mag_arm_x",
    "mag_arm_y",
    "mag_arm_z",
]

LABEL_COLUMN = "activity"
SUBJECT_COLUMN = "subject"
TIMESTAMP_COLUMN = "timestamp"

ACTIVITY_MAP = {
    0: "Sin clasificar",
    1: "De pie",
    2: "Sentado",
    3: "Acostado",
    4: "Caminando",
    5: "Subiendo escaleras",
    6: "Flexión de cintura",
    7: "Elevación frontal de brazos",
    8: "Flexión de rodillas",
    9: "Ciclismo",
    10: "Trote",
    11: "Corriendo",
    12: "Saltando",
}

ALL_COLUMNS = SENSOR_COLUMNS + [LABEL_COLUMN]
