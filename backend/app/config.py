from __future__ import annotations

import os
import pathlib

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    model_artifact: str = Field(default="ml/artifacts/model.joblib", alias="MODEL_ARTIFACT")
    feature_metadata: str = Field(default="ml/artifacts/features.json", alias="FEATURE_METADATA")
    metrics_artifact: str = Field(default="ml/artifacts/metrics.json", alias="METRICS_ARTIFACT")
    model_info_artifact: str = Field(default="ml/artifacts/model_info.json", alias="MODEL_INFO_ARTIFACT")
    allowed_origins: str = Field(default="http://localhost:5173", alias="ALLOWED_ORIGINS")
    config_yaml: str = Field(default="config/config.yaml", alias="CONFIG_YAML")

    class Config:
        env_file = ".env"
        extra = "ignore"


def load_settings() -> Settings:
    return Settings()
