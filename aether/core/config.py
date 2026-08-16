from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("AETHER_ENV", "development")
    host: str = os.getenv("AETHER_HOST", "0.0.0.0")
    port: int = int(os.getenv("AETHER_PORT", "8000"))
    data_dir: Path = Path(os.getenv("AETHER_DATA_DIR", "./data"))
    model_cache: Path = Path(os.getenv("AETHER_MODEL_CACHE", "./model-cache"))
    cors_origins: tuple[str, ...] = tuple(x.strip() for x in os.getenv("AETHER_CORS_ORIGINS", "http://localhost:5173").split(",") if x.strip())
    max_upload_mb: int = int(os.getenv("AETHER_MAX_UPLOAD_MB", "100"))
    max_model_params_millions: int = int(os.getenv("AETHER_MAX_MODEL_PARAMS_MILLIONS", "1500"))
    max_training_steps: int = int(os.getenv("AETHER_MAX_TRAINING_STEPS", "5000"))
    max_sequence_length: int = int(os.getenv("AETHER_MAX_SEQUENCE_LENGTH", "1024"))
    max_active_jobs_per_user: int = int(os.getenv("AETHER_MAX_ACTIVE_JOBS_PER_USER", "1"))
    max_jobs_per_day: int = int(os.getenv("AETHER_MAX_JOBS_PER_DAY", "3"))
    jwt_secret: str = os.getenv("AETHER_JWT_SECRET", "change-me-in-production")
    jwt_exp_minutes: int = int(os.getenv("AETHER_JWT_EXP_MINUTES", "1440"))
    hf_token: str | None = os.getenv("HF_TOKEN") or None


settings = Settings()
for directory in (settings.data_dir, settings.model_cache):
    directory.mkdir(parents=True, exist_ok=True)
