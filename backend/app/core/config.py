"""Runtime settings, read from environment variables prefixed ``SILENTVOICE_`` or .env."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.utils.upstream import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SILENTVOICE_", env_file=REPO_ROOT / ".env", extra="ignore")

    app_name: str = "SilentVoice API"
    version: str = "0.1.0"

    # Model. Leave checkpoint_path unset to run without a model (upload, validation and
    # mouth-region preprocessing still work; recognition returns 503 model_unavailable).
    checkpoint_path: Path | None = None
    device: str = "auto"  # auto -> cuda if available, else cpu (MPS unsupported, see ml/utils/device.py)
    beam_size: int = 40
    load_model_on_startup: bool = True

    # Uploads
    max_upload_mb: int = 50
    max_duration_s: float = 30.0
    min_duration_s: float = 0.4
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".mpg", ".m4v")
    upload_dir: Path = Path(tempfile.gettempdir()) / "silentvoice_uploads"

    # Jobs
    max_jobs_retained: int = 100

    # Research results
    experiments_dir: Path = REPO_ROOT / "experiments"

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
