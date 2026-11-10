"""Reproducibility helpers: seeding, environment capture, file hashing."""

from __future__ import annotations

import hashlib
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ml.utils.upstream import AUTO_AVSR_COMMIT, REPO_ROOT


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def md5_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 - used for checkpoint identity, matching the model zoo
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "auto_avsr_commit": AUTO_AVSR_COMMIT,
        "silentvoice_commit": git_revision(),
    }
    for module in ("cv2", "mediapipe", "jiwer", "sentencepiece", "torchvision"):
        try:
            versions[module] = __import__(module).__version__
        except (ImportError, AttributeError):
            versions[module] = None
    return versions


def environment_record() -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "software": software_versions(),
    }
