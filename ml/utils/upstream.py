"""Access to the pinned Auto-AVSR checkout (git submodule at third_party/auto_avsr).

SilentVoice reuses the upstream model definition, beam search, tokenizer files and
mouth-cropping code unchanged, so that checkpoints from the Auto-AVSR model zoo load
exactly and preprocessing matches what the checkpoints were trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTO_AVSR_DIR = REPO_ROOT / "third_party" / "auto_avsr"
# Commit the submodule is pinned to (verified 2026-11-10).
AUTO_AVSR_COMMIT = "182b62837773ab01052d4ac21ef1d2203ea7d267"

SPM_MODEL_PATH = AUTO_AVSR_DIR / "spm" / "unigram" / "unigram5000.model"
SPM_UNITS_PATH = AUTO_AVSR_DIR / "spm" / "unigram" / "unigram5000_units.txt"
MEDIAPIPE_DIR = AUTO_AVSR_DIR / "preparation" / "detectors" / "mediapipe"


class UpstreamMissingError(RuntimeError):
    """Raised when the Auto-AVSR submodule has not been checked out."""


def ensure_auto_avsr_on_path() -> Path:
    """Put the Auto-AVSR checkout on ``sys.path`` so ``espnet.*`` imports resolve."""
    marker = AUTO_AVSR_DIR / "espnet" / "nets" / "pytorch_backend" / "e2e_asr_conformer.py"
    if not marker.is_file():
        raise UpstreamMissingError(
            f"Auto-AVSR not found at {AUTO_AVSR_DIR}. "
            "Run: git submodule update --init --recursive"
        )
    path = str(AUTO_AVSR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return AUTO_AVSR_DIR
