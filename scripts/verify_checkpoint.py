"""Verify a downloaded Auto-AVSR checkpoint: MD5 prefix (from the model zoo table)
and a strict load into the verified VSR architecture.

    python scripts/verify_checkpoint.py checkpoints/vsr_trlrs3vox2_base.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ml.inference.model import CheckpointError, VSRModel, count_parameters  # noqa: E402
from ml.utils.env import md5_file  # noqa: E402

# MD5 prefixes as published in the Auto-AVSR README model zoo (commit 182b628).
ZOO_MD5_PREFIX = {
    "vsr_trlrs3_23h_base.pth": "fc8db",
    "vsr_trlrs3_base.pth": "c00a7",
    "vsr_trlrs3vox2_base.pth": "774a6",
    "vsr_trlrs2lrs3vox2avsp_base.pth": "49f77",
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Not found: {path}")
        return 1
    digest = md5_file(path)
    expected = ZOO_MD5_PREFIX.get(path.name)
    print(f"md5 {digest}")
    if expected is None:
        print(f"(no published MD5 for '{path.name}'; skipping prefix check)")
    elif not digest.startswith(expected):
        print(f"MD5 MISMATCH: expected prefix {expected}. Re-download the file.")
        return 1
    else:
        print(f"MD5 prefix matches model zoo ({expected})")
    try:
        model = VSRModel(path, torch.device("cpu"))
    except CheckpointError as exc:
        print(f"LOAD FAILED: {exc}")
        return 1
    print("strict load OK; parameters:", count_parameters(model.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
