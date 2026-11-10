"""Explicit device selection.

Rules (from the project spec):
* Training never silently falls back from CUDA to CPU.
* Inference may use ``auto``, which picks CUDA > CPU and reports what it chose.

MPS (Apple GPU) is deliberately not chosen by ``auto``: Auto-AVSR's CTC prefix scorer
(espnet/nets/ctc_prefix_score.py) allocates index tensors on "cuda" or "cpu" only, so
beam search fails on MPS with a device-mismatch error (verified on torch 2.15).
"""

from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "mps", "cuda"]


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested accelerator is not available."""


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def resolve_device(requested: DeviceName | str = "auto") -> torch.device:
    """Resolve a device name, raising instead of silently downgrading."""
    requested = str(requested).lower()
    if requested == "auto":
        if cuda_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not cuda_available():
            raise DeviceUnavailableError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "GPU work must run on the RTX 5070 machine (see docs/GPU_SETUP_RTX5070.md)."
            )
        return torch.device(requested)
    if requested == "mps":
        raise DeviceUnavailableError(
            "MPS is not supported: the upstream CTC prefix scorer only handles CUDA and CPU. "
            "Use 'cpu' on the Mac."
        )
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{requested}'. Use auto, cpu, mps or cuda.")


def describe_device(device: torch.device) -> dict[str, object]:
    """Hardware description recorded alongside every measurement."""
    info: dict[str, object] = {"type": device.type}
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            name=props.name,
            total_memory_gb=round(props.total_memory / 1024**3, 2),
            capability=f"{props.major}.{props.minor}",
            cuda_runtime=torch.version.cuda,
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU (MPS)"
    else:
        info["name"] = "CPU"
    return info
