"""Verify the CUDA environment on the RTX 5070 machine.

    python scripts/gpu_check.py

Checks: CUDA visible to PyTorch, GPU name and VRAM, compute capability vs. the
architectures this PyTorch build was compiled for (RTX 50-series = sm_120 needs a
CUDA 12.8+ build), bf16 support, and real GPU kernels (matmul + bf16 Conv3d).
Exits non-zero on any failure. Never falls back to CPU.
"""

from __future__ import annotations

import json
import sys

import torch


def main() -> int:
    report: dict[str, object] = {"torch": torch.__version__, "cuda_build": torch.version.cuda}
    if not torch.cuda.is_available():
        report["error"] = "torch.cuda.is_available() is False (driver missing, CPU-only wheel, or no GPU)."
        print(json.dumps(report, indent=2))
        return 1
    props = torch.cuda.get_device_properties(0)
    cap = f"sm_{props.major}{props.minor}"
    arch_list = torch.cuda.get_arch_list()
    report.update(
        gpu=props.name,
        vram_gb=round(props.total_memory / 1024**3, 2),
        compute_capability=cap,
        compiled_arches=arch_list,
        native_arch_in_build=cap in arch_list,  # informational; the kernel test below decides
        bf16_supported=torch.cuda.is_bf16_supported(),
        cudnn=torch.backends.cudnn.version(),
    )
    try:
        a = torch.randn(2048, 2048, device="cuda")
        (a @ a).sum().item()
        conv = torch.nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            conv(torch.randn(1, 1, 50, 88, 88, device="cuda")).sum().item()
        report["kernel_test"] = "ok"
    except RuntimeError as exc:  # e.g. "no kernel image is available for execution on the device"
        report["kernel_test"] = f"FAILED: {exc}"
    print(json.dumps(report, indent=2))
    if report["kernel_test"] != "ok":
        print("\nGPU check FAILED. Install a PyTorch build for CUDA 12.8+ (see docs/GPU_SETUP_RTX5070.md).")
        return 1
    print("\nGPU check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
