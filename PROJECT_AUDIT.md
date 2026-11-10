# Project audit (2026-11-10)

## Repository state at start

- The session's working folder was empty. There was no existing code, data or git history to preserve.
- A git repository was initialised on `main`. Auto-AVSR was added as a submodule at `third_party/auto_avsr`, pinned to commit `182b628` (2025-01-08).
- No git remote exists yet: no GitHub CLI is installed and no remote URL was supplied.

## Environment A: development machine (audited)

| | |
|---|---|
| Machine | Apple M4 (8 cores, Metal 4), 16 GB RAM |
| OS | macOS 26.2 (build 25C56), arm64 |
| GPU / CUDA | **No NVIDIA GPU, no CUDA.** PyTorch MPS is available but unusable for decoding (see below) |
| Disk | 127 GB free of 228 GB |
| System Python | 3.9.6 (too old); **Python 3.11.18 installed via uv** in `.venv` |
| Homebrew | Present but owned by another user: `brew install` failed with permission denied. Not worked around with sudo |
| Tools | git, Node 26.12.1 / npm 11.20.0. No ffmpeg binary (PyAV/imageio-ffmpeg bundle their own) |

Resolved Python packages:
- torch 2.15.0, torchvision 0.29.2
- mediapipe 0.10.14
- opencv-contrib-python 5.0.0.93
- numpy 2.2.6
- scikit-image 0.26.0
- sentencepiece 0.2.3
- jiwer 4.0.0
- fastapi 0.162.2
- pydantic 2.14.3
- uvicorn 0.54.0
- pytest 9.2.0

Resolved frontend packages: React 19.3, Vite 8.3, Tailwind 4.3, Recharts 3.10, lucide-react 1.70 and TypeScript 7.0.

## Environment B: training machine (not audited; specification from the brief)

RTX 5070 with 12 GB VRAM and 32 GB RAM. The OS and CUDA version are unknown. Setup and
verification steps are in `docs/GPU_SETUP_RTX5070.md`.

## ECC plugin

Not installed at the first audit. Installed during the session as `ecc` 2.2.3.
How it was used is in `ECC_USAGE.md`.

## Compatibility findings (all verified on the Mac)

| Finding | Consequence |
|---|---|
| mediapipe 1.2.2 (latest) has no `mp.solutions` | pinned `mediapipe==0.10.14`, the version the upstream detector expects |
| PyAV and OpenCV each bundle libavdevice; macOS warns about duplicate classes | PyAV removed; decoding uses OpenCV only |
| mediapipe pulls in `opencv-contrib-python`; installing `opencv-python-headless` as well overlays `cv2` | only `opencv-contrib-python` is kept |
| Upstream beam search on MPS: device mismatch in `ctc_prefix_score.py` | `auto` picks CUDA, else CPU; `mps` is rejected with an explanation |
| Model builds unchanged on torch 2.15; total 250,383,410 params | matches the model zoo's "250M" |
| Upstream LRS3 preprocessing script always raises (`!= "cuda" or "mps"`) | own `prepare_dataset.py` reusing upstream detector and crop code |
| Upstream `train.py` validates on LRS3 **test** by default | own training loop with a separate validation split |
| Tutorial checkpoint URLs (`doc.ic.ac.uk`) return 404 | Google Drive links from the README are documented instead |
| Official LRS3 page returns 404 | GRID chosen as the primary dataset (DATASET_GUIDE.md) |

## Main risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| PyTorch CUDA build for sm_120 not matching pinned versions | medium | `gpu_check.py`; guide allows the newest CUDA ≥ 12.8 pair |
| VRAM too small for the chosen strategy | low-medium | frame-budget batching, bf16, `decoder_only` fallback, measured probe before training |
| GRID archive layout differs from the expected one | medium | download script prints per-speaker counts; prepare fails loudly with 0 clips |
| Many GRID clips fail MediaPipe detection | low | failures listed and counted; excluded identically for A and B |
| Fine-tuning gives no improvement | possible | still a valid result; report honestly with curves |
| Google Drive checkpoint download blocked or quota-limited | medium | manual browser download; MD5 verification script |
| Results misread as LRS3-comparable | medium | GRID vs LRS3 caveats in the README, Methodology page and EXPERIMENT_DESIGN |
