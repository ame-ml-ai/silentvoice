# GPU environment setup: RTX 5070 desktop (Environment B)

This guide configures the training machine. Nothing here has been run yet: the project
was developed on an Apple M4 Mac without CUDA. Every step ends with a check, so problems
show up immediately rather than mid-training.

## 0. What the hardware requires (facts to verify on the machine)

| Item | Requirement | Why |
|---|---|---|
| GPU | GeForce RTX 5070, 12 GB, Blackwell architecture, compute capability **sm_120** | Must be in the PyTorch build's arch list |
| NVIDIA driver | A driver that supports **CUDA 12.8 or newer** (R570 series or later) | sm_120 kernels first shipped with CUDA 12.8 |
| PyTorch | A **CUDA 12.8+** wheel. Builds for CUDA 12.6 or older have no sm_120 kernels and fail with *"no kernel image is available for execution on the device"* | |
| Python | **3.11.x** (the version the project is tested with) | mediapipe 0.10.14 wheels exist for 3.11 |
| RAM / disk | 32 GB RAM; ~30 GB free for GRID (13.6 GB zips + extracted + crops) and checkpoints | |
| OS | Linux (Ubuntu 22.04/24.04) or Windows 11. WSL2 on Windows works with the Windows NVIDIA driver | |

Check the driver first:

```bash
nvidia-smi
```

The header must show the RTX 5070 and a "CUDA Version" of 12.8 or higher. That number
is the highest CUDA runtime the driver supports, not what is installed. No separate CUDA
toolkit is needed: PyTorch wheels bundle their own CUDA runtime.

## 1. Clone with the Auto-AVSR submodule

```bash
git clone --recurse-submodules <your-repo-url> silentvoice
cd silentvoice
git -C third_party/auto_avsr rev-parse HEAD   # expect 182b62837773ab01052d4ac21ef1d2203ea7d267
```

## 2. Python environment

Linux/WSL:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell: `py -3.11 -m venv .venv`, then `.venv\Scripts\Activate.ps1`.

## 3. Install PyTorch for CUDA, then everything else

Pick the wheel on <https://pytorch.org/get-started/locally/>: *Stable · your OS · Pip ·
Python · CUDA 12.8 (or the newest CUDA listed)*. The Mac uses `torch==2.15.0` /
`torchvision==0.29.2`; use the same versions if the selector offers them with CUDA ≥ 12.8.
For example (verify the index URL on the site; do not guess):

```bash
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu128
```

If that exact pair is not published for CUDA ≥ 12.8, install the newest stable pair the
selector offers for CUDA ≥ 12.8, run step 4, and record the versions in `PROGRESS.md`.
The code uses only stable PyTorch APIs (`torch.amp`, `torch.load(weights_only=True)`,
standard modules).

Then:

```bash
pip install -r requirements/gpu.txt
```

Do **not** install `opencv-python` or `opencv-python-headless`: mediapipe already pulls in
`opencv-contrib-python`, and two OpenCV distributions in one environment conflict.

## 4. Verify CUDA (never skip)

```bash
python scripts/gpu_check.py
```

Expected: `"gpu": "NVIDIA GeForce RTX 5070"`, `vram_gb` ≈ 12, `compute_capability: sm_120`,
`bf16_supported: true`, `kernel_test: "ok"`, exit code 0. If the kernel test fails with
"no kernel image", the wheel is not a CUDA 12.8+ build: reinstall step 3.

Then run the full test suite. It is CPU-only and must pass on this machine too:

```bash
python -m pytest -q
```

## 5. Download and verify the pretrained checkpoint

The Auto-AVSR model zoo is hosted on Google Drive (the old `doc.ic.ac.uk` mirror returns
404 as of 2026-11-10). Use the checkpoint named in the configs:

| File | Training data | Published LRS3 WER | Drive id |
|---|---|---|---|
| `vsr_trlrs3vox2_base.pth` | LRS3 + VoxCeleb2, 1,759 h | 24.6% | `1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i` |
| `vsr_trlrs2lrs3vox2avsp_base.pth` (optional, Experiment C) | 3,291 h | 20.3% | `1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_` |

```bash
mkdir -p checkpoints
pip install gdown            # or download in a browser from the README links
gdown 1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i -O checkpoints/vsr_trlrs3vox2_base.pth
python scripts/verify_checkpoint.py checkpoints/vsr_trlrs3vox2_base.pth
```

`verify_checkpoint.py` checks the MD5 prefix from the model zoo (`774a6…`) and does a
strict load into the architecture: 250,383,410 parameters.

## 6. GPU inference smoke test

```bash
SILENTVOICE_CHECKPOINT_PATH=checkpoints/vsr_trlrs3vox2_base.pth SILENTVOICE_DEVICE=cuda \
  uvicorn backend.app.main:app --port 8000
```

In another terminal, run `npm --prefix frontend run dev` and open
<http://localhost:5173/recognize>. The header pill must say **Model ready**. Upload a short,
frontal clip of someone speaking: you get a transcript. Health check:
`curl localhost:8000/api/health` shows `"device":"cuda"`.

## 7. Memory probe before any training

```bash
python scripts/gpu_memory_probe.py --strategy encoder_top_k --blocks 4 --amp bf16 \
    --frames 150 300 600 900 1200
```

It writes `experiments/hardware/memory_probe.json` with the measured peak VRAM per frame
budget. Choose the largest `frames_per_batch` whose peak stays below ~10 GB. That leaves
headroom: the probe does not include validation beam search or allocator fragmentation.
Set it as `train.max_frames_per_batch` (and `max_batch_size = frames / 75` for GRID) in
`ml/configs/finetune_grid.yaml`.

Arithmetic estimates (not measurements) for orientation:

| | Weights (fp32) | Gradients | AdamW state | Static total |
|---|---|---|---|---|
| `decoder_only` (68.4M trainable) | 1.00 GB | 0.27 GB | 0.55 GB | ~1.8 GB |
| `encoder_top_k`, k=4 (125.2M) | 1.00 GB | 0.50 GB | 1.00 GB | ~2.5 GB |
| `full` (250.4M) | 1.00 GB | 1.00 GB | 2.00 GB | ~4.0 GB |

Activations come on top and scale with frames per batch; only the probe measures them.

## 8. Command reference

| Purpose | Command |
|---|---|
| GPU verification | `python scripts/gpu_check.py` |
| Memory probe | `python scripts/gpu_memory_probe.py ...` |
| Baseline (Experiment A) | `python -m ml.evaluation.evaluate --config ml/configs/baseline_grid.yaml` |
| Short training test | `python -m ml.training.finetune --config ml/configs/finetune_grid.yaml --max-steps 20` |
| Fine-tuning (Experiment B) | `python -m ml.training.finetune --config ml/configs/finetune_grid.yaml` |
| Resume | `... finetune --config ml/configs/finetune_grid.yaml --resume experiments/finetuning/checkpoints/last.pth` |
| Final evaluation | `python -m ml.evaluation.evaluate --config ml/configs/finetuned_grid_eval.yaml` |
| GPU inference API | `SILENTVOICE_CHECKPOINT_PATH=... SILENTVOICE_DEVICE=cuda uvicorn backend.app.main:app` |

Training refuses to start without CUDA (exit code 3). There is no silent CPU fallback.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.is_available()` is False | CPU-only wheel or driver too old → step 3 with a CUDA index URL; update driver |
| `no kernel image is available` | Wheel built for CUDA < 12.8 → reinstall a CUDA 12.8+ build |
| `CUDA out of memory` during training | Lower `max_frames_per_batch`; keep `amp_dtype: bf16`; use `decoder_only` |
| `module 'mediapipe' has no attribute 'solutions'` | mediapipe ≥ 1.0 installed → `pip install mediapipe==0.10.14` |
| Non-finite gradient warnings | Logged and skipped; more than 50 aborts the run → lower `optim.lr` |
