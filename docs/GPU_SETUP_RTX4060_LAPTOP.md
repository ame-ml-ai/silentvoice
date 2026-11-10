# GPU phase on the RTX 4060 Laptop (Windows 11)

This replaces [CONTINUE_ON_RTX5070.md](../CONTINUE_ON_RTX5070.md) for the machine actually
available. The software stack is unchanged; only memory settings, Windows commands and
the run order differ. **Nothing below has been executed yet** (status: 2026-11-10).

## 0. The machine (measured on 2026-11-10)

| Item | Value | Consequence |
|---|---|---|
| GPU | GeForce RTX 4060 Laptop, **8 GB** VRAM, Ada, **sm_89**, bf16 supported | Any CUDA 12.x PyTorch build works (the 12.8+ requirement was only for sm_120) |
| CPU | Intel i7-13620H (10 cores / 16 threads) | `prepare_dataset --workers 6` |
| RAM | 16 GB | 4 DataLoader workers; Windows spawns a full Python+torch process per worker |
| Disk | ~78 GB free on C: | GRID needs ~30 GB (zips + extracted + crops) |
| OS | Windows 11, PowerShell | Commands below are PowerShell unless marked *Git Bash* |

**Open issue:** Device Manager lists the NVIDIA GPU with status `Unknown` and
`nvidia-smi` fails with *"insufficient permissions"*. Fix this first (step 1).

## 1. Make the NVIDIA GPU visible

1. Plug in the charger (laptop GPUs are throttled or powered down on battery).
2. Run `nvidia-smi` in an **administrator** PowerShell. If it now works, the earlier error was only the non-admin sandbox.
3. If not: Device Manager → Display adapters → *NVIDIA GeForce RTX 4060 Laptop GPU* → if disabled, *Enable device*; otherwise *Update driver*.
4. Still failing: install the latest Game Ready or Studio driver for *GeForce RTX 40 Series (Notebooks)* from nvidia.com (clean install), reboot.
5. Done when `nvidia-smi` shows the RTX 4060 and a "CUDA Version" of 12.x or higher.

## 2. Python 3.11 environment

Only Python 3.14 is installed; mediapipe 0.10.14 has no 3.14 wheels, so 3.11 is required.

```powershell
winget install -e --id Python.Python.3.11
cd C:\Users\temir\Projects\silentvoice
git submodule update --init --recursive
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. PyTorch with CUDA, then the rest

Pick the exact command on <https://pytorch.org/get-started/locally/> (Stable · Windows ·
Pip · Python · newest CUDA). The project was tested with `torch==2.15.0` /
`torchvision==0.29.2`; prefer that pair if it is offered, e.g.:

```powershell
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/gpu.txt
python scripts/gpu_check.py      # expect sm_89, bf16_supported true, kernel_test ok
python -m pytest -q              # CPU tests; the two slow smoke tests need ~2 GB RAM
```

If a different torch pair is installed, record it in `PROGRESS.md`.

## 4. Checkpoint

```powershell
pip install gdown
New-Item -ItemType Directory -Force checkpoints | Out-Null
gdown 1shcWXUK2iauRhW9NbwCc25FjU1CoMm8i -O checkpoints/vsr_trlrs3vox2_base.pth
python scripts/verify_checkpoint.py checkpoints/vsr_trlrs3vox2_base.pth
```

## 5. Dataset (GRID, ~13.6 GB download)

*Git Bash* (it ships `curl` and `unzip`):

```bash
bash scripts/download_grid.sh data/raw/grid s1 s5        # quick layout check first
bash scripts/download_grid.sh data/raw/grid              # then all 33 talkers
```

PowerShell:

```powershell
# 1-minute pipeline check
python -m ml.preprocessing.prepare_dataset grid --raw-dir data/raw/grid --out-dir data/processed/grid_smoke `
    --split-config ml/configs/splits/grid.yaml --speakers s1,s5 --limit 50 --workers 4
# full run (CPU-bound, roughly 1-3 h; safe to interrupt and re-run, it resumes)
python -m ml.preprocessing.prepare_dataset grid --raw-dir data/raw/grid --out-dir data/processed/grid `
    --split-config ml/configs/splits/grid.yaml --workers 6
```

Check `data/processed/grid/prep_report.json`: `split_report.ok` must be `true`.
After a successful run, `data/raw/grid/_zips` (13.6 GB) can be deleted to free space.

## 6. Memory probe (decides the batch size)

```powershell
python scripts/gpu_memory_probe.py --strategy encoder_top_k --blocks 4 --amp bf16 --frames 150 225 300 375 450 600
```

It prints a recommendation for peak ≤ 80 % of VRAM (~6.5 GB). `ml/configs/finetune_grid.yaml`
ships with a conservative **300 frames / 4 clips / 8 accumulation steps** (effective batch
32, same as the original plan). Update the three values only from the probe output, and
commit `experiments/hardware/memory_probe.json`.

If even 150 frames does not fit: set `freeze.strategy: decoder_only` (≈1.8 GB static
instead of ≈2.5 GB) and document the deviation in `EXPERIMENT_DESIGN.md`.

## 7. Experiments

Laptop preparation for long runs: charger connected, Windows power mode *Best
performance*, no sleep on AC (`powercfg /change standby-timeout-ac 0`), laptop on a hard
surface. Watch temperature with `nvidia-smi -l 5` in a second terminal.

```powershell
# A: baseline (no training)
python -m ml.evaluation.evaluate --config ml/configs/baseline_grid.yaml

# B: 20-step check (memory, speed, loss going down), then the real run
python -m ml.training.finetune --config ml/configs/finetune_grid.yaml --max-steps 20
Get-Content experiments/finetuning/train_log.jsonl -Tail 3
Remove-Item -Recurse -Force experiments/finetuning
python -m ml.training.finetune --config ml/configs/finetune_grid.yaml
# after an interruption:
python -m ml.training.finetune --config ml/configs/finetune_grid.yaml --resume experiments/finetuning/checkpoints/last.pth

# final evaluation with the same protocol as A
python -m ml.evaluation.evaluate --config ml/configs/finetuned_grid_eval.yaml
```

Commit the result files after each experiment (see CONTINUE_ON_RTX5070.md §4–6 for what is
committed). `last.pth` is written once per epoch, so an interrupted run loses at most one epoch.

## 8. Demo with the GPU model

```powershell
$env:SILENTVOICE_CHECKPOINT_PATH = "experiments/finetuning/checkpoints/best.pth"
$env:SILENTVOICE_DEVICE = "cuda"
python -m uvicorn backend.app.main:app --port 8000
# second terminal
npm --prefix frontend install
npm --prefix frontend run dev        # http://localhost:5173
```

## Troubleshooting (laptop-specific)

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` False | Step 1 (driver / device disabled), or a CPU-only torch wheel was installed |
| `CUDA out of memory` | Lower `max_frames_per_batch` to 225 or 150 (raise `grad_accum_steps` to keep ~32 clips); last resort `decoder_only` |
| Training gets slower over time | Thermal throttling: check `nvidia-smi` clocks/temperature, improve airflow |
| DataLoader workers crash / RAM full | `train.num_workers: 2` |
| `python` opens the Microsoft Store | Activate the venv first, or use `py -3.11` |
