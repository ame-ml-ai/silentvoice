# SilentVoice

**Visual-only speech recognition research prototype.** SilentVoice reads speech from video
of a speaker's mouth, with no audio, using the pretrained Auto-AVSR visual speech
recognition model. It studies whether partial fine-tuning on a new domain (the GRID
corpus) improves recognition on unseen speakers, within one 12 GB GPU.

> **Status (2026-11-10):** the stack runs end to end on an RTX 4060 Laptop (Windows).
> **Pilot result:** partial fine-tuning on 2 GRID speakers cut WER on unseen speakers from
> **0.867 to 0.224** (500 test clips; see [Research results](#research-results)). The full
> GRID run is next: [docs/GPU_SETUP_RTX4060_LAPTOP.md](docs/GPU_SETUP_RTX4060_LAPTOP.md).

## Features

- **Visual-only pipeline:**
  - Video is decoded from the video stream only and resampled to 25 fps.
  - MediaPipe face keypoints feed Auto-AVSR's own alignment and a 96×96 mouth crop.
  - Inference runs the 250M-parameter Conformer model with CTC/attention beam search.
  - Tests prove that changing the audio track changes nothing.
- **FastAPI backend:**
  - Upload validation (format, size, duration) and background jobs.
  - Honest model state: without a checkpoint, recognition returns `503 model_unavailable`, while mouth-region extraction still works.
  - Experiment endpoints that only report results written by the evaluation scripts.
- **React frontend:**
  - Upload or webcam recording (video only).
  - A live pipeline view that shows the real mouth crops the model sees.
  - A research dashboard with pending states, a model explorer and a methodology page.
- **Reproducible ML pipeline:**
  - Dataset preparation with speaker-disjoint splits and automatic leakage checks.
  - WER/CER evaluation that records checkpoint hash, seed, versions, hardware and command.
  - A fine-tuning script (partial freezing, bf16, gradient accumulation, early stopping on validation WER, resume).
  - A GPU memory probe.

## Architecture at a glance

```text
browser (React, :5173) ──/api──▶ FastAPI (:8000) ──▶ JobService (1 worker thread)
                                     │                    │
                                     │                    ▼
                                     │      video_io (OpenCV, video stream only, 25 fps)
                                     │      → MouthCropper (MediaPipe + Auto-AVSR VideoProcess)
                                     │      → transforms (88×88 gray, normalised)
                                     │      → VSRModel (Auto-AVSR E2E + beam search)
                                     ▼
                       experiments/*/results.json ◀── ml.evaluation.evaluate / ml.training.finetune
```

Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Repository layout

```text
backend/app/        FastAPI app: api/ core/ services/ schemas/      backend/tests/
frontend/           React + TypeScript + Vite + Tailwind
ml/                 data/ preprocessing/ inference/ evaluation/ training/ utils/ configs/
scripts/            gpu_check.py · gpu_memory_probe.py · verify_checkpoint.py · download_grid.sh
tests/ml/           ML unit, smoke and no-audio tests
experiments/        results written by the scripts (currently empty: pending)
third_party/auto_avsr   git submodule, pinned to 182b628 (Apache-2.0)
docs/GPU_SETUP_RTX5070.md
```

## Setup: Mac / CPU development (Environment A)

Requirements: Python 3.11, Node 20+, git.

```bash
git clone --recurse-submodules <repo-url> silentvoice && cd silentvoice
python3.11 -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.11 .venv
pip install -r requirements/cpu.txt
npm --prefix frontend install
cp .env.example .env    # optional
```

The setup above was verified on macOS 26.2 (Apple M4, Python 3.11.18 via uv). For
GPU setup see [docs/GPU_SETUP_RTX5070.md](docs/GPU_SETUP_RTX5070.md).

## Running

```bash
# backend (http://localhost:8000/docs for the OpenAPI UI)
uvicorn backend.app.main:app --port 8000

# frontend (http://localhost:5173), proxies /api to :8000
npm --prefix frontend run dev
```

Without a checkpoint, the app runs fully except the transcript: "Recognize speech" is
disabled with an explanation, and "Extract mouth region only" runs the real preprocessing.
With a checkpoint:

```bash
SILENTVOICE_CHECKPOINT_PATH=checkpoints/vsr_trlrs3vox2_base.pth uvicorn backend.app.main:app --port 8000
```

On the Mac this runs on CPU (MPS is not supported by the upstream CTC scorer). CPU
latency is not representative of the RTX 5070.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, model state, device |
| GET | `/api/model/info` | verified architecture, parameter counts, loaded checkpoint |
| POST | `/api/preprocess` | upload → validate → mouth crops (no model needed) → `202 {job_id}` |
| POST | `/api/recognize` | upload → full recognition → `202 {job_id}`; `503 model_unavailable` without a model |
| GET | `/api/jobs/{id}` | job status, mouth-crop preview, text, timings |
| GET | `/api/experiments` | all experiment records (`complete` / `smoke` / `pending`) |
| GET | `/api/experiments/compare` | baseline vs fine-tuned, only when both are complete on the same split |

Errors always have the shape `{"error": {"code", "message", "details"}}`.

## Tests

```bash
python -m pytest -q                 # 69 tests, ~30 s on the M4 (incl. CPU smoke tests of the 250M model)
python -m pytest -q -m "not slow"   # fast subset
npm --prefix frontend run build     # type-check + production build
```

What the tests cover:
- Metrics and tokenizer.
- Video decoding and resampling.
- Mouth cropping.
- Preprocessing parity with Auto-AVSR.
- Model architecture and parameter count.
- Checkpoint loading.
- No silent device fallback.
- Freezing strategies.
- Split leakage detection and batch sampling.
- **Audio independence.**
- Evaluate and fine-tune CPU smoke runs.
- All API endpoints and error paths.
- End-to-end recognition through the API with a random-weight checkpoint.

## Commands by environment

| | Mac (CPU) | RTX 5070 (CUDA) |
|---|---|---|
| Environment check | `python -m pytest -q` | `python scripts/gpu_check.py` |
| Smoke: evaluation | `... evaluate --config ml/configs/baseline_grid.yaml --device cpu --limit 5` | — |
| Smoke: training | `... finetune --config ml/configs/finetune_grid.yaml --cpu-smoke` | `... finetune ... --max-steps 20` |
| Baseline (A) | not run on the Mac | `python -m ml.evaluation.evaluate --config ml/configs/baseline_grid.yaml` |
| Fine-tuning (B) | refused (exit 3) | `python -m ml.training.finetune --config ml/configs/finetune_grid.yaml` |
| Final evaluation | not run on the Mac | `python -m ml.evaluation.evaluate --config ml/configs/finetuned_grid_eval.yaml` |

## Research results

Full protocol (all GRID talkers): **pending**.

### Pilot results (2026-11-10, RTX 4060 Laptop) — subset, not the final protocol

Same checkpoint, same decoding (beam 40, CTC weight 0.1, no LM), same **500 test clips** from the
unseen test speakers s1 + s2 (250 each, seed 0; ids in `experiments/pilot/test_ids.txt`).
Fine-tuning used only GRID **s5 + s6** for training (2,000 clips) and s3 + s4 for validation,
2 epochs (124 optimizer steps, 7.8 min), `encoder_top_k` (top 4 Conformer blocks + decoder +
CTC head = 125.2M of 250.4M parameters trainable), bf16, lr 1e-5, warmup 20.

| Model | WER | CER | Source |
|---|---|---|---|
| Pretrained Auto-AVSR (baseline) | **0.867** | **0.628** | `experiments/pilot/baseline/results.json` |
| Fine-tuned, 2 epochs on 2 speakers | **0.224** | **0.146** | `experiments/pilot/finetuning/results.json` |

Validation WER (200 clips, beam 10): 0.239 after epoch 1, 0.165 after epoch 2. Peak GPU memory:
2.96 GB training, 1.3 GB inference. Latency figures in the result files are not representative:
both evaluations shared the GPU with each other and with training.

Caveats: 4 of the 33 talkers, one seed, 500 test clips. GRID's 51-word grammar makes this gain
far easier than on open speech. The full protocol (27 training talkers, all ~4k test clips of
s1, s2, s20, s22) is the next run.

Design and protocol: [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md).

## Limitations

- Only a pilot result exists (2 training speakers, 500 test clips); the full-protocol numbers are pending.
- GRID has a tiny fixed vocabulary, so gains there do not transfer to open-vocabulary speech. LRS3 needs authorised access ([DATASET_GUIDE.md](DATASET_GUIDE.md)).
- MediaPipe replaces the RetinaFace detector used for Auto-AVSR's official LRS3 crops.
- The live demo on new faces, poses and lighting will be less accurate than test-set numbers.
- No authentication, and jobs live in memory: this is a local research demo, not a service.

## Documentation

| File | Contents |
|---|---|
| [PROJECT_AUDIT.md](PROJECT_AUDIT.md) | environment audit, risks |
| [MODEL_SELECTION.md](MODEL_SELECTION.md) | candidates, verified architecture, checkpoints, licences |
| [DATASET_GUIDE.md](DATASET_GUIDE.md) | GRID / LRS3 access, splits, preprocessing |
| [ARCHITECTURE.md](ARCHITECTURE.md) | system components and data flow |
| [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) | research question, protocol, metrics |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) / [PROGRESS.md](PROGRESS.md) | phases and status |
| [ECC_USAGE.md](ECC_USAGE.md) | how the ECC plugin was actually used |
| [CONTINUE_ON_RTX5070.md](CONTINUE_ON_RTX5070.md) | step-by-step GPU phase |

## Credits and licences

- **Auto-AVSR**: P. Ma, A. Haliassos, A. Fernandez-Lopez, H. Chen, S. Petridis, M. Pantic, "Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels", ICASSP 2023. Code is Apache-2.0 (used unmodified as a submodule). Pretrained weights carry terms derived from their training data.
- **GRID corpus**: M. Cooke, J. Barker, S. Cunningham, X. Shao (2006), CC BY 4.0, [Zenodo 10.5281/zenodo.3625687](https://zenodo.org/records/3625687).
- **MediaPipe** (Google), Apache-2.0.
- SilentVoice code: Apache-2.0.
