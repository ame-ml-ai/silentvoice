# Progress

## 2026-11-10: Mac phase (Environment A)

### Completed
- Environment audit; Python 3.11 venv via uv; dependency compatibility fixes (mediapipe pin, OpenCV-only decoding, no MPS for decoding).
- Auto-AVSR pinned as a submodule. Architecture verified against code, 250,383,410 parameters.
- ML library:
  - video decoding with 25 fps resampling and bounded memory
  - MediaPipe + upstream mouth cropping
  - upstream-identical transforms
  - tokenizer
  - WER/CER
  - manifests with leakage checks
  - speaker-mixed frame-budget sampler
  - model loading (`weights_only`)
  - beam search
  - end-to-end pipeline
- Scripts:
  - `prepare_dataset` (GRID, LRS3)
  - `evaluate` (results.json with full provenance)
  - `finetune` (partial freezing, bf16, accumulation, early stopping on validation WER, resume, NaN guard)
  - `gpu_check`, `gpu_memory_probe`, `verify_checkpoint`, `download_grid.sh`
- FastAPI backend: health, model info, preprocess, recognize, jobs, experiments, compare. Early 413/503, bounded queue, error envelope.
- React frontend: five pages, upload and video-only webcam, live pipeline, mouth-crop strip, research dashboard with pending states, model explorer, methodology.
- ECC review by three ECC agents (ML, FastAPI, React). All high and medium findings were fixed (see ECC_USAGE.md).
- Documentation set, GPU setup guide, RTX 5070 continuation guide.

### Test status
- `python -m pytest -q`: **69 passed** (includes slow CPU smoke tests of the full 250M model).
- `npm --prefix frontend run build`: type-check and build pass.
- Browser check against the running API:
  - all pages render at desktop width and at 375 px with no horizontal scroll
  - a browser-generated WebM upload (30 fps) was resampled to 25 fps
  - the `no_face` error was shown correctly
  - Recognize was disabled with an explanation while no model was loaded
  - the 404 error envelope was correct

### Not done / not verified
- **No real checkpoint has been downloaded or run.** Recognition with real weights is unverified.
- **No dataset downloaded.** The GRID archive layout is not verified.
- **No experiment run.** All metrics are pending.
- MediaPipe face detection on a real face was not exercised on the Mac: no face video was available, and test videos are synthetic. The crop code is tested with fixed keypoints.
- The webcam recorder was not exercised with a real camera in this session.

### Repository
- Pushed to <https://github.com/ame-ml-ai/silentvoice> (public), branch `main`. Clone with `--recurse-submodules`.

### Blockers
- GPU work needs the RTX 5070 machine.

## 2026-11-10: Windows laptop phase (Environment B = RTX 4060 Laptop, not RTX 5070)

### Found on clone
- **`ml/data/` was missing from GitHub.** The `.gitignore` rule `data/` also matched `ml/data/`, so
  `dataset.py`, `manifest.py`, `sources.py` and `text.py` were never pushed; nothing could import.
  The Mac is not available, so the package was **re-implemented** from the tests and call sites
  (`ml/data/*`). It is not byte-identical to the Mac version. `.gitignore` now uses `/data/`.
- `ml/evaluation/evaluate.py` imported `resource` (Unix-only) and would crash on Windows.
- Hardware differs from the plan: RTX 4060 Laptop 8 GB (sm_89), 16 GB RAM, i7-13620H, Windows 11.

### Changes
- `ml/data/`: re-implemented. Notable choices: the tokenizer is picklable (Windows DataLoader
  workers use spawn); the frame-budget sampler budgets *padded* frames (max length x batch size).
- `ml/utils/env.py`: `peak_rss_gb()` for Windows/macOS/Linux; used by `evaluate.py`.
- `ml/configs/finetune_grid.yaml`: 300 frames / batch 4 / accum 8 (effective batch 32 unchanged), 4 workers.
- `scripts/gpu_memory_probe.py`: `--budget-gb` (default 80 % of VRAM) and a printed recommendation.
- `prepare_dataset` default workers capped at 8; the local dev-server launch config uses the Windows venv layout;
  device/gpu_check messages no longer assume an RTX 5070.
- New guide: [docs/GPU_SETUP_RTX4060_LAPTOP.md](docs/GPU_SETUP_RTX4060_LAPTOP.md).

### Environment (installed and verified 2026-11-10)
- Python 3.11.9 (winget), `.venv`, **torch 2.15.0+cu132** / torchvision 0.29.2 (the cu128 index stops at
  torch 2.11; NVIDIA driver 595.97 supports CUDA 13.x), then `requirements/gpu.txt`, plus `gdown`.
- CUDA works: RTX 4060 Laptop GPU, 8.0 GB, sm_89, bf16, matmul OK. (`nvidia-smi` and the `Unknown`
  device status were only an artefact of the non-admin sandbox; driver problem code is 0.)
- `python -m pytest -q`: **68 passed, 1 skipped** (the "refuses without CUDA" test skips on a CUDA machine).
  This validates the re-implemented `ml/data/` against the full suite, including both slow smoke tests.
- Checkpoint `vsr_trlrs3vox2_base.pth`: MD5 774a6305c0b57ae0cffc97b58b68c3bf (zoo prefix matches), strict load,
  250,383,410 parameters.

### Memory probe (`experiments/hardware/memory_probe.json`)
encoder_top_k k=4, bf16, 75-frame clips: peak 2.84 GB at 150 frames ... 2.88 GB at 600 frames. Activations are
negligible for 3 s clips; weights + AdamW dominate. `finetune_grid.yaml` is back at 600 frames / 8 clips /
4 accumulation steps (effective batch 32).

### GRID small test (s1 + s5 only)
- Download layout matches `<raw>/<spk>/*.mpg` + `<raw>/alignments/<spk>/*.align`.
- **Data issue found:** for s5, 0 of 1,000 video names have a matching `.align` file (e.g. `bbae8n.mpg`
  vs `bbae8s.align`); s1 matches 1,000/1,000. GRID names encode the sentence, and the decoded name equals the
  alignment text for all 2,000 alignment files, so `iter_grid` now falls back to the file-name transcript
  (`parse_grid_filename`, tested). Check: baseline CER on 40 s5 clips is 0.651 vs the file-name sentence
  and 0.895 vs other clips' sentences, so the file names describe the videos.
- Preprocessing: 2,000/2,000 clips OK, 0 failures, ~2 min with 6 workers (~15 clips/s).
- Baseline smoke (`--limit 100`, s1, beam 40, GPU): WER 0.857, CER 0.617, 1.21 s/clip, RTF 0.40,
  peak GPU 1.19 GB. Smoke numbers only; not an experiment result.

### Pilot experiment (deadline run, 17:28-17:50)
- Data: s1-s6 preprocessed (6,000 clips, 0 failures). Pilot manifest `data/processed/grid/manifest_pilot.jsonl`:
  train s5+s6 (2,000), val s3+s4 (2,000; 200 sampled for WER), test 500 clips (250 of s1, 250 of s2, seed 0;
  ids committed in `experiments/pilot/test_ids.txt`).
- Configs `ml/configs/pilot/{baseline,finetune,finetuned_eval}.yaml`; fine-tune = finetune_grid.yaml with
  2 epochs, warmup 20, val_max_samples 200.
- **Baseline WER 0.867 / CER 0.628 -> fine-tuned WER 0.224 / CER 0.146** (same 500 clips, beam 40).
  Training: 124 steps, 470 s, peak 2.96 GB; val WER 0.239 (epoch 1) -> 0.165 (epoch 2).
- Latency in these result files is confounded (evaluations ran concurrently with each other and with training).
- New `scripts/progress.py` (`--watch`): download / preprocessing / training ETA / evaluation progress bars.
- GRID download: all 34 zips present (parallel fetch, 6 connections; ~4-9 MB/s vs ~1.3 MB/s sequential).

### Remaining
- Unzip s7-s34 (the parallel fetch did not extract them), preprocess all talkers, then the full Experiments A
  and B with `ml/configs/baseline_grid.yaml`, `finetune_grid.yaml`, `finetuned_grid_eval.yaml` (~5-10 h).

## Next
Follow [docs/GPU_SETUP_RTX4060_LAPTOP.md](docs/GPU_SETUP_RTX4060_LAPTOP.md). First action after
installing the environment: `python -m pytest -q` to validate the re-implemented `ml/data/`.
(The original RTX 5070 plan stays in [CONTINUE_ON_RTX5070.md](CONTINUE_ON_RTX5070.md).)
