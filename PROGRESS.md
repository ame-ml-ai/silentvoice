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
- No git remote; nothing pushed.

### Blockers
- GPU work needs the RTX 5070 machine.
- Pushing to GitHub needs a remote URL (or `gh` login) from the user.

## Next (Environment B)
Follow [CONTINUE_ON_RTX5070.md](CONTINUE_ON_RTX5070.md).
