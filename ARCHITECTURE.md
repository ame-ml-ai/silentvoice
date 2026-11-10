# Architecture

## Components

```text
┌──────────────── frontend (React 19, TS, Vite, Tailwind 4, Recharts) ────────────────┐
│ Overview · Recognize (upload / webcam, pipeline rail, mouth strip, transcript)        │
│ Research (results, charts, pending states) · Model explorer · Methodology             │
└───────────────────────────── fetch /api/* (Vite proxy) ──────────────────────────────┘
                                         │
┌──────────────────────────────── backend (FastAPI) ───────────────────────────────────┐
│ middleware: early 413 (Content-Length) / early 503 (no model) before body parsing     │
│ api/routes.py ─ thin: save upload → validate (thread) → submit job                    │
│ services/video_service   size/extension limits, temp files, decode validation         │
│ services/job_service     bounded queue, 1 worker thread, job state, PNG preview       │
│ services/model_service   loads checkpoint once at startup (thread), honest state      │
│ services/experiment_service  reads experiments/*/results.json; never synthesises      │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │ imports
┌─────────────────────────────────── ml/ (library) ────────────────────────────────────┐
│ preprocessing/video_io   OpenCV decode (video only), 25 fps resampling, size bound    │
│ preprocessing/mouth      MediaPipe keypoints + Auto-AVSR VideoProcess (96×96 crop)     │
│ preprocessing/transforms upstream-identical eval/train transforms                     │
│ inference/model          E2E build, weights_only checkpoint load, beam search         │
│ inference/pipeline       video → text                                                 │
│ data/                    text norm + tokenizer, manifests + split checks, datasets    │
│ evaluation/              WER/CER (jiwer), evaluate CLI → results.json                 │
│ training/                freezing strategies, fine-tune CLI                           │
│ utils/                   device policy, seeding, env capture, upstream path           │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │ sys.path
                          third_party/auto_avsr (submodule @182b628, unmodified)
```

## Recognition data flow (`POST /api/recognize`)

1. Middleware rejects early if `Content-Length` exceeds the limit or the model isn't `ready` (503).
2. `save_upload` streams the file to the upload directory in 1 MB chunks. It checks the extension and size limit and deletes the file on any error.
3. `validate_video` runs in a worker thread. It decodes once with a bounded output-frame count and downscaling, and returns metadata. Invalid, too long or too short videos get a 422 with a specific code.
4. `JobService.submit` returns 429 when the queue holds 8 active jobs. Otherwise it returns `202 {job_id}`.
5. The worker thread runs: decode, mouth crop, an 8-frame PNG preview, transform, `VSRModel.transcribe` (encoder plus beam search), then text. The job's status is written last, so pollers never see a half-updated job. The upload is deleted.
6. The client polls `GET /api/jobs/{id}` every 600 ms with backoff on transient errors.

`POST /api/preprocess` runs steps 1–5 without the model. That lets the Mac demonstrate the
real preprocessing even without a checkpoint.

## Visual-only guarantee

The model never receives audio, and this is enforced in three ways:
- **Code paths:** OpenCV `VideoCapture` returns video frames only. `E2E` is constructed with modality `"video"`.
- **Static test:** `test_no_audio_imports_in_inference_path` scans every module on the path for `torchaudio`, `librosa`, `soundfile`, `av`, `whisper` and others.
- **Dynamic test:** the same video with no audio, a 300 Hz tone and a 3 kHz tone must give bit-identical frames, model input and model output.

The webcam recorder also requests `audio: false`.

## ML pipeline

```text
raw dataset ── prepare_dataset (MediaPipe crops, CPU, multiprocess) ──▶ videos/*.mp4 + manifest.jsonl
                                                                            │
          ┌──────────────── evaluate (A: pretrained / B: best.pth, same config) ◀┤
          │                                                                  │
  experiments/*/results.json            finetune (train+val only) ──▶ experiments/finetuning/{train_log, checkpoints}
```

Device policy (`ml/utils/device.py`):
- `auto` picks CUDA, otherwise CPU.
- Requesting `cuda` without CUDA raises.
- `mps` is rejected, because upstream beam search is CUDA/CPU only.
- Training exits with code 3 without CUDA unless `--cpu-smoke` is given.

## Frontend structure

- `src/services/api.ts` is a typed fetch client. It turns the error envelope into `ApiError`.
- `src/hooks/useApi.ts` provides `useApi`, which polls and drops stale or post-unmount responses, and `useJob`, which polls a job with retry and backoff and is scoped to the job id.
- `src/components/` holds UI primitives, `PipelineRail`, `MouthStrip`, `VideoInput` (upload plus video-only recorder) and `Layout`.
- `src/pages/` holds one file per route. The Research page is lazy-loaded, so its charts don't weigh down the first load.

States designed per page:
- loading (skeletons)
- empty (prompts)
- error (code-specific hints)
- partial (crops without a transcript when no model is loaded)
- success
- permission (camera denied, missing or busy)

## Integration points

| From | To | Contract |
|---|---|---|
| frontend | backend | `backend/app/schemas/api.py` ⇄ `frontend/src/types/api.ts` |
| backend | ml | `VisualSpeechPipeline`, `VSRModel`, `MouthCropper`, `read_video_rgb` |
| scripts | backend | `experiments/*/results.json`, `training_summary.json`, `train_log.jsonl` |
| ml | upstream | `espnet.nets.*`, `preparation.detectors.mediapipe.*`, `spm/unigram/*` |
