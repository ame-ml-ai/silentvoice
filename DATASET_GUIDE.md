# Dataset guide

## Decision: GRID as the evaluation and fine-tuning dataset (documented scope change)

The brief proposed LRS3. Two verified facts led to GRID as the primary dataset, with LRS3
kept as an optional extension:

1. **Every available Auto-AVSR checkpoint was trained on LRS3's training data** (the
   `trlrs3` in each filename). Fine-tuning on LRS3 trainval again mostly re-teaches data
   the model has already seen. That is a weak test of the research question.
2. **LRS3 access is restricted and uncertain.** The official VGG page
   (`robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html`) returned HTTP 404 on 2026-11-10.
   Mirror pages state that downloads are no longer offered and access is by application.
   SilentVoice does not download restricted data without authorisation.

GRID is openly licensed, and no checkpoint has seen it. That gives a clean domain-shift
test: does partial fine-tuning adapt a large pretrained lip-reader to a new recording
setup and speakers, measured on speakers never seen in training?

## GRID audio-visual sentence corpus

| | |
|---|---|
| Source | Cooke, Barker, Cunningham, Shao (2006), University of Sheffield; [Zenodo record 3625687](https://zenodo.org/records/3625687), DOI 10.5281/zenodo.3625687 |
| Licence | **CC BY 4.0**, open access (verified via the Zenodo API, 2026-11-10) |
| Size | 36 files, 16.21 GB in total. SilentVoice downloads only `alignments.zip` (20 MB) and the 33 per-talker video zips (~13.6 GB). `audio_25k.zip` (2.6 GB) is **not** downloaded |
| Content | 34 talkers × 1,000 sentences of the form *command colour preposition letter digit adverb* ("bin blue at f two now"). Talker s21 has no video. Video is 25 fps, 3 s per sentence (75 frames) |
| Transcripts | Word alignments (`.align`: `start end word`). `sil` and `sp` are dropped, then text is normalised to upper case |

### Splits (speaker-disjoint), `ml/configs/splits/grid.yaml`

| Split | Speakers | Purpose |
|---|---|---|
| test | s1, s2, s20, s22 | LipNet "unseen speakers" protocol (Assael et al., 2016). Used **only** for final evaluation |
| val | s3, s4 | checkpoint selection and early stopping |
| train | the remaining 27 talkers | fine-tuning |

Leakage checks are automatic:
- `validate_splits` and `assert_no_leakage` reject any speaker or clip that appears in two splits. Fine-tuning and evaluation both run them.
- `finetune.py` never loads the test split.

Note: GRID's sentence grammar is shared across speakers by design. A test sentence may
also be spoken by a training speaker. That is inherent to the protocol, not leakage.

### Download and preprocess (on the RTX 5070 machine)

```bash
bash scripts/download_grid.sh data/raw/grid                 # or list speakers: ... s1 s3 s5
python -m ml.preprocessing.prepare_dataset grid --raw-dir data/raw/grid \
    --out-dir data/processed/grid --split-config ml/configs/splits/grid.yaml --workers 8
```

Expected raw layout: `data/raw/grid/<speaker>/*.mpg` and
`data/raw/grid/alignments/<speaker>/*.align`. The download script prints per-speaker
counts. The archive's internal layout was not inspected on the Mac (no download was
made), so check those counts.

## LRS3-TED (optional; only with authorised access)

| | |
|---|---|
| Source | Afouras, Chung, Zisserman (2018), VGG Oxford |
| Access | Application-based. The official page returned 404 on 2026-11-10; check the current route with VGG |
| Licence | Research use. Mirrors state CC BY 4.0 for annotations, with video copyright remaining with the original owners (TED). Confirm when access is granted |
| Splits | `pretrain` (long, not used), `trainval`, `test` (official). SilentVoice holds out a deterministic ~2% of trainval **video ids** as validation, so the test set is never used for selection |
| Caveat | Every checkpoint was trained on LRS3 `pretrain` + `trainval`. The official `test` set is speaker-disjoint from those, so baseline test WER is a legitimate held-out number. Fine-tuning on trainval, however, re-uses data the model has already seen |

```bash
python -m ml.preprocessing.prepare_dataset lrs3 --raw-dir data/raw/lrs3 --out-dir data/processed/lrs3
```

Expected layout: `data/raw/lrs3/{trainval,test}/<video_id>/<clip>.mp4`, plus `<clip>.txt`
whose first line is `Text:  ...`.

## Preprocessing pipeline (identical for datasets and the live demo)

1. **Decode** the video stream with OpenCV. Audio is never read. Videos declaring 25 fps are kept frame-for-frame; others are resampled to 25 fps by presentation timestamp.
2. **Face keypoints**: the MediaPipe face detector gives 4 keypoints per frame (right eye, left eye, nose tip, mouth centre). It uses the full-range model, with short-range as fallback. This is Auto-AVSR's own `detectors/mediapipe/detector.py`.
3. **Interpolation**: frames without a detection get linearly interpolated keypoints. Leading and trailing gaps are padded.
4. **Alignment and crop**: a similarity transform maps the 4 points onto Auto-AVSR's 20-word mean face (256×256 reference). Keypoints are smoothed over a 12-frame window. A **96×96** patch is cut around the mouth centre. This is Auto-AVSR's `video_process.py`, `convert_gray=False`.
5. **Model transform** (`ml/preprocessing/transforms.py`, numerically identical to upstream `VideoTransform`): /255, centre crop 88 (random crop 88 plus time masking for training), grayscale, normalise (0.421, 0.165).

Stored format: 96×96 RGB MPEG-4 (`mp4v`) at 25 fps under `data/processed/<ds>/videos/`,
plus `manifest.jsonl`:

```json
{"id": "grid/s3/bbaf2n", "dataset": "grid", "split": "val", "speaker": "s3",
 "video": "s3/bbaf2n.mp4", "num_frames": 75, "text": "BIN BLUE AT F TWO NOW"}
```

`failures.jsonl` lists every clip that could not be processed, with its split and reason.
`prep_report.json` holds counts and the split-validation report.

## Data limitations

- **Detector difference:** Auto-AVSR's LRS3 crops used RetinaFace (or its published LRS3 landmarks); SilentVoice uses MediaPipe on GRID. That is the same downstream code with slightly different keypoints. It affects baseline and fine-tuned models equally.
- **Mouth-crop storage:** crops are stored as lossy MPEG-4, as upstream does with H.264.
- **Small vocabulary:** GRID has 51 words and a fixed grammar. WER on GRID is far lower than on open-vocabulary data and is not comparable to LRS3 numbers.
- **Uploads:** large uploads are downscaled to a 960 px longest side while decoding, to bound memory. Dataset preprocessing never downscales.

## Never committed

`data/`, `checkpoints/`, `*.pth` and experiment checkpoints are in `.gitignore`.
