# Model selection

## Candidates

| Model | Visual-only checkpoints | LRS3 VSR WER (published) | Code licence | Notes |
|---|---|---|---|---|
| **Auto-AVSR** (Ma et al., ICASSP 2023), [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) | yes, 4 sizes of training data | 20.3% (best) | Apache-2.0 | Plain PyTorch + vendored ESPnet modules; inference and training code included |
| AV-HuBERT (Shi et al., 2022) | yes | ~26.9% (large, self-training) | Meta research licence (non-commercial) | Built on fairseq, which is hard to install on current PyTorch; heavier |

**Selected: Auto-AVSR.** It meets every requirement of the brief:
- Real visual-only checkpoints.
- The model code is small and self-contained, and builds on PyTorch 2.15 unmodified.
- A permissive code licence.
- Training and evaluation code to compare against.

AV-HuBERT was not needed, so no model switch happened.

## Verified architecture

Read from `third_party/auto_avsr/espnet/nets/pytorch_backend/e2e_asr_conformer.py` and
`frontend/resnet.py` at commit `182b628`. Parameter counts were computed by building the
model (`ml.inference.model.count_parameters`) and are asserted in `tests/ml/test_model.py`.

| Stage | Implementation | Parameters |
|---|---|---|
| Visual frontend | Conv3d(1→64, k 5×7×7, s 1×2×2) + BN + SiLU + MaxPool3d, then a 2D ResNet-18 (BasicBlocks [2,2,2,2], SiLU) per frame, global average pool → 512 | 11,182,784 |
| Projection | Linear 512 → 768 | 393,984 |
| Encoder | Conformer, 12 blocks, d=768, 12 heads, FFN 3072, conv kernel 31, relative positional encoding | 170,451,456 |
| Decoder | Transformer, 6 blocks, d=768, 12 heads, FFN 3072 | 64,472,505 |
| CTC head | Linear 768 → 5049 | 3,882,681 |
| **Total** | | **250,383,410** (model zoo: "250M") |

Other details:
- **Loss:** `0.1 · CTC + 0.9 · attention` (label smoothing 0.1).
- **Decoding:** batch beam search, beam 40, joint CTC (0.1) and decoder (0.9) scores, length bonus 0, no language model.
- **Vocabulary:** a SentencePiece unigram model with 5,047 units. With `<blank>` and `<eos>` that makes 5,049 outputs.
- **Input:** grayscale 88×88 mouth crops at 25 fps, normalised with mean 0.421 and std 0.165.

## Checkpoints (model zoo, README at commit 182b628)

| File | Training data (h) | LRS3 WER | MD5 prefix | Use here |
|---|---|---|---|---|
| `vsr_trlrs3_23h_base.pth` | 438 | 93.0 | fc8db | — |
| `vsr_trlrs3_base.pth` | 438 | 36.0 | c00a7 | — |
| **`vsr_trlrs3vox2_base.pth`** | 1,759 (LRS3 + VoxCeleb2) | 24.6 | 774a6 | **Experiments A and B** |
| `vsr_trlrs2lrs3vox2avsp_base.pth` | 3,291 | 20.3 | 49f77 | optional Experiment C |

Why `vsr_trlrs3vox2_base`:
- Its training data are two well-documented academic corpora.
- None of the checkpoints has seen GRID.
- It is the configured default, and switching to the 3,291 h model is a one-line config change.

All checkpoints are hosted on Google Drive only. The old direct mirror at
`doc.ic.ac.uk/~pm4115/autoAVSR/` returned HTTP 404 when checked on 2026-11-10.

Checkpoint status: **not downloaded on the Mac.** Loading is tested with random-weight
state dicts saved in the same format. `scripts/verify_checkpoint.py` checks the MD5 and
does a strict load when the real file is present.

## Licence

- **Code:** Apache-2.0 ("Code is Apache 2.0 licensed").
- **Weights:** the README says they "may have their own licenses or terms and conditions derived from the dataset used for training". LRS3 and VoxCeleb2 are research datasets.

SilentVoice uses the weights for non-commercial university research only, and does not redistribute them; they are git-ignored.

## Hardware requirements

| | Mac M4, 16 GB (verified) | RTX 5070, 12 GB (to verify) |
|---|---|---|
| Model build + forward (random weights) | works on CPU: 50 frames → encoder in 0.34 s | expected to fit easily (1 GB fp32 weights) |
| Beam search | CPU works. **MPS fails**: upstream `ctc_prefix_score.py` allocates index tensors only on `cuda` or `cpu` | CUDA supported upstream |
| Fine-tuning | not attempted (out of scope for the Mac) | static memory estimate 2.5 GB for `encoder_top_k`; activations measured by `scripts/gpu_memory_probe.py` |

## Known limitations

- **Upstream bugs found (not fixed, worked around):**
  - `preparation/preprocess_lrs2lrs3.py` checks `if args.gpu_type != "cuda" or "mps"`, which is always true, so the script always raises. SilentVoice has its own `prepare_dataset.py`.
  - `datamodule/data_module.py` uses `random` without importing it. That path only runs with shuffle on, and SilentVoice doesn't use it.
  - `train.py` defaults `--val-file` to the **LRS3 test set**, which would leak test data into checkpoint selection. SilentVoice always validates on a separate, speaker-disjoint split.
- The upstream MediaPipe detector uses `mp.solutions`, which was removed in mediapipe 1.x. It is pinned to 0.10.14.
- The checkpoints were trained with RetinaFace crops for LRS3; SilentVoice uses MediaPipe (see DATASET_GUIDE.md).
