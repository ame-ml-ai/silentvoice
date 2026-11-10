# Experiment design

## Research question

Can fine-tuning a pretrained visual speech recognition model improve recognition
performance on a selected evaluation dataset under limited computational resources?

Operationalised:
- **Model:** Auto-AVSR `vsr_trlrs3vox2_base` (250.4M parameters).
- **Target domain:** GRID.
- **Evaluation set:** the 4 unseen test speakers (s1, s2, s20, s22).
- **Budget:** a single RTX 5070 with 12 GB.

**Hypothesis:** partial fine-tuning on 27 GRID training speakers lowers test WER relative
to the zero-shot pretrained model.

## Experiments

| | A: Baseline | B: Partial fine-tuning | C: Ablations (optional) |
|---|---|---|---|
| Weights | pretrained, unchanged | pretrained → fine-tuned on GRID train | e.g. `decoder_only`; full fine-tune if memory allows; 3,291 h checkpoint |
| Trainable params | 0 | 125,173,362 (top 4/12 Conformer blocks + decoder + CTC) | decoder_only: 68,355,186 |
| Config | `ml/configs/baseline_grid.yaml` | `finetune_grid.yaml` → `finetuned_grid_eval.yaml` | copies with `freeze.strategy` changed |
| Output | `experiments/baseline/` | `experiments/finetuning/` | `experiments/ablations/<name>/` |

### Fine-tuning settings (starting hypotheses, validated on the GPU)

| Setting | Value | Reason |
|---|---|---|
| Frozen | Visual frontend, projection, Conformer blocks 1–8, all BatchNorm statistics | keeps low-level visual features; saves memory |
| Optimiser | AdamW, lr 1e-5, weight decay 0.03, betas (0.9, 0.98) | Auto-AVSR's optimiser and betas, with a conservative fine-tuning LR |
| Schedule | linear warmup over 200 steps, then cosine decay | as upstream (`WarmupCosineScheduler`) |
| Precision | bf16 autocast; non-finite gradients are skipped and logged | RTX 50-series bf16 support; bf16 has no GradScaler |
| Batching | frame budget (default 600 frames ≈ 8 GRID clips) × gradient accumulation 4; speaker-mixed batches | memory control; set from the measured probe |
| Augmentation | random crop 88 + adaptive time masking | upstream training transform |
| Gradient clip | 10.0 | upstream Trainer setting |
| Selection | lowest **validation** WER (400 seeded val clips, beam 10) each epoch; early stopping patience 3; max 10 epochs | the test split is never seen |
| Seed | 42 | recorded in every result |

## Metrics

- **WER** = (S + D + I) / N at word level.
- **CER** is the same at character level, with spaces counted.

Both are computed **corpus-wide** (summed errors over summed reference length) with `jiwer`
4.0.0 after identical normalisation:
- upper-case
- accents stripped
- only `[A-Z0-9' ]` kept
- whitespace collapsed

Implementation: `ml/evaluation/metrics.py`, with hand-computed tests in `tests/ml/test_metrics_and_text.py`.

Also recorded:
- **Latency:** model forward plus beam search per clip, as mean, median, p95 and real-time factor.
- **Memory:** peak GPU allocation and peak process RSS.
- **Parameter counts:** total and trainable.
- **Training curves:** loss, LR and validation WER per epoch.
- **Run metadata:** training time, checkpoint MD5, software versions, hardware, command, and clips excluded by preprocessing.

## Evaluation protocol

1. Both experiments use the same manifest, the same `split: test`, and the same decoding settings (beam 40, CTC weight 0.1, no LM). `evaluate.py` refuses training configs, so the decoding protocol can't diverge.
2. The dashboard compares A and B only when both have `status: complete` and the same split and sample count.
3. Clips whose preprocessing failed are excluded identically for both; their count is in `results.json`.
4. Smoke runs (`--limit`, `--cpu-smoke`) write separate, git-ignored files labelled `smoke`.

## Contamination check

| Data | Seen by the checkpoint? | Seen by fine-tuning? |
|---|---|---|
| GRID test speakers | no (no GRID in LRS3/VoxCeleb2) | no (speaker-disjoint, checked) |
| GRID val speakers | no | selection only |
| LRS3 test (optional extension) | no: speaker-disjoint official test set | no |

## Experimental limitations

- **Single seed, single run:** no confidence intervals. With time to spare, rerun B with 2 more seeds.
- **Domain:** GRID's 51-word grammar makes the task much easier than open speech, so absolute WERs are not comparable to LRS3 results.
- **Forgetting:** the fine-tuned model may get worse on general speech. That can only be measured with LRS3 access.
- **Noisy selection:** validation WER comes from 2 speakers and 400 clips decoded at beam 10. That is noisy on near-saturated GRID WER.
- **Different detector:** MediaPipe here versus RetinaFace upstream (see DATASET_GUIDE.md).

## Results

**Pending: no experiment has been run.** This section is filled from
`experiments/*/results.json` after the RTX 5070 runs (see CONTINUE_ON_RTX5070.md).
