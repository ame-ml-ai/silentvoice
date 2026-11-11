# experiments/

Results are written here **only** by the scripts, never by hand.

| Path | Written by | Contents |
|---|---|---|
| `baseline/results.json` | `ml.evaluation.evaluate --config ml/configs/baseline_grid.yaml` | Experiment A metrics, latency, memory, checkpoint MD5, config, environment, command |
| `baseline/predictions.jsonl` | same | `{id, reference, hypothesis, seconds}` per test clip |
| `finetuning/training_summary.json`, `train_log.jsonl` | `ml.training.finetune` | loss / LR / memory log, validation WER per epoch, best epoch |
| `finetuning/checkpoints/` | same | `best.pth`, `last.pth` (git-ignored) |
| `finetuning/results.json` | `ml.evaluation.evaluate ... --checkpoint experiments/finetuning/checkpoints/best.pth` | Experiment B test metrics |
| `ablations/<name>/` | same scripts with `--output-dir` | optional Experiment C |
| `hardware/memory_probe.json` | `scripts/gpu_memory_probe.py` | measured peak VRAM per frame budget |

`*_smoke.json` and `*/smoke/` come from CPU pipeline checks on the Mac. They are git-ignored,
labelled `status: "smoke"`, and never shown as experiment results.

Status on 2026-11-10: the **pilot** (2 training speakers, 500 test clips) is complete on the RTX 4060 Laptop, see `pilot/`. The full protocol is pending.
