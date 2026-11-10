# Continue on the RTX 5070

The Mac phase built and tested the whole software stack. This checklist runs the GPU
phase end to end. It needs environment setup and execution only, no code changes.
Expect roughly half a day of wall-clock time, most of it preprocessing and training.

> Rule: results go into `experiments/` **only** via these scripts. Commit them after each
> step so the dashboard and the report use the measured numbers.

## 1. Clone and set up (≈30 min)

```bash
git clone --recurse-submodules <your-repo-url> silentvoice && cd silentvoice
```

Follow **[docs/GPU_SETUP_RTX5070.md](docs/GPU_SETUP_RTX5070.md)** steps 2–5:
- Python 3.11 venv.
- A CUDA 12.8+ PyTorch build, then `pip install -r requirements/gpu.txt`.
- `python scripts/gpu_check.py` must pass, and so must `python -m pytest -q`.
- Download the checkpoint, then `python scripts/verify_checkpoint.py checkpoints/vsr_trlrs3vox2_base.pth`.

## 2. Dataset (≈1–3 h, mostly download + CPU preprocessing)

GRID is CC BY 4.0 and openly downloadable. Only video and alignments are fetched, no audio.

```bash
bash scripts/download_grid.sh data/raw/grid          # ~13.6 GB, 33 talkers
python -m ml.preprocessing.prepare_dataset grid \
    --raw-dir data/raw/grid --out-dir data/processed/grid \
    --split-config ml/configs/splits/grid.yaml --workers 8
```

Check `data/processed/grid/prep_report.json`:
- `split_report.ok` must be `true`, meaning no speaker overlap and no missing files.
- Note `failed_by_split`. Clips without a detectable face are excluded, and `evaluate.py` reports that count in every result.
- Expected counts are about 27k train, 2k val and 4k test clips. The exact numbers depend on failures.

Tip: first run with `--speakers s1,s5 --limit 50` to check the pipeline in a minute.

LRS3 is optional and needs authorised access (see [DATASET_GUIDE.md](DATASET_GUIDE.md)).

## 3. Memory check (≈5 min)

```bash
python scripts/gpu_memory_probe.py --strategy encoder_top_k --blocks 4 --amp bf16 --frames 150 300 600 900 1200
```

Set `train.max_frames_per_batch` and `max_batch_size` in `ml/configs/finetune_grid.yaml`
from `experiments/hardware/memory_probe.json` (peak below ~10 GB). Commit the probe file.

## 4. Experiment A: baseline (≈30–60 min)

```bash
python -m ml.evaluation.evaluate --config ml/configs/baseline_grid.yaml
git add experiments/baseline experiments/hardware && git commit -m "Experiment A: baseline on GRID unseen speakers"
```

This writes `experiments/baseline/results.json` (WER, CER, latency, peak VRAM, parameters,
checkpoint MD5, environment) and `predictions.jsonl`.

## 5. Experiment B: fine-tuning

Short test first (≈5 min). This confirms memory, speed and that the loss decreases:

```bash
python -m ml.training.finetune --config ml/configs/finetune_grid.yaml --max-steps 20
cat experiments/finetuning/train_log.jsonl | tail -3
```

Then delete the probe output (`rm -r experiments/finetuning`) and run the real thing:

```bash
python -m ml.training.finetune --config ml/configs/finetune_grid.yaml
# interrupted? resume:
python -m ml.training.finetune --config ml/configs/finetune_grid.yaml --resume experiments/finetuning/checkpoints/last.pth
```

How it behaves:
- It validates every epoch on the val speakers (s3, s4).
- It keeps `checkpoints/best.pth` by validation WER.
- It stops early after 3 epochs without improvement.
- It never touches the test split.

## 6. Final evaluation (same protocol as A)

```bash
python -m ml.evaluation.evaluate --config ml/configs/finetuned_grid_eval.yaml
git add experiments/finetuning && git commit -m "Experiment B: partial fine-tuning results"
```

Open the dashboard (`uvicorn backend.app.main:app` + `npm --prefix frontend run dev`,
<http://localhost:5173/research>). With both result files present, it shows the measured
comparison. It refuses to compare runs whose split or sample count differ.

## 7. Optional Experiment C (ablations)

```bash
# decoder-only fine-tuning
cp ml/configs/finetune_grid.yaml ml/configs/ablation_decoder_only.yaml
#   edit: experiment, output_dir: experiments/ablations/decoder_only, freeze.strategy: decoder_only
python -m ml.training.finetune --config ml/configs/ablation_decoder_only.yaml
python -m ml.evaluation.evaluate --config ml/configs/finetuned_grid_eval.yaml \
    --checkpoint experiments/ablations/decoder_only/checkpoints/best.pth \
    --output-dir experiments/ablations/decoder_only --experiment ablation_decoder_only
```

## 8. Update the write-up

- `PROGRESS.md`: record what ran, the versions, wall-clock times and any deviations.
- `EXPERIMENT_DESIGN.md` "Results": copy numbers from the result files, not from memory.
- Run the demo with the GPU model: `SILENTVOICE_CHECKPOINT_PATH=experiments/finetuning/checkpoints/best.pth SILENTVOICE_DEVICE=cuda uvicorn backend.app.main:app`.
