"""Evaluate a VSR checkpoint on one split of a manifest and record everything.

    # RTX 5070 (real result)
    python -m ml.evaluation.evaluate --config ml/configs/baseline_grid.yaml

    # Mac smoke run (CPU, first 5 clips; written to results_smoke.json, never shown as a result)
    python -m ml.evaluation.evaluate --config ml/configs/baseline_grid.yaml --device cpu --limit 5

Writes ``<output_dir>/results.json`` and ``predictions.jsonl``. A run with ``--limit``
is a smoke run and writes ``results_smoke.json`` instead, so it can never be
mistaken for the experiment result.
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ml.data.manifest import assert_no_leakage, read_manifest, select_split, validate_splits
from ml.evaluation.metrics import char_errors, word_errors
from ml.inference.model import DecodeConfig, VSRModel
from ml.preprocessing.transforms import eval_transform, frames_to_tensor
from ml.preprocessing.video_io import read_video_rgb
from ml.utils.device import describe_device, resolve_device
from ml.utils.env import environment_record, md5_file, set_seed

log = logging.getLogger("evaluate")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def excluded_clip_count(manifest: Path, split: str) -> int | None:
    """Clips of this split dropped during preprocessing (failures.jsonl next to the manifest)."""
    path = manifest.parent / "failures.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sum(r.get("split") == split for r in rows)


def run(cfg: dict[str, Any], limit: int | None, command: str) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))
    all_entries = read_manifest(cfg["manifest"])
    assert_no_leakage(validate_splits(all_entries))
    entries = select_split(all_entries, cfg.get("split", "test"))
    if limit:
        entries = entries[:limit]
    data_root = Path(cfg["data_root"])
    ckpt = Path(cfg["checkpoint"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = VSRModel(ckpt, device, DecodeConfig(**cfg.get("decode", {})))
    transform = eval_transform()

    refs, hyps, latencies, frames_total, rows = [], [], [], 0, []
    start = time.perf_counter()
    for i, e in enumerate(entries, 1):
        frames, _ = read_video_rgb(data_root / e.video)
        video = transform(frames_to_tensor(frames))
        out = model.transcribe(video)
        refs.append(e.text)
        hyps.append(out.text)
        latencies.append(out.inference_seconds)
        frames_total += out.num_frames
        rows.append({"id": e.id, "reference": e.text, "hypothesis": out.text, "seconds": round(out.inference_seconds, 4)})
        if i % 50 == 0 or i == len(entries):
            log.info("%d/%d  running WER %.4f", i, len(entries), word_errors(refs, hyps).rate)
    wall = time.perf_counter() - start

    w, c = word_errors(refs, hyps), char_errors(refs, hyps)
    audio_seconds = frames_total / 25.0
    return {
        "status": "smoke" if limit else "complete",
        "experiment": cfg.get("experiment"),
        "split": cfg.get("split", "test"),
        "num_samples": len(entries),
        "excluded_by_preprocessing": excluded_clip_count(Path(cfg["manifest"]), cfg.get("split", "test")),
        "wer": w.rate,
        "cer": c.rate,
        "word_counts": w.to_dict(),
        "char_counts": c.to_dict(),
        "latency": {
            "mean_s": statistics.fmean(latencies),
            "median_s": statistics.median(latencies),
            "p95_s": _percentile(latencies, 0.95),
            "real_time_factor": sum(latencies) / audio_seconds,
            "measured_on": "model forward + beam search only (excludes decode and mouth cropping)",
        },
        "memory": {
            "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device.type == "cuda" else None,
            "peak_process_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2), 3),
        },
        "parameters": model.info()["parameters"],
        "checkpoint": {"name": ckpt.name, "md5": md5_file(ckpt)},
        "decode": cfg.get("decode", {}),
        "device": describe_device(device),
        "wall_clock_s": round(wall, 1),
        "seed": cfg.get("seed", 42),
        "config": cfg,
        "environment": environment_record(),
        "command": command,
        "_predictions": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", help="override config checkpoint (e.g. fine-tuned best.pth)")
    p.add_argument("--split", choices=["val", "test"])
    p.add_argument("--device")
    p.add_argument("--output-dir")
    p.add_argument("--experiment")
    p.add_argument("--limit", type=int, help="smoke run on the first N clips")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    if "freeze" in cfg or "train" in cfg:
        p.error(f"{args.config} is a training config. Evaluate with baseline_grid.yaml or finetuned_grid_eval.yaml "
                "so both experiments share one decoding protocol.")
    for key in ("checkpoint", "split", "device", "output_dir", "experiment"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    command = " ".join(shlex.quote(a) for a in [sys.executable, "-m", "ml.evaluation.evaluate", *(argv or sys.argv[1:])])
    result = run(cfg, args.limit, command)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = result.pop("_predictions")
    suffix = "_smoke" if args.limit else ""
    with open(out_dir / f"predictions{suffix}.jsonl", "w") as fh:
        for row in preds:
            fh.write(json.dumps(row) + "\n")
    (out_dir / f"results{suffix}.json").write_text(json.dumps(result, indent=2))
    log.info("%s WER=%.4f CER=%.4f on %d %s clips -> %s", result["status"], result["wer"], result["cer"],
             result["num_samples"], result["split"], out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
