"""Live progress of the long-running SilentVoice jobs, read from the files they write.

    python scripts/progress.py                       # one snapshot
    python scripts/progress.py --watch               # refresh every 10 s (Ctrl+C to stop)
    python scripts/progress.py --train-config ml/configs/finetune_grid.yaml --eval-log eval.log

Shows: GRID download (zips), preprocessing (crops vs raw videos), fine-tuning (optimizer
steps vs the planned total, ETA, validation WER per epoch) and evaluation (clips decoded,
parsed from the evaluate.py log). It only reads files; it never touches a running job.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID_ZIPS = 34  # 33 talker video zips + alignments.zip


def bar(done: float, total: float, width: int = 30) -> str:
    frac = 0.0 if total <= 0 else min(1.0, done / total)
    fill = int(round(frac * width))
    return f"[{'#' * fill}{'.' * (width - fill)}] {frac * 100:5.1f}%"


def fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def download(raw: Path) -> str:
    zips = raw / "_zips"
    if not zips.is_dir():
        return "download      not started"
    done = len(list(zips.glob("*.zip")))
    parts = sorted(p.name.removesuffix(".zip.part") for p in zips.glob("*.part"))
    size_gb = sum(p.stat().st_size for p in zips.iterdir() if p.is_file()) / 1024**3
    extra = f"  active: {', '.join(parts)}" if parts else ""
    return f"download      {bar(done, GRID_ZIPS)} {done}/{GRID_ZIPS} zips, {size_gb:.1f} GB{extra}"


def preprocessing(raw: Path, processed: Path) -> str:
    total = sum(1 for _ in raw.glob("s*/*.mpg")) if raw.is_dir() else 0
    done = sum(1 for _ in (processed / "videos").glob("s*/*.mp4")) if (processed / "videos").is_dir() else 0
    note = "  (counts extracted videos only)" if total else ""
    return f"preprocess    {bar(done, total)} {done}/{total} clips{note}"


def training(cfg_path: Path) -> list[str]:
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf8"))
    out = ROOT / cfg["output_dir"]
    log = out / "train_log.jsonl"
    name = cfg.get("experiment", cfg_path.stem)
    if not log.is_file():
        return [f"training      {name}: not started ({log.relative_to(ROOT)} missing)"]
    rows = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines() if line.strip()]
    steps = [r for r in rows if "loss" in r]
    vals = [r for r in rows if r.get("type") == "val"]
    t = cfg["train"]
    manifest = ROOT / cfg["manifest"]
    n_train = sum(1 for line in manifest.read_text(encoding="utf8").splitlines() if '"split": "train"' in line) \
        if manifest.is_file() else 0
    per_epoch = max(1, math.ceil(n_train / t["max_batch_size"]) // t["grad_accum_steps"])
    total = per_epoch * t["max_epochs"]
    last = steps[-1] if steps else {"step": 0, "elapsed_s": 0.0, "loss": float("nan")}
    rate = last["elapsed_s"] / last["step"] if last["step"] else None
    eta = rate * (total - last["step"]) if rate else None
    summary = out / "training_summary.json"
    status = "FINISHED" if summary.is_file() and summary.stat().st_mtime >= log.stat().st_mtime - 5 else "running"
    lines = [f"training      {bar(last['step'], total)} step {last['step']}/{total} "
             f"(epoch {len(vals)}/{t['max_epochs']}, max; early stop may end sooner)  "
             f"loss {last['loss']:.3f}  ETA {fmt_time(eta)}  [{status}]"]
    for v in vals:
        lines.append(f"              epoch {v['epoch']}: val WER {v['val_wer']:.4f}  val loss {v['val_loss']:.3f}")
    return lines


def evaluation(log_path: Path) -> str:
    if not log_path.is_file():
        return f"evaluation    {log_path.name}: no log"
    text = log_path.read_text(encoding="utf8", errors="replace")
    final = re.findall(r"(\w+) WER=([\d.]+) CER=([\d.]+) on (\d+)", text)
    if final:
        status, wer, cer, n = final[-1]
        return f"evaluation    {bar(1, 1)} {n} clips  WER {float(wer):.4f}  CER {float(cer):.4f}  [{status}]"
    prog = re.findall(r"(\d+)/(\d+)\s+running WER ([\d.]+)", text)
    if not prog:
        return f"evaluation    {log_path.name}: starting"
    done, total, wer = prog[-1]
    return f"evaluation    {bar(int(done), int(total))} {done}/{total} clips  running WER {float(wer):.4f}"


def snapshot(args: argparse.Namespace) -> str:
    lines = [time.strftime("SilentVoice progress  %H:%M:%S"), ""]
    lines.append(download(args.raw))
    lines.append(preprocessing(args.raw, args.processed))
    for cfg in args.train_config:
        lines.extend(training(cfg))
    for log in args.eval_log:
        lines.append(evaluation(log))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw/grid")
    p.add_argument("--processed", type=Path, default=ROOT / "data/processed/grid")
    p.add_argument("--train-config", type=Path, nargs="*", default=[ROOT / "ml/configs/finetune_grid.yaml"])
    p.add_argument("--eval-log", type=Path, nargs="*", default=[], help="stdout/stderr log of evaluate.py")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=10.0)
    args = p.parse_args()
    if not args.watch:
        print(snapshot(args))
        return 0
    try:
        while True:
            sys.stdout.write("\033[2J\033[H" + snapshot(args) + "\n\n(Ctrl+C to stop)\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
