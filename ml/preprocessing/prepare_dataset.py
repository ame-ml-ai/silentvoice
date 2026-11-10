"""Preprocess a raw dataset into 96x96 mouth-crop clips + a JSONL manifest.

CPU-only (MediaPipe), so it can run on either machine. Examples::

    python -m ml.preprocessing.prepare_dataset grid \
        --raw-dir data/raw/grid --out-dir data/processed/grid \
        --split-config ml/configs/splits/grid.yaml --workers 6

    python -m ml.preprocessing.prepare_dataset lrs3 \
        --raw-dir data/raw/lrs3 --out-dir data/processed/lrs3 --workers 6

Clips that fail (no face, crop out of bounds, undecodable) are written to
failures.jsonl with the reason, never silently dropped. Re-running resumes: clips
whose output already exists are not reprocessed.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from ml.data.manifest import ManifestEntry, validate_splits, write_manifest
from ml.data.sources import RawClip, iter_grid, iter_lrs3

log = logging.getLogger("prepare_dataset")
_cropper = None


def _init_worker() -> None:
    global _cropper
    from ml.preprocessing.mouth import MouthCropper

    _cropper = MouthCropper()


def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _process(args: tuple[RawClip, str]) -> dict[str, object]:
    from ml.preprocessing.mouth import MouthCropError
    from ml.preprocessing.video_io import VideoDecodeError, read_video_rgb, write_video_rgb

    clip, out_root = args
    out_path = Path(out_root) / clip.rel_out
    try:
        if out_path.is_file() and (n := _frame_count(out_path)) > 0:
            return {"ok": True, "clip": clip, "num_frames": n, "resumed": True}
        frames, _info = read_video_rgb(clip.video_path)
        assert _cropper is not None
        crop = _cropper.crop(frames)
        write_video_rgb(out_path, crop.frames)
        return {"ok": True, "clip": clip, "num_frames": int(crop.frames.shape[0]), "detection_rate": crop.detection_rate}
    except (MouthCropError, VideoDecodeError) as exc:
        return {"ok": False, "clip": clip, "reason": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["grid", "lrs3"])
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split-config", type=Path, default=Path("ml/configs/splits/grid.yaml"))
    parser.add_argument("--lrs3-val-fraction", type=float, default=0.02)
    parser.add_argument("--speakers", help="comma-separated subset of speakers (GRID)")
    parser.add_argument("--limit", type=int, help="process only the first N clips (smoke tests)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset == "grid":
        clips = list(iter_grid(args.raw_dir, args.split_config))
    else:
        clips = list(iter_lrs3(args.raw_dir, args.lrs3_val_fraction))
    if args.speakers:
        wanted = set(args.speakers.split(","))
        clips = [c for c in clips if c.speaker in wanted]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        log.error("No clips found under %s. See DATASET_GUIDE.md for the expected layout.", args.raw_dir)
        return 1

    video_root = args.out_dir / "videos"
    log.info("Processing %d clips with %d workers -> %s", len(clips), args.workers, video_root)
    start = time.time()
    entries: list[ManifestEntry] = []
    failures: list[dict[str, str]] = []
    jobs = [(c, str(video_root)) for c in clips]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap(_process, jobs, chunksize=4), 1):
            clip: RawClip = res["clip"]  # type: ignore[assignment]
            if res["ok"]:
                entries.append(ManifestEntry(
                    id=clip.id, dataset=clip.dataset, split=clip.split, speaker=clip.speaker,
                    video=clip.rel_out, num_frames=int(res["num_frames"]), text=clip.text,
                ))
            else:
                failures.append({"id": clip.id, "split": clip.split, "video": str(clip.video_path),
                                 "reason": str(res["reason"])})
            if i % 500 == 0:
                log.info("%d/%d done (%d failed)", i, len(jobs), len(failures))

    write_manifest(args.out_dir / "manifest.jsonl", entries)
    with open(args.out_dir / "failures.jsonl", "w") as fh:
        for f in failures:
            fh.write(json.dumps(f) + "\n")
    report = validate_splits(entries, root=video_root)
    summary = {
        "dataset": args.dataset, "raw_dir": str(args.raw_dir), "clips_found": len(clips),
        "clips_ok": len(entries), "clips_failed": len(failures),
        "failed_by_split": {s: sum(f["split"] == s for f in failures) for s in ("train", "val", "test")},
        "elapsed_s": round(time.time() - start, 1),
        "split_report": report.to_dict(), "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out_dir / "prep_report.json").write_text(json.dumps(summary, indent=2))
    log.info("Done: %d ok, %d failed. Split report ok=%s", len(entries), len(failures), report.ok)
    return 0 if report.ok else 2


if __name__ == "__main__":
    sys.exit(main())
