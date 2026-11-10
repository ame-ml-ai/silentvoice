"""Background jobs for preprocessing and recognition.

A single worker thread runs jobs in order: there is one model, MediaPipe is not
thread-safe, and long CPU/GPU work must not block the event loop. The queue is
bounded; finished jobs are kept (for polling) up to ``max_retained``.
"""

from __future__ import annotations

import base64
import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.services.model_service import ModelService
from ml.preprocessing.mouth import MouthCropError
from ml.preprocessing.video_io import VideoDecodeError, VideoTooLongError

log = logging.getLogger("silentvoice.jobs")
ACTIVE = ("queued", "running")


class QueueFullError(RuntimeError):
    """Too many jobs are waiting."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mouth_preview_png(frames: np.ndarray, n: int = 8) -> str:
    """Base64 PNG of up to n evenly spaced 96x96 mouth crops side by side."""
    idx = np.linspace(0, len(frames) - 1, num=min(n, len(frames))).round().astype(int)
    strip = np.concatenate([frames[i] for i in idx], axis=1)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class Job:
    job_id: str
    kind: str
    video_path: Path
    video: dict[str, Any]
    status: str = "queued"
    stage: str = "Waiting in queue"
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    detection_rate: float | None = None
    mouth_preview_png: str | None = None
    preview_frames: int | None = None
    text: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    model_checkpoint: str | None = None
    error: dict[str, str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict[str, Any]:
        with self.lock:
            out = {k: v for k, v in self.__dict__.items() if k not in ("video_path", "lock")}
            out["timings"] = dict(self.timings)
            return out

    def update(self, **changes: Any) -> None:
        with self.lock:
            for k, v in changes.items():
                setattr(self, k, v)


class JobService:
    def __init__(self, model_service: ModelService, max_retained: int = 100, max_active: int = 8) -> None:
        self.model_service = model_service
        self.max_retained = max_retained
        self.max_active = max_active
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vsr-worker")
        self._cropper = None
        self._pipeline = None

    def _get_pipeline(self):
        from ml.inference.pipeline import VisualSpeechPipeline
        from ml.preprocessing.mouth import MouthCropper

        if self._cropper is None:
            self._cropper = MouthCropper()
        model = self.model_service.model
        if self._pipeline is None or self._pipeline.model is not model:
            self._pipeline = VisualSpeechPipeline(self._cropper, model)
        return self._pipeline

    def _active_count(self) -> int:
        return sum(j.status in ACTIVE for j in self._jobs.values())

    def is_full(self) -> bool:
        with self._lock:
            return self._active_count() >= self.max_active

    def submit(self, kind: str, video_path: Path, video_meta: dict[str, Any]) -> Job:
        with self._lock:
            if self._active_count() >= self.max_active:
                raise QueueFullError("Too many videos are waiting. Try again when the current jobs finish.")
            job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, video_path=video_path, video=video_meta)
            self._jobs[job.job_id] = job
            # evict only finished jobs, oldest first
            excess = len(self._jobs) - self.max_retained
            for jid in [j for j, v in self._jobs.items() if v.status not in ACTIVE][: max(0, excess)]:
                del self._jobs[jid]
        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _fail(self, job: Job, stage: str, code: str, message: str) -> None:
        job.update(stage=stage, error={"code": code, "message": message}, finished_at=_now())
        job.update(status="failed")  # status last: a poller never sees "failed" without details

    def _run(self, job: Job) -> None:
        job.update(status="running", stage="Detecting face and cropping mouth region")
        try:
            pipeline = self._get_pipeline()
            pre = pipeline.preprocess(job.video_path)
            timings = {k: round(v, 4) for k, v in pre.timings.items()}
            job.update(detection_rate=round(pre.detection_rate, 4), preview_frames=len(pre.mouth_frames),
                       mouth_preview_png=mouth_preview_png(pre.mouth_frames), timings=dict(timings))
            text = checkpoint = None
            if job.kind == "recognize":
                model = self.model_service.model
                if model is None:
                    raise RuntimeError("Model became unavailable before recognition started.")
                job.update(stage="Running visual speech recognition")
                out = model.transcribe(pre.model_input)
                text = out.text
                checkpoint = model.checkpoint_path.name if model.checkpoint_path else None
                timings["model_s"] = round(out.inference_seconds, 4)
            timings["total_s"] = round(sum(timings.values()), 4)
            job.update(text=text, model_checkpoint=checkpoint, timings=timings, stage="Done", finished_at=_now())
            job.update(status="completed")
        except MouthCropError as exc:
            self._fail(job, "Mouth region not found", "no_face", str(exc))
        except VideoTooLongError as exc:
            self._fail(job, "Video too long", "video_too_long", str(exc))
        except VideoDecodeError as exc:
            self._fail(job, "Video could not be decoded", "invalid_video", str(exc))
        except Exception:  # job boundary: log details server-side, return a generic message
            log.exception("Job %s failed", job.job_id)
            self._fail(job, "Unexpected error", "internal_error", "Processing failed unexpectedly; see the server log.")
        finally:
            job.video_path.unlink(missing_ok=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for job in self._jobs.values():
                if job.status == "queued":
                    job.video_path.unlink(missing_ok=True)
