"""Upload handling: size/extension limits, temp files, and decode-level validation."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.core.config import Settings
from backend.app.core.errors import ApiError
from ml.preprocessing.video_io import TARGET_FPS, VideoDecodeError, VideoInfo, VideoTooLongError, read_video_rgb

CHUNK = 1024 * 1024


async def save_upload(file: UploadFile, settings: Settings) -> tuple[Path, int]:
    """Stream the upload to a temp file, enforcing extension and size limits."""
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ApiError(415, "unsupported_format",
                       f"'{ext or name}' is not supported. Use one of: {', '.join(settings.allowed_extensions)}.")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise ApiError(413, "file_too_large", f"File exceeds the {settings.max_upload_mb} MB limit.")
                await run_in_threadpool(out.write, chunk)
    except BaseException:  # includes client disconnects and cancellation
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    return dest, size


UPLOAD_MAX_SIDE = 960  # frames are downscaled while decoding so memory stays bounded


def validate_video(path: Path, settings: Settings) -> VideoInfo:
    """Decode the video stream once (blocking: call via a thread) to check it is
    readable and within limits. Memory is bounded: frames are resampled to 25 fps and
    downscaled while decoding, and decoding stops at the duration limit."""
    max_frames = int(settings.max_duration_s * TARGET_FPS)
    try:
        _, info = read_video_rgb(path, max_frames=max_frames, max_side=UPLOAD_MAX_SIDE)
    except VideoTooLongError as exc:
        raise ApiError(422, "video_too_long", f"Video is longer than the {settings.max_duration_s:.0f}s limit.") from exc
    except VideoDecodeError as exc:
        raise ApiError(422, "invalid_video", str(exc)) from exc
    if info.output_frames < int(settings.min_duration_s * TARGET_FPS):
        raise ApiError(422, "video_too_short", f"Video must be at least {settings.min_duration_s}s long.")
    return info
