"""Video decoding for the visual-only pipeline.

OpenCV's ``VideoCapture`` demuxes and decodes the *video stream only*; any audio
track in the container is never decoded or returned. This is the first of the
guarantees that the recognition model receives no audio information
(see tests/ml/test_no_audio_dependency.py).

Auto-AVSR checkpoints were trained on 25 fps video. Inputs whose container declares
~25 fps are kept frame-for-frame (exactly like upstream). Anything else (30/60 fps
phone video, browser recordings with bogus declared rates) is resampled to 25 fps by
nearest-previous presentation timestamp, *while decoding*, so memory only ever holds
output frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

TARGET_FPS = 25.0
_FPS_TOLERANCE = 0.5


class VideoDecodeError(ValueError):
    """The file could not be opened or contains no decodable video frames."""


class VideoTooLongError(VideoDecodeError):
    """The video exceeds the configured output-frame limit."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    source_fps: float
    source_frames: int
    duration_s: float
    output_fps: float
    output_frames: int
    resampled: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _downscale(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None or max(frame.shape[:2]) <= max_side:
        return frame
    scale = max_side / max(frame.shape[:2])
    size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def read_video_rgb(
    path: str | Path,
    target_fps: float = TARGET_FPS,
    max_frames: int | None = None,
    max_side: int | None = None,
) -> tuple[np.ndarray, VideoInfo]:
    """Decode a video to RGB uint8 frames (T, H, W, 3) at ``target_fps``.

    ``max_frames`` bounds the number of *output* frames (raises ``VideoTooLongError``).
    ``max_side`` downscales frames whose longer side exceeds it (used for uploads to
    bound memory; the dataset pipeline leaves it unset to match Auto-AVSR exactly).
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoDecodeError(f"Could not open video '{path.name}'.")
    declared_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    declared_valid = 1.0 <= declared_fps <= 240.0
    fallback_period = 1.0 / (declared_fps if declared_valid else target_fps)
    keep_all = declared_valid and abs(declared_fps - target_fps) <= _FPS_TOLERANCE

    out: list[np.ndarray] = []
    prev: np.ndarray | None = None
    t0: float | None = None
    last_raw = last_rel = 0.0
    n_src = 0

    def check_length() -> None:
        if max_frames is not None and len(out) > max_frames:
            raise VideoTooLongError(f"Video is longer than {max_frames / target_fps:.0f} s.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            raw = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if n_src == 0:
                t0, rel = raw, 0.0
            elif raw > last_raw:
                rel = raw - t0  # type: ignore[operator]
            else:  # missing / non-monotonic timestamps: assume a constant frame period
                rel = last_rel + fallback_period
            frame = _downscale(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), max_side)
            if keep_all:
                out.append(frame)
            else:
                # every target time strictly before this frame shows the previous frame
                while prev is not None and len(out) / target_fps < rel:
                    out.append(prev)
            check_length()
            prev, last_raw, last_rel = frame, raw, rel
            n_src += 1
    finally:
        cap.release()
    if prev is None:
        raise VideoDecodeError(f"'{path.name}' contains no decodable video frames.")

    period = last_rel / (n_src - 1) if n_src >= 2 and last_rel > 0 else fallback_period
    duration = last_rel + period
    if not keep_all:
        target_count = max(1, round(duration * target_fps))
        while len(out) < target_count:
            out.append(prev)
        del out[target_count:]
        check_length()

    video = np.stack(out).astype(np.uint8, copy=False)
    h, w = video.shape[1:3]
    info = VideoInfo(
        width=int(w), height=int(h), source_fps=round(1.0 / period, 3), source_frames=n_src,
        duration_s=round(duration, 3), output_fps=target_fps, output_frames=int(video.shape[0]),
        resampled=not keep_all,
    )
    return video, info


def write_video_rgb(path: str | Path, frames: np.ndarray, fps: float = TARGET_FPS) -> None:
    """Write RGB (T, H, W, 3) or grayscale (T, H, W) uint8 frames as MPEG-4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    t, h, w, _ = frames.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise VideoDecodeError(f"Could not open a video writer for '{path}'.")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
