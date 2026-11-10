"""Shared fixtures: synthetic videos only (no real people, no datasets)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.preprocessing.video_io import write_video_rgb


def make_frames(t: int = 30, h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, w, dtype=np.float32)[None, None, :, None]
    frames = np.broadcast_to(base, (t, h, w, 3)).copy()
    frames += rng.normal(0, 20, size=frames.shape)
    return np.clip(frames, 0, 255).astype(np.uint8)


def fake_landmarks(t: int, h: int = 240, w: int = 320) -> list[np.ndarray]:
    """MediaPipe-style 4 keypoints per frame: right eye, left eye, nose tip, mouth centre."""
    cx, cy = w / 2, h / 2
    pts = np.array([[cx - 30, cy - 30], [cx + 30, cy - 30], [cx, cy], [cx, cy + 30]], dtype=np.float64)
    return [pts.copy() for _ in range(t)]


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def mux_audio(video: Path, out: Path, freq: int) -> Path:
    """Attach a synthetic sine-tone audio track without re-encoding the video stream."""
    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=2",
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )
    return out


@pytest.fixture
def silent_video(tmp_path: Path) -> Path:
    path = tmp_path / "silent.mp4"
    write_video_rgb(path, make_frames(40), fps=25.0)
    return path


@pytest.fixture(scope="session")
def random_vsr_model():
    """Random-weight model, built once (250M params; ~1 GB RAM)."""
    from ml.inference.model import DecodeConfig, VSRModel

    torch.manual_seed(0)
    return VSRModel(None, torch.device("cpu"), DecodeConfig(beam_size=3), allow_random_init=True)
