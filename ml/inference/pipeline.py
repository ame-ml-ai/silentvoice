"""End-to-end visual-only pipeline: video file -> mouth crops -> model -> text.

    Video file
      -> OpenCV decode (video stream only), resample to 25 fps
      -> MediaPipe face detection (4 keypoints per frame) + interpolation
      -> similarity alignment to mean face, 96x96 mouth crop      (Auto-AVSR code)
      -> /255, center crop 88, grayscale, normalise               (Auto-AVSR transform)
      -> 3D-conv + ResNet-18 -> Conformer encoder
      -> joint CTC/attention beam search -> SentencePiece detokenise -> text

No audio is read at any step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ml.inference.model import VSRModel
from ml.preprocessing.mouth import MouthCropper
from ml.preprocessing.transforms import to_model_input
from ml.preprocessing.video_io import VideoInfo, read_video_rgb


@dataclass
class PreprocessResult:
    video_info: VideoInfo
    mouth_frames: np.ndarray  # (T, 96, 96, 3) uint8
    model_input: torch.Tensor  # (T, 1, 88, 88) float
    detection_rate: float
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class RecognitionResult:
    text: str
    preprocess: PreprocessResult
    timings: dict[str, float]


class VisualSpeechPipeline:
    def __init__(
        self, cropper: MouthCropper, model: VSRModel | None = None, max_frames: int = 25 * 30,
        max_side: int | None = 960,
    ):
        self.cropper = cropper
        self.model = model
        self.max_frames = max_frames
        self.max_side = max_side  # bounds memory for large uploads; faces stay far above crop size

    def preprocess(self, video_path: str | Path, landmarks: list[np.ndarray | None] | None = None) -> PreprocessResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        frames, info = read_video_rgb(video_path, max_frames=self.max_frames, max_side=self.max_side)
        timings["decode_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        crop = self.cropper.crop(frames, landmarks)
        timings["mouth_crop_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        model_input = to_model_input(crop.frames)
        timings["transform_s"] = time.perf_counter() - t2
        return PreprocessResult(info, crop.frames, model_input, crop.detection_rate, timings)

    def recognize(self, video_path: str | Path) -> RecognitionResult:
        if self.model is None:
            raise RuntimeError("No model loaded; recognition is unavailable.")
        pre = self.preprocess(video_path)
        out = self.model.transcribe(pre.model_input)
        timings = {**pre.timings, "model_s": out.inference_seconds}
        timings["total_s"] = sum(timings.values())
        return RecognitionResult(out.text, pre, timings)
