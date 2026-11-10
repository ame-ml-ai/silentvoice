"""Face detection and mouth-region extraction.

Wraps Auto-AVSR's own MediaPipe detector and ``VideoProcess`` (similarity alignment
to a mean face, temporal landmark smoothing, 96x96 crop around the mouth) without
modification, so crops match the preprocessing used to train the checkpoints.

Only MediaPipe is wired up here: it runs on CPU on both machines. Auto-AVSR's
RetinaFace path needs the separate ``ibug`` face packages and a GPU; LRS3 users can
instead pass the precomputed landmarks Auto-AVSR publishes (``landmarks=``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.utils.upstream import ensure_auto_avsr_on_path

CROP_SIZE = 96


class MouthCropError(ValueError):
    """No face was found, or the mouth region could not be cropped."""


@dataclass(frozen=True)
class MouthCropResult:
    frames: np.ndarray  # (T, 96, 96, 3) uint8 RGB
    detection_rate: float  # fraction of frames with a detected face (before interpolation)


class MouthCropper:
    def __init__(self) -> None:
        ensure_auto_avsr_on_path()
        from preparation.detectors.mediapipe.detector import LandmarksDetector
        from preparation.detectors.mediapipe.video_process import VideoProcess

        self._detector = LandmarksDetector()
        # convert_gray=False exactly as in Auto-AVSR's preprocessing and inference
        # tutorial; grayscale conversion happens later in the model transform.
        self._process = VideoProcess(convert_gray=False)

    def detect(self, frames: np.ndarray) -> list[np.ndarray | None]:
        try:
            return self._detector(frames)
        except AssertionError as exc:  # upstream signals "no face in any frame" this way
            raise MouthCropError("No face was detected in any frame of the video.") from exc

    def crop(self, frames: np.ndarray, landmarks: list[np.ndarray | None] | None = None) -> MouthCropResult:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise MouthCropError(f"Expected RGB frames (T, H, W, 3), got shape {frames.shape}.")
        if landmarks is None:
            landmarks = self.detect(frames)
        if len(landmarks) != len(frames):
            raise MouthCropError("Landmark count does not match frame count.")
        detected = sum(lm is not None for lm in landmarks)
        if detected == 0:
            raise MouthCropError("No face was detected in any frame of the video.")
        try:
            # upstream mutates the list while interpolating; pass a copy
            patches = self._process(frames, list(landmarks))
        except (OverflowError, AssertionError) as exc:
            raise MouthCropError(
                "The mouth region could not be cropped (face too close to the frame edge)."
            ) from exc
        if patches is None or len(patches) != len(frames):
            raise MouthCropError("Mouth cropping produced an empty sequence.")
        patches = np.asarray(patches, dtype=np.uint8)
        if patches.shape[1:] != (CROP_SIZE, CROP_SIZE, 3):
            raise MouthCropError(f"Unexpected crop shape {patches.shape}.")
        return MouthCropResult(frames=patches, detection_rate=detected / len(frames))
