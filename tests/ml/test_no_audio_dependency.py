"""The recognition pipeline must not depend on the audio stream.

1. Static: no module in the inference path imports an audio library.
2. Dynamic: the same video with no audio, a 300 Hz tone and a 3 kHz tone produces
   bit-identical decoded frames, identical model input and identical model output.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.preprocessing.mouth import MouthCropper
from ml.preprocessing.transforms import to_model_input
from ml.preprocessing.video_io import read_video_rgb
from tests.conftest import fake_landmarks, mux_audio

REPO = Path(__file__).resolve().parents[2]
AUDIO_MODULES = {"torchaudio", "librosa", "soundfile", "pydub", "scipy.io.wavfile", "speech_recognition", "whisper", "av"}
INFERENCE_PATH = [
    REPO / "ml" / "preprocessing" / "video_io.py",
    REPO / "ml" / "preprocessing" / "mouth.py",
    REPO / "ml" / "preprocessing" / "transforms.py",
    REPO / "ml" / "inference" / "model.py",
    REPO / "ml" / "inference" / "pipeline.py",
    *sorted((REPO / "backend" / "app" / "services").glob("*.py")),
]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", INFERENCE_PATH, ids=lambda p: p.name)
def test_no_audio_imports_in_inference_path(path):
    bad = {n for n in _imports(path) if any(n == m or n.startswith(m + ".") for m in AUDIO_MODULES)}
    assert not bad, f"{path.name} imports audio libraries: {bad}"


def test_upstream_model_only_receives_video_modality():
    src = (REPO / "ml" / "inference" / "model.py").read_text()
    assert 'E2E(vocab_size, "video"' in src


def test_audio_track_does_not_change_frames_or_prediction(silent_video, tmp_path, random_vsr_model):
    tone_a = mux_audio(silent_video, tmp_path / "tone300.mp4", 300)
    tone_b = mux_audio(silent_video, tmp_path / "tone3000.mp4", 3000)
    decoded = [read_video_rgb(p)[0] for p in (silent_video, tone_a, tone_b)]
    assert decoded[0].shape[0] > 0
    for other in decoded[1:]:
        assert np.array_equal(decoded[0], other)

    cropper = MouthCropper()
    inputs = [to_model_input(cropper.crop(f, fake_landmarks(len(f))).frames) for f in decoded]
    for other in inputs[1:]:
        assert torch.equal(inputs[0], other)

    outs = [random_vsr_model.transcribe(x[:20]).token_ids for x in inputs]
    assert outs[0] == outs[1] == outs[2]
