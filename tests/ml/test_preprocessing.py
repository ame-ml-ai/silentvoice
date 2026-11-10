import numpy as np
import pytest
import torch
import torchvision.transforms as T

from ml.preprocessing.mouth import MouthCropError, MouthCropper
from ml.preprocessing.transforms import MEAN, STD, to_model_input, train_transform, frames_to_tensor
from ml.preprocessing.video_io import VideoDecodeError, read_video_rgb, write_video_rgb
from tests.conftest import fake_landmarks, make_frames


def test_video_roundtrip_keeps_25fps(silent_video):
    frames, info = read_video_rgb(silent_video)
    assert frames.shape == (40, 240, 320, 3) and frames.dtype == np.uint8
    assert info.output_fps == 25.0 and not info.resampled
    assert info.source_fps == pytest.approx(25.0, abs=0.1)


def test_30fps_video_is_resampled_to_25(tmp_path):
    path = tmp_path / "v30.mp4"
    write_video_rgb(path, make_frames(60), fps=30.0)  # 2.0 s
    frames, info = read_video_rgb(path)
    assert info.resampled
    assert info.output_frames == 50 == frames.shape[0]


def test_corrupt_or_empty_file_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(VideoDecodeError):
        read_video_rgb(bad)


def test_max_frames_limit(tmp_path):
    path = tmp_path / "long.mp4"
    write_video_rgb(path, make_frames(60), fps=25.0)
    with pytest.raises(VideoDecodeError):
        read_video_rgb(path, max_frames=30)


def test_model_input_matches_reference_formula():
    rng = np.random.default_rng(1)
    crops = rng.integers(0, 256, size=(5, 96, 96, 3), dtype=np.uint8)
    x = to_model_input(crops)
    assert x.shape == (5, 1, 88, 88)
    # reference: torchvision ops written out explicitly
    ref = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255.0
    ref = T.functional.rgb_to_grayscale(ref[:, :, 4:92, 4:92])
    ref = (ref - MEAN) / STD
    assert torch.allclose(x, ref, atol=1e-5)


def test_train_transform_shapes():
    crops = make_frames(30, 96, 96)
    out = train_transform()(frames_to_tensor(crops))
    assert out.shape == (30, 1, 88, 88)


@pytest.fixture(scope="module")
def cropper():
    return MouthCropper()


def test_mouth_crop_with_given_landmarks(cropper):
    frames = make_frames(20)
    result = cropper.crop(frames, fake_landmarks(20))
    assert result.frames.shape == (20, 96, 96, 3)
    assert result.detection_rate == 1.0


def test_missing_landmarks_are_interpolated(cropper):
    lms = fake_landmarks(20)
    for i in (3, 4, 5, 15):
        lms[i] = None
    result = cropper.crop(make_frames(20), lms)
    assert result.frames.shape[0] == 20
    assert result.detection_rate == pytest.approx(16 / 20)


def test_no_face_raises(cropper):
    blank = np.zeros((10, 240, 320, 3), dtype=np.uint8)
    with pytest.raises(MouthCropError):
        cropper.crop(blank)
