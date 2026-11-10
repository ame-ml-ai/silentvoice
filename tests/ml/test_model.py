import pytest
import torch

from ml.inference.model import CheckpointError, VSRModel, count_parameters, load_state_dict
from ml.training.freeze import apply_freeze
from ml.utils.device import DeviceUnavailableError, resolve_device


def test_parameter_count_matches_model_zoo(random_vsr_model):
    counts = count_parameters(random_vsr_model.model)
    assert counts["total"] == 250_383_410  # README model zoo: "250M"
    assert counts["encoder"] == 170_451_456 and counts["decoder"] == 64_472_505


def test_encoder_output_shape(random_vsr_model):
    feats = random_vsr_model.encode(torch.randn(25, 1, 88, 88))
    assert feats.shape == (25, 768)


def test_rejects_wrong_input_shape(random_vsr_model):
    with pytest.raises(ValueError):
        random_vsr_model.encode(torch.randn(25, 3, 96, 96))


def test_transcription_is_deterministic(random_vsr_model):
    x = torch.randn(15, 1, 88, 88, generator=torch.Generator().manual_seed(3))
    assert random_vsr_model.transcribe(x).token_ids == random_vsr_model.transcribe(x).token_ids


def test_checkpoint_roundtrip_and_formats(tmp_path, random_vsr_model):
    sd = random_vsr_model.model.state_dict()
    plain = tmp_path / "plain.pth"
    torch.save(sd, plain)
    wrapped = tmp_path / "wrapped.pth"
    torch.save({"model_state_dict": sd, "epoch": 1}, wrapped)
    for path in (plain, wrapped):
        loaded = load_state_dict(path)
        assert loaded.keys() == sd.keys()
    m = VSRModel(plain, torch.device("cpu"))
    assert torch.equal(next(iter(m.model.state_dict().values())), next(iter(sd.values())))


def test_incompatible_or_missing_checkpoint(tmp_path):
    with pytest.raises(CheckpointError):
        VSRModel(tmp_path / "nope.pth", torch.device("cpu"))
    bad = tmp_path / "bad.pth"
    torch.save({"weight": torch.zeros(3)}, bad)
    with pytest.raises(CheckpointError):
        VSRModel(bad, torch.device("cpu"))
    with pytest.raises(CheckpointError):
        VSRModel(None, torch.device("cpu"))


def test_device_resolution_never_silently_downgrades():
    with pytest.raises(DeviceUnavailableError):
        resolve_device("mps")
    if not torch.cuda.is_available():
        with pytest.raises(DeviceUnavailableError):
            resolve_device("cuda")
        assert resolve_device("auto").type == "cpu"


@pytest.mark.parametrize("strategy,expected_min,expected_max", [
    ("decoder_only", 60_000_000, 70_000_000),
    ("encoder_top_k", 110_000_000, 130_000_000),
    ("full", 250_383_410, 250_383_410),
])
def test_freeze_strategies(random_vsr_model, strategy, expected_min, expected_max):
    model = random_vsr_model.model
    try:
        apply_freeze(model, strategy, 4)
        trainable = count_parameters(model)["trainable"]
        assert expected_min <= trainable <= expected_max
        if strategy != "full":
            assert not any(p.requires_grad for p in model.frontend.parameters())
    finally:
        for p in model.parameters():
            p.requires_grad = True
