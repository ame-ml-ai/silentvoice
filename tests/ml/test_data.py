import json

import pytest
import torch

from ml.data.dataset import FrameBudgetBatchSampler, collate_pad
from ml.data.manifest import (
    ManifestEntry, SplitLeakageError, assert_no_leakage, read_manifest, validate_splits, write_manifest,
)
from ml.data.sources import load_split_config, parse_grid_alignment, parse_lrs3_text


def entry(i, split, speaker, frames=75):
    return ManifestEntry(f"grid/{speaker}/c{i}", "grid", split, speaker, f"{speaker}/c{i}.mp4", frames, "BIN BLUE")


def test_manifest_roundtrip(tmp_path):
    rows = [entry(1, "train", "s5"), entry(2, "val", "s3")]
    path = tmp_path / "m.jsonl"
    assert write_manifest(path, rows) == 2
    assert read_manifest(path) == rows


def test_manifest_rejects_bad_rows(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({**entry(1, "train", "s5").__dict__, "split": "dev"}) + "\n")
    with pytest.raises(ValueError):
        read_manifest(path)


def test_speaker_leakage_detected():
    rows = [entry(1, "train", "s5"), entry(2, "test", "s5"), entry(3, "val", "s3")]
    report = validate_splits(rows)
    assert report.speaker_overlap == {"train/test": ["s5"]}
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(report)


def test_duplicate_ids_detected():
    rows = [entry(1, "train", "s5"), entry(1, "train", "s5")]
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(validate_splits(rows))


def test_missing_files_reported(tmp_path):
    report = validate_splits([entry(1, "train", "s5")], root=tmp_path)
    assert report.missing_files == ["grid/s5/c1"] and not report.ok


def test_grid_split_config_is_speaker_disjoint():
    splits = load_split_config("ml/configs/splits/grid.yaml")
    all_spk = [s for v in splits.values() for s in v]
    assert len(all_spk) == len(set(all_spk)) == 33  # 34 talkers minus s21 (no video)
    assert "s21" not in all_spk
    assert splits["test"] == ["s1", "s2", "s20", "s22"]


def test_split_config_rejects_overlap(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("train: [s1, s2]\nval: [s3]\ntest: [s2]\n")
    with pytest.raises(ValueError):
        load_split_config(cfg)


def test_parse_grid_alignment(tmp_path):
    f = tmp_path / "bbaf2n.align"
    f.write_text("0 23750 sil\n23750 29500 bin\n29500 34000 blue\n34000 35500 at\n"
                 "35500 41000 f\n41000 47250 two\n47250 53000 now\n53000 74500 sil\n")
    assert parse_grid_alignment(f) == "BIN BLUE AT F TWO NOW"


def test_parse_lrs3_text(tmp_path):
    f = tmp_path / "00001.txt"
    f.write_text("Text:  AND THAT'S WHAT {LG} WE DID\nConf:  4\n\nWORD START END ASDSCORE\n")
    assert parse_lrs3_text(f) == "AND THAT'S WHAT WE DID"


def test_collate_pads_inputs_and_targets():
    batch = [
        {"input": torch.ones(3, 1, 88, 88), "target": torch.tensor([5, 6]), "id": "a", "text": "A"},
        {"input": torch.ones(5, 1, 88, 88), "target": torch.tensor([7]), "id": "b", "text": "B"},
    ]
    out = collate_pad(batch)
    assert out["inputs"].shape == (2, 5, 1, 88, 88)
    assert out["inputs"][0, 3:].abs().sum() == 0
    assert out["targets"].tolist() == [[5, 6], [7, -1]]
    assert out["input_lengths"].tolist() == [3, 5]


def test_sampler_mixes_equal_length_clips_across_speakers_and_epochs():
    # 4 speakers x 10 clips, all 75 frames, in speaker-contiguous manifest order (like GRID)
    lengths = [75] * 40
    speaker = [i // 10 for i in range(40)]
    sampler = FrameBudgetBatchSampler(lengths, max_frames=600, shuffle=True, seed=0)
    first = list(sampler)
    assert any(len({speaker[i] for i in b}) > 1 for b in first), "batches must not be single-speaker"
    sampler.set_epoch(1)
    assert [sorted(b) for b in sampler] != [sorted(b) for b in first], "batch composition must change per epoch"


def test_frame_budget_sampler_respects_budget():
    lengths = [75] * 20 + [40] * 10
    sampler = FrameBudgetBatchSampler(lengths, max_frames=300, shuffle=True, seed=0)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(30))
    assert all(sum(lengths[i] for i in b) <= 300 for b in batches)
    with pytest.raises(ValueError):
        FrameBudgetBatchSampler([500], max_frames=300, shuffle=False)
