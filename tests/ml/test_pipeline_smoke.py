"""CPU smoke tests for the evaluation and training entry points.

They use synthetic clips and a random-weight checkpoint saved to a temp dir, so they
exercise the real code paths (manifest -> dataset -> model -> metrics / optimizer)
without any dataset or downloaded checkpoint. Their numbers are meaningless.
"""

import json

import pytest
import torch
import yaml

from ml.data.manifest import ManifestEntry, write_manifest
from ml.preprocessing.video_io import write_video_rgb
from tests.conftest import make_frames


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / "videos"
    rows = []
    for i, (split, spk) in enumerate([("train", "s5")] * 4 + [("val", "s3")] * 2 + [("test", "s1")] * 2):
        rel = f"{spk}/clip{i}.mp4"
        write_video_rgb(root / rel, make_frames(20, 96, 96, seed=i))
        rows.append(ManifestEntry(f"grid/{spk}/clip{i}", "grid", split, spk, rel, 20, "BIN BLUE AT F TWO NOW"))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    return tmp_path, manifest, root


@pytest.mark.slow
def test_evaluate_smoke(tiny_dataset, random_vsr_model):
    from ml.evaluation import evaluate

    tmp, manifest, root = tiny_dataset
    ckpt = tmp / "random.pth"
    torch.save(random_vsr_model.model.state_dict(), ckpt)
    cfg = {"experiment": "smoke", "output_dir": str(tmp / "out"), "seed": 1, "checkpoint": str(ckpt),
           "manifest": str(manifest), "data_root": str(root), "split": "test", "device": "cpu",
           "decode": {"beam_size": 2, "ctc_weight": 0.1, "length_penalty": 0.0}}
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert evaluate.main(["--config", str(cfg_path), "--limit", "2"]) == 0
    res = json.loads((tmp / "out" / "results_smoke.json").read_text())
    assert res["status"] == "smoke" and res["num_samples"] == 2
    assert 0.0 <= res["cer"] and res["parameters"]["total"] == 250_383_410
    assert not (tmp / "out" / "results.json").exists()  # smoke never writes the real result file


@pytest.mark.slow
def test_finetune_cpu_smoke(tiny_dataset):
    from ml.training import finetune

    tmp, manifest, root = tiny_dataset
    cfg = yaml.safe_load(open("ml/configs/finetune_grid.yaml"))
    cfg.update(output_dir=str(tmp / "ft"), manifest=str(manifest), data_root=str(root))
    cfg["freeze"]["strategy"] = "decoder_only"
    cfg_path = tmp / "ft.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert finetune.main(["--config", str(cfg_path), "--cpu-smoke", "--random-init", "--max-steps", "1"]) == 0
    summary = json.loads((tmp / "ft" / "smoke" / "training_summary.json").read_text())
    assert summary["status"] == "smoke" and summary["optimizer_steps"] == 1
    assert summary["parameters"]["trainable"] < summary["parameters"]["total"]
    assert (tmp / "ft" / "smoke" / "checkpoints" / "best.pth").exists()


def test_finetune_refuses_without_cuda(tiny_dataset):
    from ml.training import finetune

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    tmp, manifest, root = tiny_dataset
    cfg = yaml.safe_load(open("ml/configs/finetune_grid.yaml"))
    cfg.update(output_dir=str(tmp / "ft"), manifest=str(manifest), data_root=str(root))
    cfg_path = tmp / "ft.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert finetune.main(["--config", str(cfg_path)]) == 3
    assert not (tmp / "ft" / "training_summary.json").exists()
