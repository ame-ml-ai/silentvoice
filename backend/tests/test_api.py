"""API tests. Synthetic videos and temporary experiment folders only."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from ml.preprocessing.video_io import write_video_rgb
from tests.conftest import fake_landmarks, make_frames


def make_client(tmp_path: Path, **overrides) -> TestClient:
    settings = Settings(experiments_dir=tmp_path / "experiments", upload_dir=tmp_path / "uploads",
                        checkpoint_path=None, **overrides)
    return TestClient(create_app(settings))


def video_bytes(tmp_path: Path, frames: int = 40, name: str = "clip.mp4") -> bytes:
    path = tmp_path / name
    write_video_rgb(path, make_frames(frames), fps=25.0)
    return path.read_bytes()


def wait_for(client: TestClient, job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_health_and_model_info_without_model(tmp_path):
    with make_client(tmp_path) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok" and health["model_state"] == "not_configured"
        info = client.get("/api/model/info").json()
        assert info["state"] == "not_configured"
        assert info["input_modality"] == "video" and info["checkpoint"] is None
        assert info["parameters"]["total"] == 250_383_410  # architecture counts (meta device)
        assert [s["name"] for s in info["architecture"]][:3] == ["Input", "Visual frontend", "Projection"]


def test_recognize_without_model_is_503(tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/api/recognize", files={"file": ("a.mp4", video_bytes(tmp_path), "video/mp4")})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_unavailable"


@pytest.mark.parametrize("name,content,status,code", [
    ("notes.txt", b"hello", 415, "unsupported_format"),
    ("empty.mp4", b"", 400, "empty_file"),
    ("garbage.mp4", b"\x00" * 2048, 422, "invalid_video"),
])
def test_upload_validation_errors(tmp_path, name, content, status, code):
    with make_client(tmp_path) as client:
        r = client.post("/api/preprocess", files={"file": (name, content, "application/octet-stream")})
        assert r.status_code == status and r.json()["error"]["code"] == code
    assert not any((tmp_path / "uploads").glob("*")), "rejected uploads must be deleted"


def test_upload_size_limit(tmp_path):
    with make_client(tmp_path, max_upload_mb=1) as client:
        r = client.post("/api/preprocess", files={"file": ("big.mp4", b"\x00" * (2 * 1024 * 1024), "video/mp4")})
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_duration_limits(tmp_path):
    with make_client(tmp_path, max_duration_s=1.0) as client:
        r = client.post("/api/preprocess", files={"file": ("long.mp4", video_bytes(tmp_path, 50), "video/mp4")})
        assert r.json()["error"]["code"] == "video_too_long"
        r = client.post("/api/preprocess", files={"file": ("s.mp4", video_bytes(tmp_path, 3, "s.mp4"), "video/mp4")})
        assert r.json()["error"]["code"] == "video_too_short"


def test_preprocess_job_reports_no_face(tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/api/preprocess", files={"file": ("clip.mp4", video_bytes(tmp_path), "video/mp4")})
        assert r.status_code == 202
        created = r.json()
        assert created["video"]["output_frames"] == 40 and created["kind"] == "preprocess"
        done = wait_for(client, created["job_id"])
        assert done["status"] == "failed" and done["error"]["code"] == "no_face"
        assert done["text"] is None


def test_unknown_job_404(tmp_path):
    with make_client(tmp_path) as client:
        r = client.get("/api/jobs/doesnotexist")
        assert r.status_code == 404 and r.json()["error"]["code"] == "job_not_found"


def test_unknown_route_and_method_use_error_envelope(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/api/nope").json()["error"]["code"] == "not_found"
        assert client.get("/api/recognize").json()["error"]["code"] == "method_not_allowed"


def test_recognize_rejected_before_body_is_read(tmp_path):
    """The 503 comes from middleware, so even an invalid body is never parsed."""
    with make_client(tmp_path) as client:
        r = client.post("/api/recognize", content=b"not multipart", headers={"content-type": "text/plain"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_queue_full_returns_429_and_cleans_up(tmp_path):
    with make_client(tmp_path) as client:
        client.app.state.job_service.max_active = 0
        r = client.post("/api/preprocess", files={"file": ("clip.mp4", video_bytes(tmp_path), "video/mp4")})
        assert r.status_code == 429 and r.json()["error"]["code"] == "queue_full"
    assert not any((tmp_path / "uploads").glob("*"))


def test_experiments_pending_when_nothing_has_run(tmp_path):
    with make_client(tmp_path) as client:
        exps = client.get("/api/experiments").json()["experiments"]
        assert {e["id"]: e["status"] for e in exps} == {"baseline": "pending", "finetuning": "pending"}
        cmp_ = client.get("/api/experiments/compare").json()
        assert cmp_["status"] == "pending" and len(cmp_["reasons"]) == 2
        assert cmp_["wer_relative_change"] is None


def _write_result(d: Path, status: str, wer: float, cer: float, n: int = 100, name: str = "results.json") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({
        "status": status, "split": "test", "num_samples": n, "wer": wer, "cer": cer,
        "latency": {"mean_s": 0.2, "real_time_factor": 0.07}, "memory": {"peak_gpu_allocated_gb": 1.5},
        "parameters": {"total": 250383410, "trainable": 250383410}, "checkpoint": {"name": "x.pth"},
        "device": {"name": "TEST DEVICE"},
    }))


def test_compare_uses_only_complete_results(tmp_path):
    exp = tmp_path / "experiments"
    _write_result(exp / "baseline", "complete", 0.40, 0.20)
    _write_result(exp / "finetuning", "smoke", 0.10, 0.05, name="results_smoke.json")
    with make_client(tmp_path) as client:
        cmp_ = client.get("/api/experiments/compare").json()
        assert cmp_["status"] == "pending"  # smoke run must not count
        ids = {e["id"]: e["status"] for e in client.get("/api/experiments").json()["experiments"]}
        assert ids["finetuning_smoke"] == "smoke" and ids["finetuning"] == "pending"

    _write_result(exp / "finetuning", "complete", 0.30, 0.15)
    with make_client(tmp_path) as client:
        cmp_ = client.get("/api/experiments/compare").json()
        assert cmp_["status"] == "complete"
        assert cmp_["wer_absolute_change"] == pytest.approx(-0.10)
        assert cmp_["wer_relative_change"] == pytest.approx(-0.25)


def test_compare_refuses_mismatched_protocol(tmp_path):
    exp = tmp_path / "experiments"
    _write_result(exp / "baseline", "complete", 0.40, 0.20, n=100)
    _write_result(exp / "finetuning", "complete", 0.30, 0.15, n=90)
    with make_client(tmp_path) as client:
        cmp_ = client.get("/api/experiments/compare").json()
        assert cmp_["status"] == "pending" and "same split" in cmp_["reasons"][0]


@pytest.mark.slow
def test_full_recognition_with_random_weight_checkpoint(tmp_path, random_vsr_model):
    """End-to-end through the real API with a real (random-weight) checkpoint file.

    The face detector is replaced by fixed landmarks because the synthetic video has
    no face; everything after detection (alignment, crop, transform, model, beam
    search, job lifecycle) is the production code path.
    """
    ckpt = tmp_path / "random.pth"
    torch.save(random_vsr_model.model.state_dict(), ckpt)
    settings = Settings(experiments_dir=tmp_path / "exp", upload_dir=tmp_path / "up", checkpoint_path=ckpt,
                        beam_size=2, device="cpu")
    with TestClient(create_app(settings)) as client:
        for _ in range(300):
            if client.get("/api/health").json()["model_state"] == "ready":
                break
            time.sleep(0.2)
        info = client.get("/api/model/info").json()
        assert info["state"] == "ready" and info["parameters"]["total"] == 250_383_410
        assert info["architecture"][1]["parameters"] == info["parameters"]["frontend"]

        jobs = client.app.state.job_service
        pipeline = jobs._get_pipeline()
        real_crop = pipeline.cropper.crop
        pipeline.cropper.crop = lambda frames, landmarks=None: real_crop(frames, fake_landmarks(len(frames)))

        r = client.post("/api/recognize", files={"file": ("clip.mp4", video_bytes(tmp_path, 30), "video/mp4")})
        assert r.status_code == 202
        done = wait_for(client, r.json()["job_id"])
        assert done["status"] == "completed", done
        assert isinstance(done["text"], str) and done["model_checkpoint"] == "random.pth"
        assert done["mouth_preview_png"] and done["timings"]["model_s"] > 0
