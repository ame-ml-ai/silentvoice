"""Pydantic response schemas (documented in OpenAPI at /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ModelState = Literal["not_configured", "loading", "ready", "error"]
JobStatus = Literal["queued", "running", "completed", "failed"]
JobKind = Literal["preprocess", "recognize"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    model_state: ModelState
    device: str
    cuda_available: bool


class ArchitectureStage(BaseModel):
    name: str
    detail: str
    parameters: int | None = None


class ModelInfoResponse(BaseModel):
    state: ModelState
    message: str
    model_name: str = "Auto-AVSR visual-only (Ma et al., ICASSP 2023)"
    checkpoint: str | None = None
    checkpoint_md5: str | None = None
    device: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    parameters: dict[str, int] | None = None
    vocab_size: int | None = None
    architecture: list[ArchitectureStage]
    upstream: dict[str, str]
    input_modality: Literal["video"] = "video"


class VideoMeta(BaseModel):
    filename: str
    size_bytes: int
    width: int
    height: int
    source_fps: float
    duration_s: float
    output_frames: int
    resampled: bool


class JobCreated(BaseModel):
    job_id: str
    kind: JobKind
    status: JobStatus
    video: VideoMeta


class JobResponse(BaseModel):
    job_id: str
    kind: JobKind
    status: JobStatus
    stage: str = Field(description="human-readable current step")
    created_at: str
    finished_at: str | None = None
    video: VideoMeta
    detection_rate: float | None = None
    mouth_preview_png: str | None = Field(None, description="base64 PNG strip of up to 8 mouth crops")
    preview_frames: int | None = Field(None, description="number of mouth-crop frames the model input has")
    text: str | None = None
    timings: dict[str, float] = {}
    model_checkpoint: str | None = None
    error: dict[str, str] | None = None


class ExperimentRecord(BaseModel):
    id: str
    kind: Literal["baseline", "finetuning", "ablation"]
    status: Literal["complete", "smoke", "pending"]
    path: str
    split: str | None = None
    num_samples: int | None = None
    wer: float | None = None
    cer: float | None = None
    latency_mean_s: float | None = None
    real_time_factor: float | None = None
    peak_gpu_gb: float | None = None
    parameters_total: int | None = None
    parameters_trainable: int | None = None
    checkpoint: str | None = None
    device: str | None = None
    training: dict[str, Any] | None = None


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentRecord]
    note: str


class CompareResponse(BaseModel):
    status: Literal["complete", "pending"]
    reasons: list[str] = []
    baseline: ExperimentRecord | None = None
    finetuned: ExperimentRecord | None = None
    wer_absolute_change: float | None = None
    wer_relative_change: float | None = None
    cer_absolute_change: float | None = None
    cer_relative_change: float | None = None
    loss_curve: list[dict[str, float]] = []
    val_curve: list[dict[str, float]] = []
