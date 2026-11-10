"""HTTP routes. Thin: validation and orchestration live in services."""

from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.core.errors import ApiError
from backend.app.services.job_service import QueueFullError
from backend.app.schemas.api import (
    CompareResponse, ExperimentsResponse, HealthResponse, JobCreated, JobResponse, ModelInfoResponse,
)
from backend.app.services.video_service import save_upload, validate_video
from ml.utils.device import cuda_available

router = APIRouter(prefix="/api")


def _svc(request: Request):
    return request.app.state


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    st = _svc(request)
    return HealthResponse(version=st.settings.version, model_state=st.model_service.state,
                          device=st.model_service.device.type, cuda_available=cuda_available())


@router.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info(request: Request) -> ModelInfoResponse:
    return ModelInfoResponse(**_svc(request).model_service.info())


async def _accept_video(request: Request, file: UploadFile, kind: str) -> JobCreated:
    st = _svc(request)
    if st.job_service.is_full():
        raise ApiError(429, "queue_full", "Too many videos are waiting. Try again when the current jobs finish.")
    path, size = await save_upload(file, st.settings)
    try:
        info = await run_in_threadpool(validate_video, path, st.settings)  # decoding must not block the loop
        meta = {"filename": (file.filename or "upload")[:200], "size_bytes": size, "width": info.width,
                "height": info.height, "source_fps": info.source_fps, "duration_s": info.duration_s,
                "output_frames": info.output_frames, "resampled": info.resampled}
        job = st.job_service.submit(kind, path, meta)
    except QueueFullError as exc:
        path.unlink(missing_ok=True)
        raise ApiError(429, "queue_full", str(exc)) from exc
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return JobCreated(job_id=job.job_id, kind=job.kind, status=job.status, video=meta)


@router.post("/preprocess", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED, tags=["recognition"],
             summary="Validate a video and extract the mouth region (works without a model)")
async def preprocess(request: Request, file: UploadFile = File(...)) -> JobCreated:
    return await _accept_video(request, file, "preprocess")


@router.post("/recognize", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED, tags=["recognition"],
             summary="Visual-only speech recognition on an uploaded video",
             responses={503: {"description": "No model loaded (error code model_unavailable)"}})
async def recognize(request: Request, file: UploadFile = File(...)) -> JobCreated:
    ms = _svc(request).model_service
    if ms.state != "ready":
        raise ApiError(503, "model_unavailable",
                       f"Recognition is unavailable: {ms.message} Inference requires a configured Auto-AVSR "
                       "checkpoint (see CONTINUE_ON_RTX5070.md).", {"model_state": ms.state})
    return await _accept_video(request, file, "recognize")


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["recognition"])
async def job_status(request: Request, job_id: str) -> JobResponse:
    job = _svc(request).job_service.get(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", f"No job with id '{job_id}'.")
    return JobResponse(**job.public())


@router.get("/experiments", response_model=ExperimentsResponse, tags=["research"])
async def experiments(request: Request) -> ExperimentsResponse:
    return ExperimentsResponse(
        experiments=_svc(request).experiment_service.list(),
        note="Only results.json files written by ml/evaluation/evaluate.py with status 'complete' are results. "
             "'smoke' entries are CPU pipeline checks and are not experimental results.",
    )


@router.get("/experiments/compare", response_model=CompareResponse, tags=["research"])
async def compare(request: Request) -> CompareResponse:
    return CompareResponse(**_svc(request).experiment_service.compare())
