"""SilentVoice API.

    uvicorn backend.app.main:app --reload --port 8000     # docs at http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import error_response, install_error_handlers
from backend.app.services.experiment_service import ExperimentService
from backend.app.services.job_service import JobService
from backend.app.services.model_service import ModelService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # remove uploads orphaned by a previous crash (the directory is ours alone)
        if settings.upload_dir.is_dir():
            for leftover in settings.upload_dir.iterdir():
                if leftover.is_file():
                    leftover.unlink(missing_ok=True)
        model_service = ModelService(settings)
        app.state.settings = settings
        app.state.model_service = model_service
        app.state.job_service = JobService(model_service, settings.max_jobs_retained)
        app.state.experiment_service = ExperimentService(settings.experiments_dir)
        if settings.load_model_on_startup and settings.checkpoint_path:
            # load in a thread so the API (health, uploads, preprocessing) is up immediately
            asyncio.get_running_loop().run_in_executor(None, model_service.load)
        yield
        app.state.job_service.shutdown()

    app = FastAPI(
        title=settings.app_name, version=settings.version, lifespan=lifespan,
        description="Visual-only speech recognition API built on Auto-AVSR. The model never receives audio.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    install_error_handlers(app)

    @app.middleware("http")
    async def reject_early(request: Request, call_next):
        """Answer before the multipart body is spooled: oversize bodies and recognition without a model."""
        if request.method == "POST" and request.url.path in ("/api/preprocess", "/api/recognize"):
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > settings.max_upload_bytes + 1024 * 1024:
                return error_response(413, "file_too_large", f"File exceeds the {settings.max_upload_mb} MB limit.")
            ms = getattr(request.app.state, "model_service", None)
            if request.url.path == "/api/recognize" and ms is not None and ms.state != "ready":
                return error_response(
                    503, "model_unavailable",
                    f"Recognition is unavailable: {ms.message} Inference requires a configured Auto-AVSR "
                    "checkpoint (see CONTINUE_ON_RTX5070.md).", {"model_state": ms.state})
        return await call_next(request)

    app.include_router(router)
    return app


app = create_app()
