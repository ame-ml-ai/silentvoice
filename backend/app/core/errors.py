"""Error type shared by services and the JSON error envelope returned by the API.

Every error response has the shape ``{"error": {"code", "message", "details"}}``.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("silentvoice.api")


class ApiError(Exception):
    """An expected, user-facing error with a stable machine-readable code."""

    def __init__(self, status_code: int, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.status_code, self.code, self.message, self.details = status_code, code, message, details or {}


def error_response(status_code: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": jsonable_encoder(details or {})}}
    return JSONResponse(status_code=status_code, content=body)


_HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(exc.status_code, _HTTP_CODES.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(422, "invalid_request", "Request validation failed.", {"errors": exc.errors()})

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "Unexpected server error.")
