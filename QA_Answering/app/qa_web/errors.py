"""Typed application errors + centralised handlers.

Blueprints raise ApiError instead of building error responses inline, so the
JSON error shape is defined in exactly one place.
"""

from __future__ import annotations

from flask import Flask, jsonify, request


class ApiError(Exception):
    """A client- or server-side failure with a stable JSON shape."""

    def __init__(self, message: str, status: int = 400, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code or ("client_error" if status < 500 else "server_error")

    def to_dict(self) -> dict:
        return {"error": self.message, "code": self.code, "status": self.status}


class ModelUnavailableError(ApiError):
    def __init__(self, message: str = "The QA model could not be loaded."):
        super().__init__(message, status=503, code="model_unavailable")


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):
        app.logger.warning("ApiError %s: %s", exc.status, exc.message)
        return jsonify(exc.to_dict()), exc.status

    @app.errorhandler(404)
    def _not_found(_exc):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found", "code": "not_found", "status": 404}), 404
        return _exc, 404

    @app.errorhandler(500)
    def _server_error(exc):
        app.logger.exception("Unhandled error: %s", exc)
        return jsonify({"error": "Internal server error", "code": "server_error",
                        "status": 500}), 500
