"""JSON API routes.

Thin by design: each handler validates input, calls the QAService that hangs off
the app, and returns JSON. No model code, no torch imports here - that all lives
in the service layer, which is what makes these routes testable with a stub.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from ..errors import ApiError
from ..services.validation import parse_answer_request

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _service():
    svc = current_app.extensions.get("qa_service")
    if svc is None:
        raise ApiError("QA service is not configured.", status=503, code="service_missing")
    return svc


@api_bp.get("/health")
def health():
    svc = _service()
    return jsonify({
        "status": "ok",
        "model_loaded": svc.loaded,
        "model_dir": svc.model_dir,
    })


@api_bp.get("/info")
def info():
    return jsonify(_service().info())


@api_bp.post("/answer")
def answer():
    payload = request.get_json(silent=True)
    question, contexts, top_k, engine = parse_answer_request(payload, current_app.config_object)
    result = _service().answer(question, contexts, top_k=top_k, engine=engine)
    return jsonify(result)
