"""Request validation, isolated from routing so it is unit-testable on its own.

Raises ApiError (400) on bad input; returns a clean (question, contexts, top_k,
engine) tuple on success. Guard rails come from the active config.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..errors import ApiError


def parse_answer_request(payload: Dict[str, Any], config) -> Tuple[str, List[str], int, str]:
    if not isinstance(payload, dict):
        raise ApiError("Request body must be a JSON object.")

    question = (payload.get("question") or "").strip()
    if not question:
        raise ApiError("Provide a non-empty 'question'.")
    if len(question) > config.MAX_QUESTION_CHARS:
        raise ApiError(f"Question exceeds {config.MAX_QUESTION_CHARS} characters.")

    # Accept either 'contexts': [..] or a single 'context': "..".
    contexts = payload.get("contexts")
    if contexts is None and payload.get("context"):
        contexts = [payload["context"]]
    if not isinstance(contexts, (list, tuple)):
        raise ApiError("Provide 'contexts' as a list of passages, or a single 'context' string.")

    cleaned = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]
    if not cleaned:
        raise ApiError("Provide at least one non-empty context passage.")
    if len(cleaned) > config.MAX_PASSAGES:
        raise ApiError(f"Too many passages (max {config.MAX_PASSAGES}).")
    for i, c in enumerate(cleaned):
        if len(c) > config.MAX_PASSAGE_CHARS:
            raise ApiError(f"Passage {i} exceeds {config.MAX_PASSAGE_CHARS} characters.")

    raw_k = payload.get("top_k", 5)
    try:
        top_k = int(raw_k)
    except (TypeError, ValueError):
        raise ApiError("'top_k' must be an integer.")
    if top_k < 1:
        raise ApiError("'top_k' must be at least 1.")

    engine = str(payload.get("engine", "finetuned")).lower()
    if engine not in ("finetuned", "baseline"):
        raise ApiError("'engine' must be 'finetuned' or 'baseline'.")

    return question, cleaned, top_k, engine
