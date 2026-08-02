"""Flask API contract tests against the modular app factory.

A stub QAService is injected, so the whole HTTP + validation + error stack runs
in milliseconds with no model download. This exercises the wiring: factory ->
blueprint -> service -> JSON, plus the centralised error handlers.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app"))

pytest.importorskip("flask")

from qa_web import create_app  # noqa: E402


class StubService:
    """Implements the QAService surface the blueprints depend on."""
    model_dir = "stub-dir"

    def __init__(self):
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def info(self):
        self._loaded = True
        return {"model": "stub", "default_top_k": 5, "max_top_k": 20}

    def answer(self, question, contexts, top_k=5, engine="finetuned"):
        self._loaded = True
        k = min(top_k, 20)
        answers = [
            {"rank": i + 1, "text": f"candidate {i}", "confidence": round(1.0 / (i + 1), 4),
             "relative_score": 0.5, "start_char": 0, "end_char": 9,
             "passage_index": 0, "support": 1, "passages": [0], "is_no_answer": False}
            for i in range(k)
        ]
        return {"question": question, "answers": answers, "engine": engine,
                "top_answer": answers[0]["text"], "n_passages": len(contexts),
                "n_windows": len(contexts), "top_k": k, "latency_ms": 1.0,
                "model": "stub", "passages": list(contexts), "error": None}


@pytest.fixture()
def client():
    app = create_app("test", qa_service=StubService())
    app.config["TESTING"] = True
    return app.test_client()


# ---------- basic reachability ----------
def test_health_is_reachable(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Ask" in r.data


def test_info_endpoint(client):
    assert client.get("/api/info").get_json()["model"] == "stub"


# ---------- the answer contract ----------
def test_answer_returns_a_ranked_list(client):
    r = client.post("/api/answer", json={"question": "who?", "contexts": ["a passage"], "top_k": 3})
    assert r.status_code == 200
    answers = r.get_json()["answers"]
    assert [a["rank"] for a in answers] == [1, 2, 3]
    confs = [a["confidence"] for a in answers]
    assert confs == sorted(confs, reverse=True)


def test_single_context_key_is_accepted(client):
    r = client.post("/api/answer", json={"question": "who?", "context": "a passage"})
    assert r.status_code == 200 and r.get_json()["n_passages"] == 1


def test_top_k_is_honoured(client):
    r = client.post("/api/answer", json={"question": "q", "contexts": ["c"], "top_k": 1})
    assert len(r.get_json()["answers"]) == 1


def test_engine_is_passed_through(client):
    r = client.post("/api/answer", json={"question": "q", "contexts": ["c"], "engine": "baseline"})
    assert r.get_json()["engine"] == "baseline"


def test_passages_are_echoed_for_highlighting(client):
    r = client.post("/api/answer", json={"question": "q", "contexts": ["alpha", "beta"]})
    assert r.get_json()["passages"] == ["alpha", "beta"]


# ---------- validation / error shape ----------
def test_missing_question_is_rejected(client):
    r = client.post("/api/answer", json={"contexts": ["a passage"]})
    assert r.status_code == 400
    body = r.get_json()
    assert body["code"] == "client_error" and "question" in body["error"].lower()


def test_blank_passages_are_rejected(client):
    r = client.post("/api/answer", json={"question": "who?", "contexts": ["   ", ""]})
    assert r.status_code == 400


def test_bad_top_k_is_rejected(client):
    r = client.post("/api/answer", json={"question": "q", "contexts": ["c"], "top_k": "many"})
    assert r.status_code == 400


def test_bad_engine_is_rejected(client):
    r = client.post("/api/answer", json={"question": "q", "contexts": ["c"], "engine": "magic"})
    assert r.status_code == 400


def test_too_many_passages_is_rejected(client):
    r = client.post("/api/answer", json={"question": "q", "contexts": ["c"] * 999})
    assert r.status_code == 400


def test_unknown_api_route_returns_json_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.get_json()["code"] == "not_found"
