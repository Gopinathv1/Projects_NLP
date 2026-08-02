"""Unit tests for request validation, independent of the HTTP layer."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

pytest.importorskip("flask")

from qa_web.errors import ApiError                       # noqa: E402
from qa_web.services.validation import parse_answer_request  # noqa: E402
from qa_web.settings import TestConfig                   # noqa: E402

CFG = TestConfig


def test_happy_path():
    q, ctx, k, engine = parse_answer_request(
        {"question": "who?", "contexts": ["a", "b"], "top_k": 3, "engine": "baseline"}, CFG)
    assert q == "who?" and ctx == ["a", "b"] and k == 3 and engine == "baseline"


def test_defaults_are_applied():
    q, ctx, k, engine = parse_answer_request({"question": "q", "context": "a"}, CFG)
    assert k == 5 and engine == "finetuned" and ctx == ["a"]


def test_whitespace_passages_dropped():
    _, ctx, _, _ = parse_answer_request({"question": "q", "contexts": ["  ", "real", ""]}, CFG)
    assert ctx == ["real"]


@pytest.mark.parametrize("payload", [
    {"contexts": ["a"]},                                  # no question
    {"question": "", "contexts": ["a"]},                  # blank question
    {"question": "q"},                                    # no contexts
    {"question": "q", "contexts": []},                    # empty contexts
    {"question": "q", "contexts": "not-a-list"},          # wrong type
    {"question": "q", "contexts": ["a"], "top_k": "x"},   # bad top_k
    {"question": "q", "contexts": ["a"], "top_k": 0},     # top_k < 1
    {"question": "q", "contexts": ["a"], "engine": "no"}, # bad engine
])
def test_bad_requests_raise_apierror(payload):
    with pytest.raises(ApiError):
        parse_answer_request(payload, CFG)
