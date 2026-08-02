"""Alignment tests - these download a tokenizer, so they are skipped if offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_system.config import QAConfig  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    except Exception as exc:  # offline / no cache
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_char_span_maps_back_to_the_gold_answer(tokenizer):
    from qa_system.preprocessing import prepare_train_features

    cfg = QAConfig(max_seq_length=64, doc_stride=16)
    context = "Hamlet is a tragedy written by William Shakespeare around 1600."
    examples = {
        "question": ["Who wrote Hamlet?"],
        "context": [context],
        "answers": [{"text": ["William Shakespeare"],
                     "answer_start": [context.index("William Shakespeare")]}],
    }
    feats = prepare_train_features(examples, tokenizer, cfg)
    s, e = feats["start_positions"][0], feats["end_positions"][0]
    decoded = tokenizer.decode(feats["input_ids"][0][s : e + 1])
    assert "shakespeare" in decoded.lower()


def test_long_context_produces_multiple_windows(tokenizer):
    from qa_system.preprocessing import describe_windowing

    cfg = QAConfig(max_seq_length=64, doc_stride=16)
    context = "The system stores documents. " * 60
    info = describe_windowing("What does the system store?", context, tokenizer, cfg)
    assert info["n_windows"] > 1
    # consecutive windows must overlap, otherwise an answer on a boundary is lost
    assert info["windows"][1]["char_start"] < info["windows"][0]["char_end"]


def test_unanswerable_is_labelled_with_cls(tokenizer):
    from qa_system.preprocessing import prepare_train_features

    cfg = QAConfig(max_seq_length=64, doc_stride=16)
    examples = {
        "question": ["Who is the CFO?"],
        "context": ["The company was founded in Bengaluru in 2019."],
        "answers": [{"text": [], "answer_start": []}],
    }
    feats = prepare_train_features(examples, tokenizer, cfg)
    assert feats["start_positions"][0] == 0 and feats["end_positions"][0] == 0
