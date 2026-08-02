"""Span extraction and ranking (numpy only - no model download)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_system.config import QAConfig
from qa_system.postprocessing import (
    NO_ANSWER_TEXT, extract_candidates, rank_candidates,
)

CONTEXT = "Hamlet is a tragedy written by William Shakespeare around 1600."
#          0-6                            31-38   39-50        58-62
OFFSETS = [None, (0, 6), (31, 38), (39, 50), (58, 62), None]


def _logits(start_peak, end_peak, cls=(-5.0, -5.0), peak=8.0, seq=6):
    s = np.full(seq, -5.0, dtype=np.float32)
    e = np.full(seq, -5.0, dtype=np.float32)
    s[0], e[0] = cls
    s[start_peak], e[end_peak] = peak, peak
    return s[None, :], e[None, :]


def test_top_candidate_is_the_highest_scoring_span():
    cfg = QAConfig()
    s, e = _logits(2, 3)
    out = extract_candidates(CONTEXT, [OFFSETS], s, e, cfg)
    assert out["candidates"][0]["text"] == "William Shakespeare"


def test_confidence_is_a_probability():
    cfg = QAConfig()
    s, e = _logits(2, 3)
    for cand in extract_candidates(CONTEXT, [OFFSETS], s, e, cfg)["candidates"]:
        assert 0.0 <= cand["confidence"] <= 1.0


def test_candidates_are_returned_in_descending_confidence():
    cfg = QAConfig()
    s, e = _logits(2, 3)
    conf = [c["confidence"] for c in extract_candidates(CONTEXT, [OFFSETS], s, e, cfg)["candidates"]]
    assert conf == sorted(conf, reverse=True)


def test_top_k_is_respected_and_ranks_are_dense():
    cfg = QAConfig()
    s, e = _logits(2, 3)
    cands = extract_candidates(CONTEXT, [OFFSETS], s, e, cfg)["candidates"]
    ranked = rank_candidates(cands, cfg, top_k=3)
    assert len(ranked) <= 3
    assert [c["rank"] for c in ranked] == list(range(1, len(ranked) + 1))


def test_top_k_is_clamped_to_max():
    cfg = QAConfig(max_top_k=4)
    s, e = _logits(2, 3)
    cands = extract_candidates(CONTEXT, [OFFSETS], s, e, cfg)["candidates"]
    assert len(rank_candidates(cands, cfg, top_k=99)) <= 4


def test_duplicate_answers_across_passages_are_merged_and_boosted():
    cfg = QAConfig(aggregate_duplicates=True, include_no_answer=False)
    weak_shared = [
        {"text": "Bengaluru", "start_char": 0, "end_char": 9, "confidence": 0.30,
         "logit_score": 1.0, "passage_index": 0, "passage_id": None, "window": 0},
        {"text": "bengaluru", "start_char": 0, "end_char": 9, "confidence": 0.30,
         "logit_score": 1.0, "passage_index": 1, "passage_id": None, "window": 0},
        {"text": "Hyderabad", "start_char": 0, "end_char": 9, "confidence": 0.45,
         "logit_score": 1.0, "passage_index": 2, "passage_id": None, "window": 0},
    ]
    ranked = rank_candidates(weak_shared, cfg, top_k=3)
    assert ranked[0]["text"].lower() == "bengaluru"   # 0.30 + 0.30 beats 0.45
    assert ranked[0]["support"] == 2
    assert sorted(ranked[0]["passages"]) == [0, 1]


def test_aggregation_can_be_switched_off():
    cfg = QAConfig(aggregate_duplicates=False, include_no_answer=False)
    cands = [
        {"text": "Bengaluru", "start_char": 0, "end_char": 9, "confidence": 0.30,
         "logit_score": 1.0, "passage_index": 0, "passage_id": None, "window": 0},
        {"text": "Bengaluru", "start_char": 0, "end_char": 9, "confidence": 0.30,
         "logit_score": 1.0, "passage_index": 1, "passage_id": None, "window": 0},
        {"text": "Hyderabad", "start_char": 0, "end_char": 9, "confidence": 0.45,
         "logit_score": 1.0, "passage_index": 2, "passage_id": None, "window": 0},
    ]
    assert rank_candidates(cands, cfg, top_k=3)[0]["text"] == "Hyderabad"


def test_no_answer_enters_the_ranking_only_when_it_wins():
    cfg = QAConfig(version_2_with_negative=True, include_no_answer=True)
    cands = [{"text": "Bengaluru", "start_char": 0, "end_char": 9, "confidence": 0.10,
              "logit_score": 1.0, "passage_index": 0, "passage_id": None, "window": 0}]
    assert rank_candidates(cands, cfg, top_k=3, null_prob=0.8)[0]["text"] == NO_ANSWER_TEXT
    assert rank_candidates(cands, cfg, top_k=3, null_prob=0.01)[0]["text"] == "Bengaluru"


def test_squad_v1_style_model_never_returns_no_answer():
    cfg = QAConfig(dataset_name="squad")           # forces include_no_answer False
    cands = [{"text": "Bengaluru", "start_char": 0, "end_char": 9, "confidence": 0.10,
              "logit_score": 1.0, "passage_index": 0, "passage_id": None, "window": 0}]
    assert rank_candidates(cands, cfg, top_k=3, null_prob=0.99)[0]["text"] == "Bengaluru"


def test_invalid_spans_are_never_emitted():
    cfg = QAConfig(max_answer_length=1)
    s, e = _logits(3, 2)                           # end before start
    for cand in extract_candidates(CONTEXT, [OFFSETS], s, e, cfg)["candidates"]:
        assert cand["start_char"] <= cand["end_char"]
        assert cand["text"] in CONTEXT


def test_tokens_outside_the_context_cannot_be_answers():
    cfg = QAConfig()
    s, e = _logits(5, 5)                           # peak on a masked offset
    assert all(c["text"] in CONTEXT
               for c in extract_candidates(CONTEXT, [OFFSETS], s, e, cfg)["candidates"])
