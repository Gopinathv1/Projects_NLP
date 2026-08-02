"""Span and ranking metrics (pure python)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_system.metrics import (
    compute_exact, compute_f1, confidence_calibration, normalize_answer,
    ranking_metrics, squad_metrics,
)


def test_normalisation_ignores_articles_case_punctuation():
    assert normalize_answer("The  Taj Mahal.") == "taj mahal"


def test_exact_match_is_normalised():
    assert compute_exact("the Taj Mahal", "Taj Mahal!") == 1
    assert compute_exact("Taj Mahal", "Red Fort") == 0


def test_partial_overlap_gets_partial_f1():
    assert 0.0 < compute_f1("in the year 1632", "1632") < 1.0


def test_top1_metrics_split_answerable_and_unanswerable():
    refs = {"a": [], "b": ["Paris"]}
    m = squad_metrics({"a": "", "b": "Paris"}, refs)
    assert m["exact_match"] == 100.0 and m["NoAns_exact"] == 100.0
    assert squad_metrics({"a": "Paris", "b": "Paris"}, refs)["exact_match"] == 50.0


def test_mrr_rewards_the_correct_answer_being_first():
    refs = {"q1": ["Paris"], "q2": ["Paris"]}
    first = {"q1": ["Paris", "Lyon", "Nice"], "q2": ["Paris", "Lyon", "Nice"]}
    third = {"q1": ["Lyon", "Nice", "Paris"], "q2": ["Lyon", "Nice", "Paris"]}
    assert ranking_metrics(first, refs)["MRR_exact"] == 1.0
    assert round(ranking_metrics(third, refs)["MRR_exact"], 4) == round(1 / 3, 4)


def test_hit_at_k_grows_with_k():
    refs = {"q1": ["Paris"]}
    preds = {"q1": ["Lyon", "Nice", "Paris"]}
    m = ranking_metrics(preds, refs, k_values=(1, 3, 5))
    assert m["hit@1"] == 0.0 and m["hit@3"] == 100.0 and m["hit@5"] == 100.0


def test_unanswerable_questions_are_excluded_from_ranking_metrics():
    refs = {"q1": ["Paris"], "q2": []}
    m = ranking_metrics({"q1": ["Paris"], "q2": ["anything"]}, refs)
    assert m["total_ranked"] == 1 and m["hit@1"] == 100.0


def test_recall_at_k_gives_credit_for_near_misses():
    refs = {"q1": ["four weeks"]}
    m = ranking_metrics({"q1": ["four weeks after"]}, refs, k_values=(1,), f1_threshold=0.5)
    assert m["hit@1"] == 0.0                       # not an exact match
    assert m["recall@1(f1>=0.5)"] == 100.0         # but a useful answer


def test_calibration_bins_confidence_against_accuracy():
    refs = {"a": ["Paris"], "b": ["Paris"]}
    ranked = {"a": [{"text": "Paris", "confidence": 0.9}],
              "b": [{"text": "Lyon", "confidence": 0.1}]}
    bins = confidence_calibration(ranked, refs, n_bins=2)
    by_bin = {b["confidence_bin"]: b["accuracy"] for b in bins}
    assert by_bin["0.0-0.5"] == 0.0 and by_bin["0.5-1.0"] == 100.0
