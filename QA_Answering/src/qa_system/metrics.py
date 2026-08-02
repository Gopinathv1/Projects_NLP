"""Answer-quality metrics.

Two families, because this system returns a *ranked list*:

1. Span quality (official SQuAD): Exact Match and F1 on the top-1 answer.
2. Ranking quality: Hit@k, Recall@k and MRR - these are what evaluation
   criterion 6.1 ("the most likely answer appears first") actually measures.
   EM alone cannot distinguish a model that puts the right answer at rank 1
   from one that buries it at rank 5.

Implemented locally so the notebook runs air-gapped and the scoring is
auditable line by line.
"""

from __future__ import annotations

import collections
import re
import string
from typing import Dict, List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Normalisation (lowercase, strip punctuation / articles / extra whitespace)
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc((s or "").lower())))


def _get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split() if s else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    gold_toks, pred_toks = _get_tokens(gold), _get_tokens(pred)
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score_against_golds(golds: Sequence[str], pred: str) -> Tuple[float, float]:
    """Best EM / F1 against any gold answer (SQuAD allows several)."""
    golds = [g for g in golds if normalize_answer(g)] or [""]
    return (
        float(max(compute_exact(g, pred) for g in golds)),
        float(max(compute_f1(g, pred) for g in golds)),
    )


# --------------------------------------------------------------------------- #
# 1. Span quality on the top-1 answer
# --------------------------------------------------------------------------- #
def squad_metrics(predictions: Dict[str, str], references: Dict[str, List[str]]) -> Dict[str, float]:
    """Overall / answerable / unanswerable EM and F1 for the rank-1 answer."""
    em_all, f1_all = {}, {}
    for ex_id, golds in references.items():
        em, f1 = score_against_golds(golds, predictions.get(ex_id, ""))
        em_all[ex_id], f1_all[ex_id] = em, f1

    has_ans = [i for i, g in references.items() if len(g) > 0]
    no_ans = [i for i, g in references.items() if len(g) == 0]
    n = max(len(references), 1)

    out: Dict[str, float] = {
        "exact_match": 100.0 * sum(em_all.values()) / n,
        "f1": 100.0 * sum(f1_all.values()) / n,
        "total": len(references),
    }
    if has_ans:
        out["HasAns_exact"] = 100.0 * sum(em_all[i] for i in has_ans) / len(has_ans)
        out["HasAns_f1"] = 100.0 * sum(f1_all[i] for i in has_ans) / len(has_ans)
        out["HasAns_total"] = len(has_ans)
    if no_ans:
        out["NoAns_exact"] = 100.0 * sum(em_all[i] for i in no_ans) / len(no_ans)
        out["NoAns_total"] = len(no_ans)
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# 2. Ranking quality
# --------------------------------------------------------------------------- #
def ranking_metrics(
    ranked_predictions: Dict[str, List[str]],
    references: Dict[str, List[str]],
    k_values: Sequence[int] = (1, 3, 5),
    f1_threshold: float = 0.5,
) -> Dict[str, float]:
    """Hit@k (exact), Recall@k (F1 >= threshold) and MRR.

    Only answerable questions are scored: a question with no gold answer has no
    correct rank, so including it would inflate every ranking number.
    """
    scored = [i for i, g in references.items() if len(g) > 0]
    if not scored:
        return {"note": "no answerable questions to rank", "total_ranked": 0}

    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    recalls = {k: 0 for k in k_values}
    rr_exact, rr_lenient = [], []

    for ex_id in scored:
        golds = references[ex_id]
        cands = ranked_predictions.get(ex_id, [])[:max_k]
        em_at = [score_against_golds(golds, c)[0] for c in cands]
        f1_at = [score_against_golds(golds, c)[1] for c in cands]

        for k in k_values:
            if any(e == 1.0 for e in em_at[:k]):
                hits[k] += 1
            if any(f >= f1_threshold for f in f1_at[:k]):
                recalls[k] += 1

        first_exact = next((i for i, e in enumerate(em_at, 1) if e == 1.0), None)
        first_lenient = next((i for i, f in enumerate(f1_at, 1) if f >= f1_threshold), None)
        rr_exact.append(1.0 / first_exact if first_exact else 0.0)
        rr_lenient.append(1.0 / first_lenient if first_lenient else 0.0)

    n = len(scored)
    out: Dict[str, float] = {"total_ranked": n}
    for k in k_values:
        out[f"hit@{k}"] = round(100.0 * hits[k] / n, 2)
        out[f"recall@{k}(f1>={f1_threshold})"] = round(100.0 * recalls[k] / n, 2)
    out["MRR_exact"] = round(sum(rr_exact) / n, 4)
    out["MRR_lenient"] = round(sum(rr_lenient) / n, 4)
    return out


def confidence_calibration(
    ranked_predictions: Dict[str, List[dict]],
    references: Dict[str, List[str]],
    n_bins: int = 5,
) -> List[Dict[str, float]]:
    """Is a 0.8-confidence answer right about 80% of the time?

    Bins the rank-1 confidence and reports the observed exact-match rate per
    bin. A well-behaved model shows accuracy rising monotonically with
    confidence, which is what justifies showing the score in the UI at all.
    """
    rows = []
    for ex_id, golds in references.items():
        cands = ranked_predictions.get(ex_id, [])
        if not cands or not golds:
            continue
        top = cands[0]
        em, _ = score_against_golds(golds, top.get("text", ""))
        rows.append((float(top.get("confidence", 0.0)), em))

    if not rows:
        return []

    edges = [i / n_bins for i in range(n_bins + 1)]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [em for conf, em in rows if lo <= conf < hi or (hi == 1.0 and conf == 1.0)]
        if bucket:
            out.append({
                "confidence_bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(bucket),
                "accuracy": round(100.0 * sum(bucket) / len(bucket), 2),
            })
    return out
