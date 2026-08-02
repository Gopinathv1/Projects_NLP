"""Span extraction, confidence scoring and cross-passage ranking (Req. 4.4).

Why probabilities and not logit sums
------------------------------------
The obvious score for a span is `start_logit + end_logit`. It ranks candidates
correctly *within* one forward pass, but it is unbounded and un-normalised, so
it cannot be compared across passages: a confident answer in a short passage
and a weak one in a long passage can produce similar sums. Since this system
must merge candidates from several passages into one ranked list, each window's
logits are softmaxed over the valid context positions instead, and a candidate
is scored

    confidence = P(start = i) * P(end = j)

which lives in [0, 1], is comparable across passages, and is the same scoring
convention the Hugging Face `question-answering` pipeline uses - so the
fine-tuned model and the out-of-the-box baseline can be compared directly.

The raw logit sum is retained on every candidate as `logit_score` for
diagnostics.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import QAConfig
from .metrics import normalize_answer

NO_ANSWER_TEXT = "[no answer in the provided context]"


def _masked_softmax(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Softmax restricted to positions that may legally start/end an answer."""
    x = np.where(valid, logits.astype(np.float64), -1e9)
    x = x - x.max()
    e = np.exp(x)
    total = e.sum()
    return e / total if total > 0 else np.full_like(e, 1.0 / len(e))


def extract_candidates(
    context: str,
    offsets_per_window: Sequence[Sequence[Optional[Sequence[int]]]],
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    cfg: QAConfig,
    passage_index: int = 0,
    passage_id: Optional[str] = None,
) -> Dict[str, Any]:
    """All plausible answer spans in ONE passage, each with a confidence score."""
    candidates: List[Dict[str, Any]] = []
    null_prob = 0.0

    for w, offsets in enumerate(offsets_per_window):
        s_logits = np.asarray(start_logits[w], dtype=np.float64)
        e_logits = np.asarray(end_logits[w], dtype=np.float64)

        # A position is valid if it maps into the context. The CLS position
        # (index 0) is additionally valid when the model was trained with
        # unanswerable questions, because that is where it puts "no answer".
        valid = np.zeros(len(s_logits), dtype=bool)
        for k, off in enumerate(offsets):
            if k < len(valid) and off is not None:
                valid[k] = True
        if cfg.version_2_with_negative:
            valid[0] = True

        p_start = _masked_softmax(s_logits, valid)
        p_end = _masked_softmax(e_logits, valid)
        null_prob = max(null_prob, float(p_start[0] * p_end[0]))

        start_idx = np.argsort(p_start)[-1 : -cfg.n_best_size - 1 : -1].tolist()
        end_idx = np.argsort(p_end)[-1 : -cfg.n_best_size - 1 : -1].tolist()

        for si in start_idx:
            for ei in end_idx:
                if si >= len(offsets) or ei >= len(offsets):
                    continue
                if offsets[si] is None or offsets[ei] is None:
                    continue                        # outside the context
                if ei < si:
                    continue                        # end before start
                if ei - si + 1 > cfg.max_answer_length:
                    continue                        # implausibly long span
                start_char, end_char = int(offsets[si][0]), int(offsets[ei][1])
                text = context[start_char:end_char].strip()
                if not text:
                    continue
                candidates.append(
                    {
                        "text": text,
                        "start_char": start_char,
                        "end_char": end_char,
                        "confidence": float(p_start[si] * p_end[ei]),
                        "logit_score": float(s_logits[si] + e_logits[ei]),
                        "passage_index": passage_index,
                        "passage_id": passage_id,
                        "window": w,
                    }
                )

    # Within a passage, the same span can be produced by two overlapping
    # windows. Keep the strongest occurrence.
    best_by_span: Dict[tuple, Dict[str, Any]] = {}
    for cand in candidates:
        key = (cand["start_char"], cand["end_char"])
        if key not in best_by_span or cand["confidence"] > best_by_span[key]["confidence"]:
            best_by_span[key] = cand

    ordered = sorted(best_by_span.values(), key=lambda c: c["confidence"], reverse=True)
    return {"candidates": ordered, "null_prob": null_prob}


def rank_candidates(
    candidates: Sequence[Dict[str, Any]],
    cfg: QAConfig,
    top_k: Optional[int] = None,
    null_prob: float = 0.0,
) -> List[Dict[str, Any]]:
    """Merge candidates from every passage into one ranked list (Req. 4.4).

    With `aggregate_duplicates`, answers that normalise to the same string are
    combined: their confidences are summed (capped at 1.0) and `support` records
    how many passages produced them. An answer corroborated by two documents
    therefore outranks a single slightly-more-confident mention, which is the
    behaviour you want when the passages are independent sources.
    """
    top_k = cfg.clamp_top_k(top_k)

    if cfg.aggregate_duplicates:
        groups: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        for cand in sorted(candidates, key=lambda c: c["confidence"], reverse=True):
            key = normalize_answer(cand["text"])
            if not key:
                continue
            if key not in groups:
                groups[key] = {**cand, "support": 1, "passages": [cand["passage_index"]],
                               "aggregate_confidence": cand["confidence"]}
            else:
                g = groups[key]
                g["aggregate_confidence"] = min(1.0, g["aggregate_confidence"] + cand["confidence"])
                if cand["passage_index"] not in g["passages"]:
                    g["passages"].append(cand["passage_index"])
                    g["support"] += 1
        merged = list(groups.values())
        for m in merged:
            m["confidence"] = m.pop("aggregate_confidence")
    else:
        merged = [{**c, "support": 1, "passages": [c["passage_index"]]} for c in candidates]

    merged.sort(key=lambda c: c["confidence"], reverse=True)

    # Optionally let "no answer" compete for a place in the ranking.
    if cfg.include_no_answer and null_prob > 0:
        best = merged[0]["confidence"] if merged else 0.0
        if null_prob > best:
            merged.insert(0, {
                "text": NO_ANSWER_TEXT, "start_char": None, "end_char": None,
                "confidence": float(null_prob), "logit_score": None,
                "passage_index": None, "passage_id": None, "window": None,
                "support": 0, "passages": [], "is_no_answer": True,
            })

    ranked = merged[:top_k]
    total = sum(c["confidence"] for c in ranked) or 1.0
    for rank, cand in enumerate(ranked, start=1):
        cand["rank"] = rank
        cand["confidence"] = round(float(cand["confidence"]), 6)
        cand["relative_score"] = round(float(cand["confidence"] / total), 4)
        cand.setdefault("is_no_answer", False)
    return ranked


def answer_question_over_passages(
    passages: Sequence[str],
    offsets_per_passage: Sequence[Sequence[Sequence[Optional[Sequence[int]]]]],
    start_logits_per_passage: Sequence[np.ndarray],
    end_logits_per_passage: Sequence[np.ndarray],
    cfg: QAConfig,
    top_k: Optional[int] = None,
    passage_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper: extract from every passage, then rank globally."""
    pooled: List[Dict[str, Any]] = []
    null_prob = 0.0
    for i, context in enumerate(passages):
        out = extract_candidates(
            context=context,
            offsets_per_window=offsets_per_passage[i],
            start_logits=start_logits_per_passage[i],
            end_logits=end_logits_per_passage[i],
            cfg=cfg,
            passage_index=i,
            passage_id=passage_ids[i] if passage_ids else None,
        )
        pooled.extend(out["candidates"])
        null_prob = max(null_prob, out["null_prob"])
    return rank_candidates(pooled, cfg, top_k=top_k, null_prob=null_prob)


def postprocess_ranked_predictions(
    examples,
    features,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    cfg: QAConfig,
    top_k: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Batch version for evaluation: example id -> ranked candidate list."""
    feature_per_example = collections.defaultdict(list)
    for idx, ex_id in enumerate(features["example_id"]):
        feature_per_example[ex_id].append(idx)

    offsets_all = features["offset_mapping"]
    predictions: Dict[str, List[Dict[str, Any]]] = {}

    for ex_id, context in zip(examples["id"], examples["context"]):
        rows = feature_per_example[ex_id]
        out = extract_candidates(
            context=context,
            offsets_per_window=[offsets_all[r] for r in rows],
            start_logits=start_logits[rows],
            end_logits=end_logits[rows],
            cfg=cfg,
        )
        predictions[ex_id] = rank_candidates(
            out["candidates"], cfg, top_k=top_k, null_prob=out["null_prob"]
        )
    return predictions
