"""Out-of-the-box Hugging Face inference pipeline (Requirement 4.3).

The brief asks for two things: a pipeline that answers questions with no
training at all, and support for fine-tuning. This module is the first half. It
also serves as the honest reference point for the second half - a fine-tuned
model that cannot beat `pipeline()` on the target domain is not worth shipping.

The pipeline's `top_k` already returns ranked candidates per passage; the
multi-passage merge and the aggregation of duplicate answers are added here so
the baseline and the fine-tuned system expose the same contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from transformers import pipeline

from .config import QAConfig
from .metrics import normalize_answer


def build_qa_pipeline(model_name: Optional[str] = None, device: Optional[int] = None,
                      cfg: Optional[QAConfig] = None):
    """Create a Hugging Face `question-answering` pipeline.

    device: 0 for the first GPU, -1 for CPU. Left as None it is auto-detected.
    """
    import torch

    cfg = cfg or QAConfig()
    model_name = model_name or cfg.baseline_model_name
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline("question-answering", model=model_name, tokenizer=model_name, device=device)


def baseline_answer(
    qa_pipe,
    question: str,
    contexts: Union[str, Sequence[str]],
    top_k: int = 5,
    max_answer_len: int = 30,
    doc_stride: int = 128,
    max_seq_len: int = 384,
    aggregate_duplicates: bool = True,
) -> List[Dict[str, Any]]:
    """Ranked answers across one or more passages, using the stock pipeline."""
    if isinstance(contexts, str):
        contexts = [contexts]

    pooled: List[Dict[str, Any]] = []
    for idx, ctx in enumerate(contexts):
        out = qa_pipe(
            question=question,
            context=ctx,
            top_k=top_k,
            max_answer_len=max_answer_len,
            doc_stride=doc_stride,
            max_seq_len=max_seq_len,
            handle_impossible_answer=False,
        )
        for item in (out if isinstance(out, list) else [out]):
            if not item.get("answer"):
                continue
            pooled.append({
                "text": item["answer"],
                "confidence": float(item["score"]),
                "start_char": int(item["start"]),
                "end_char": int(item["end"]),
                "passage_index": idx,
                "support": 1,
                "passages": [idx],
            })

    if aggregate_duplicates:
        groups: Dict[str, Dict[str, Any]] = {}
        for cand in sorted(pooled, key=lambda c: c["confidence"], reverse=True):
            key = normalize_answer(cand["text"])
            if key in groups:
                g = groups[key]
                g["confidence"] = min(1.0, g["confidence"] + cand["confidence"])
                if cand["passage_index"] not in g["passages"]:
                    g["passages"].append(cand["passage_index"])
                    g["support"] += 1
            else:
                groups[key] = dict(cand)
        pooled = list(groups.values())

    pooled.sort(key=lambda c: c["confidence"], reverse=True)
    ranked = pooled[:top_k]
    for rank, cand in enumerate(ranked, start=1):
        cand["rank"] = rank
        cand["confidence"] = round(cand["confidence"], 6)
        cand["is_no_answer"] = False
    return ranked
