"""Serving pipeline: the object the Flask app and the CLI both use (Req. 4.4-4.6)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .config import QAConfig
from .modeling import get_device, load_artifacts
from .postprocessing import NO_ANSWER_TEXT, answer_question_over_passages
from .preprocessing import featurise_single


class RankedQAPipeline:
    """Load a model once, answer many questions over one or more passages.

    >>> qa = RankedQAPipeline.from_pretrained("artifacts/qa-model")
    >>> qa.answer("Who wrote Hamlet?", ["Hamlet is a tragedy by William Shakespeare."], top_k=3)

    `model_dir` may be a fine-tuned directory produced by this project or any
    Hugging Face hub id, so the service runs before any training has happened.
    """

    def __init__(self, model, tokenizer, cfg: QAConfig, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device or get_device()
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, model_dir: str, device=None, **overrides) -> "RankedQAPipeline":
        model, tokenizer, cfg = load_artifacts(model_dir)
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cls(model, tokenizer, cfg, device)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def answer(
        self,
        question: str,
        contexts: Union[str, Sequence[str]],
        top_k: Optional[int] = None,
        passage_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Return a ranked list of candidate answers, most confident first."""
        question = (question or "").strip()
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = [c.strip() for c in (contexts or []) if c and c.strip()]

        if not question or not contexts:
            return {
                "question": question, "answers": [], "n_passages": 0, "n_windows": 0,
                "top_k": self.cfg.clamp_top_k(top_k), "latency_ms": 0.0,
                "error": "Provide a question and at least one context passage.",
            }

        t0 = time.time()
        k = self.cfg.clamp_top_k(top_k)

        offsets_all, starts_all, ends_all, n_windows = [], [], [], 0
        for context in contexts:
            inputs, offsets = featurise_single(question, context, self.tokenizer, self.cfg)
            inputs = {kk: v.to(self.device) for kk, v in inputs.items()}
            out = self.model(**inputs)
            offsets_all.append(offsets)
            starts_all.append(out.start_logits.float().cpu().numpy())
            ends_all.append(out.end_logits.float().cpu().numpy())
            n_windows += len(offsets)

        ranked = answer_question_over_passages(
            passages=contexts,
            offsets_per_passage=offsets_all,
            start_logits_per_passage=starts_all,
            end_logits_per_passage=ends_all,
            cfg=self.cfg,
            top_k=k,
            passage_ids=passage_ids,
        )

        return {
            "question": question,
            "answers": ranked,
            "top_answer": ranked[0]["text"] if ranked else None,
            "n_passages": len(contexts),
            "n_windows": n_windows,
            "top_k": k,
            "latency_ms": round(1000 * (time.time() - t0), 1),
            "model": self.cfg.model_name,
            "error": None,
        }

    # ------------------------------------------------------------------ #
    def answer_many(self, questions: Sequence[str], contexts, **kw) -> List[Dict[str, Any]]:
        return [self.answer(q, contexts, **kw) for q in questions]

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "model": self.cfg.model_name,
            "device": str(self.device),
            "max_seq_length": self.cfg.max_seq_length,
            "doc_stride": self.cfg.doc_stride,
            "default_top_k": self.cfg.top_k,
            "max_top_k": self.cfg.max_top_k,
            "aggregate_duplicates": self.cfg.aggregate_duplicates,
            "no_answer_supported": self.cfg.include_no_answer,
            "no_answer_label": NO_ANSWER_TEXT,
        }
