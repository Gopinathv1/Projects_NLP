"""The service the blueprints call. Routes never import torch or the pipeline
directly - they go through this object, which keeps the HTTP layer thin and
independently testable (a stub QAService is injected in tests).

Both engines are funnelled through one `answer()` contract:
  * "finetuned" -> RankedQAPipeline (this project's model)
  * "baseline"  -> the stock Hugging Face question-answering pipeline (Req. 4.3)

Models are loaded lazily and cached, so importing this module - and booting the
Flask app - costs nothing until the first real question arrives (unless
EAGER_LOAD is set).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class QAService:
    """Owns model lifecycle and turns a validated request into ranked answers."""

    def __init__(self, model_dir: str, fallback_model: str, baseline_model: str):
        self.model_dir = model_dir
        self.fallback_model = fallback_model
        self.baseline_model = baseline_model
        self._pipeline = None          # RankedQAPipeline (fine-tuned or fallback)
        self._baseline = None          # HF question-answering pipeline
        self._lock = threading.Lock()  # guard first-time loads under concurrency

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _load_pipeline(self):
        from qa_system.inference import RankedQAPipeline

        target = self.model_dir if os.path.isdir(self.model_dir) else self.fallback_model
        if target != self.model_dir:
            logger.warning("No checkpoint at '%s'; falling back to '%s'",
                           self.model_dir, target)
        logger.info("Loading ranked QA pipeline from '%s'", target)
        return RankedQAPipeline.from_pretrained(target)

    def pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = self._load_pipeline()
        return self._pipeline

    def baseline(self):
        if self._baseline is None:
            with self._lock:
                if self._baseline is None:
                    from qa_system.baseline import build_qa_pipeline
                    logger.info("Loading baseline pipeline '%s'", self.baseline_model)
                    self._baseline = build_qa_pipeline(self.baseline_model)
        return self._baseline

    def warmup(self) -> None:
        """Eager-load the fine-tuned pipeline (used when EAGER_LOAD is set)."""
        self.pipeline()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def info(self) -> Dict[str, Any]:
        return self.pipeline().info

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def answer(
        self,
        question: str,
        contexts: Sequence[str],
        top_k: int = 5,
        engine: str = "finetuned",
    ) -> Dict[str, Any]:
        """Dispatch to the requested engine and return a uniform response dict."""
        if engine == "baseline":
            return self._answer_baseline(question, contexts, top_k)
        return self._answer_finetuned(question, contexts, top_k)

    def _answer_finetuned(self, question, contexts, top_k) -> Dict[str, Any]:
        result = self.pipeline().answer(question, list(contexts), top_k=top_k)
        result["engine"] = "finetuned"
        result["passages"] = list(contexts)
        return result

    def _answer_baseline(self, question, contexts, top_k) -> Dict[str, Any]:
        from qa_system.baseline import baseline_answer

        ranked: List[Dict[str, Any]] = baseline_answer(
            self.baseline(), question, list(contexts), top_k=top_k
        )
        return {
            "question": question,
            "answers": ranked,
            "top_answer": ranked[0]["text"] if ranked else None,
            "engine": "baseline",
            "n_passages": len(contexts),
            "n_windows": None,
            "top_k": top_k,
            "latency_ms": None,
            "model": self.baseline_model,
            "passages": list(contexts),
            "error": None,
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config) -> "QAService":
        return cls(
            model_dir=config.MODEL_DIR,
            fallback_model=config.FALLBACK_MODEL,
            baseline_model=config.BASELINE_MODEL,
        )
