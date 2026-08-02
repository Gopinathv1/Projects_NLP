"""Central configuration object.

Every module takes a QAConfig, so one dataclass drives the notebook, the CLI
scripts and the Flask service. It is saved next to the model weights, which is
what makes a checkpoint self-describing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class QAConfig:
    # ---------------- model / tokenizer ----------------
    model_name: str = "distilbert-base-uncased"
    # Fine-tuning base. Any AutoModelForQuestionAnswering checkpoint works:
    #   bert-base-uncased | roberta-base | deberta-v3-base
    # Already-fine-tuned checkpoints that answer with zero training:
    #   distilbert-base-cased-distilled-squad | deepset/roberta-base-squad2

    baseline_model_name: str = "distilbert-base-cased-distilled-squad"
    # Used by qa_system.baseline for the out-of-the-box HF pipeline (Req. 4.3)

    # ---------------- data ----------------
    dataset_name: str = "squad_v2"      # "squad" | "squad_v2" | "local"
    dataset_config: Optional[str] = None
    train_file: Optional[str] = None    # used when dataset_name == "local"
    validation_file: Optional[str] = None
    max_train_samples: Optional[int] = None   # None = full split
    max_eval_samples: Optional[int] = None

    # ---------------- preprocessing ----------------
    max_seq_length: int = 384      # question + context window
    doc_stride: int = 128          # overlap between consecutive windows
    pad_to_max_length: bool = True

    # ---------------- training ----------------
    learning_rate: float = 3e-5
    num_train_epochs: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    fp16: bool = True              # ignored on CPU
    seed: int = 42

    # ---------------- ranking / decoding ----------------
    top_k: int = 5                 # default number of candidates returned (Req. 4.5)
    max_top_k: int = 20            # hard cap the API will honour
    n_best_size: int = 20          # search width per window before ranking
    max_answer_length: int = 30    # token cap on a candidate span
    aggregate_duplicates: bool = True
    # True: the same answer found in several passages is merged, and its evidence
    # is combined - a fact supported by two documents outranks a lone mention.

    version_2_with_negative: bool = True   # model was trained with unanswerables
    include_no_answer: bool = True         # surface "no answer" when it outranks every span

    # ---------------- io ----------------
    output_dir: str = "artifacts/qa-model"
    logging_steps: int = 50

    extra: dict = field(default_factory=dict)

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QAConfig":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def clamp_top_k(self, requested: Optional[int]) -> int:
        if requested is None:
            return self.top_k
        return max(1, min(int(requested), self.max_top_k))

    def __post_init__(self) -> None:
        if self.doc_stride >= self.max_seq_length:
            raise ValueError("doc_stride must be smaller than max_seq_length")
        if self.dataset_name == "squad":
            # SQuAD v1.1 has no unanswerable questions
            self.version_2_with_negative = False
        if not self.version_2_with_negative:
            self.include_no_answer = False
