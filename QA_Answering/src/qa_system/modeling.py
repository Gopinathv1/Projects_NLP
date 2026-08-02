"""Model / tokenizer construction and persistence (Requirements 4.2, 4.6)."""

from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import QAConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_and_tokenizer(cfg: QAConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a pretrained encoder with a span-prediction head on top."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required: offset mapping and sliding windows "
            "depend on it."
        )
    model_config = AutoConfig.from_pretrained(cfg.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name, config=model_config)
    return model, tokenizer


def save_artifacts(model, tokenizer, cfg: QAConfig, output_dir: str | None = None) -> str:
    """Persist weights + tokenizer + config so the app can reload without retraining."""
    output_dir = output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    cfg.save(os.path.join(output_dir, "qa_config.json"))
    return output_dir


def load_artifacts(model_dir: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, QAConfig]:
    """Reverse of save_artifacts. Falls back to defaults if qa_config.json is absent."""
    cfg_path = os.path.join(model_dir, "qa_config.json")
    cfg = QAConfig.load(cfg_path) if os.path.exists(cfg_path) else QAConfig()
    cfg.model_name = model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer, cfg


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_millions": round(total / 1e6, 1)}
