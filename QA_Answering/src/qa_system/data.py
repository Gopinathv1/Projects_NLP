"""Dataset loading.

Supports the Hugging Face hub (squad / squad_v2) and local JSON files that
follow the official SQuAD schema, so the same pipeline runs on a client's
own knowledge base without code changes.
"""

from __future__ import annotations

import json
from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset

from .config import QAConfig


def _read_squad_json(path: str) -> Dataset:
    """Parse an official SQuAD-format JSON file into a flat Dataset."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows: List[Dict] = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                answers = qa.get("answers", [])
                if qa.get("is_impossible", False):
                    answers = []
                rows.append(
                    {
                        "id": qa["id"],
                        "title": article.get("title", ""),
                        "context": context,
                        "question": qa["question"],
                        "answers": {
                            "text": [a["text"] for a in answers],
                            "answer_start": [a["answer_start"] for a in answers],
                        },
                    }
                )
    return Dataset.from_list(rows)


def load_qa_datasets(cfg: QAConfig) -> DatasetDict:
    """Return a DatasetDict with 'train' and 'validation' splits."""
    if cfg.dataset_name == "local":
        if not (cfg.train_file and cfg.validation_file):
            raise ValueError("train_file and validation_file are required for local data")
        ds = DatasetDict(
            train=_read_squad_json(cfg.train_file),
            validation=_read_squad_json(cfg.validation_file),
        )
    else:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config)

    if cfg.max_train_samples:
        n = min(cfg.max_train_samples, len(ds["train"]))
        ds["train"] = ds["train"].shuffle(seed=cfg.seed).select(range(n))
    if cfg.max_eval_samples:
        n = min(cfg.max_eval_samples, len(ds["validation"]))
        ds["validation"] = ds["validation"].shuffle(seed=cfg.seed).select(range(n))
    return ds


def dataset_summary(ds: DatasetDict) -> Dict[str, Dict]:
    """Light-weight corpus stats used in the notebook and the report."""
    out = {}
    for split, d in ds.items():
        n_unanswerable = sum(1 for a in d["answers"] if len(a["answer_start"]) == 0)
        ctx_len = [len(c.split()) for c in d["context"]]
        out[split] = {
            "n_examples": len(d),
            "n_unanswerable": n_unanswerable,
            "pct_unanswerable": round(100 * n_unanswerable / max(len(d), 1), 2),
            "context_words_mean": round(sum(ctx_len) / max(len(ctx_len), 1), 1),
            "context_words_max": max(ctx_len) if ctx_len else 0,
        }
    return out
