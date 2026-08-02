#!/usr/bin/env python
"""Fine-tune the QA model end to end (Requirement 4.3).

Examples
--------
    python scripts/run_train.py --max-train-samples 8000 --max-eval-samples 1500
    python scripts/run_train.py --model-name bert-base-uncased --epochs 2
    python scripts/run_train.py --dataset local --train-file data/train_sample.json \
                                --validation-file data/dev_sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_system.config import QAConfig  # noqa: E402
from qa_system.data import dataset_summary, load_qa_datasets  # noqa: E402
from qa_system.evaluation import evaluate_model  # noqa: E402
from qa_system.modeling import build_model_and_tokenizer, count_parameters  # noqa: E402
from qa_system.preprocessing import prepare_train_features, prepare_validation_features  # noqa: E402
from qa_system.train import train_model  # noqa: E402


def parse_args() -> QAConfig:
    p = argparse.ArgumentParser(description="Fine-tune a ranked extractive QA model")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--dataset", default="squad_v2", dest="dataset_name")
    p.add_argument("--train-file")
    p.add_argument("--validation-file")
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    p.add_argument("--max-seq-length", type=int, default=384)
    p.add_argument("--doc-stride", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5, dest="learning_rate")
    p.add_argument("--epochs", type=int, default=2, dest="num_train_epochs")
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output-dir", default="artifacts/qa-model")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    return QAConfig(
        model_name=a.model_name, dataset_name=a.dataset_name,
        train_file=a.train_file, validation_file=a.validation_file,
        max_train_samples=a.max_train_samples, max_eval_samples=a.max_eval_samples,
        max_seq_length=a.max_seq_length, doc_stride=a.doc_stride,
        learning_rate=a.learning_rate, num_train_epochs=a.num_train_epochs,
        train_batch_size=a.train_batch_size, eval_batch_size=a.eval_batch_size,
        top_k=a.top_k, output_dir=a.output_dir, fp16=not a.no_fp16, seed=a.seed,
    )


def main() -> None:
    cfg = parse_args()
    print(json.dumps(cfg.__dict__, indent=2, default=str))

    raw = load_qa_datasets(cfg)
    print(json.dumps(dataset_summary(raw), indent=2))

    model, tokenizer = build_model_and_tokenizer(cfg)
    print("parameters:", count_parameters(model))

    train_features = raw["train"].map(
        partial(prepare_train_features, tokenizer=tokenizer, cfg=cfg),
        batched=True, remove_columns=raw["train"].column_names, desc="tokenising train",
    )
    eval_features = raw["validation"].map(
        partial(prepare_validation_features, tokenizer=tokenizer, cfg=cfg),
        batched=True, remove_columns=raw["validation"].column_names, desc="tokenising validation",
    )
    print(f"train examples={len(raw['train'])} -> features={len(train_features)}")

    history = train_model(model, tokenizer, train_features, cfg,
                          eval_examples=raw["validation"], eval_features=eval_features)

    report = evaluate_model(model, raw["validation"], eval_features, cfg)
    summary = {
        "span_metrics": report["span_metrics"],
        "ranking_metrics": report["ranking_metrics"],
        "calibration": report["calibration"],
        "train_seconds": history["train_seconds"],
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/training_report.json", "w", encoding="utf-8") as f:
        json.dump({"loss": history["loss"], "epoch_eval": history["epoch_eval"], **summary},
                  f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
