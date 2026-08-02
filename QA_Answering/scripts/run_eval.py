#!/usr/bin/env python
"""Evaluate a saved model: span quality (EM/F1) and ranking quality (Hit@k, MRR)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_system.data import load_qa_datasets  # noqa: E402
from qa_system.evaluation import evaluate_model  # noqa: E402
from qa_system.modeling import load_artifacts  # noqa: E402
from qa_system.preprocessing import prepare_validation_features  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="artifacts/qa-model",
                   help="a fine-tuned directory, or any HF hub id")
    p.add_argument("--dataset", default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", default="reports/eval_report.json")
    args = p.parse_args()

    model, tokenizer, cfg = load_artifacts(args.model_dir)
    if args.dataset:
        cfg.dataset_name = args.dataset
    if args.max_eval_samples:
        cfg.max_eval_samples = args.max_eval_samples

    raw = load_qa_datasets(cfg)
    features = raw["validation"].map(
        partial(prepare_validation_features, tokenizer=tokenizer, cfg=cfg),
        batched=True, remove_columns=raw["validation"].column_names,
    )
    report = evaluate_model(model, raw["validation"], features, cfg,
                            k_values=(1, 3, args.top_k), top_k=args.top_k)

    out = {
        "model_dir": args.model_dir,
        "n_examples": len(raw["validation"]),
        "span_metrics": report["span_metrics"],
        "ranking_metrics": report["ranking_metrics"],
        "calibration": report["calibration"],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
