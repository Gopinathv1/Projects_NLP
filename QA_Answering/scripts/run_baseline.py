#!/usr/bin/env python
"""Out-of-the-box Hugging Face pipeline, and a side-by-side with the fine-tuned model.

    python scripts/run_baseline.py --question "..." --context "..."
    python scripts/run_baseline.py --compare --model-dir artifacts/qa-model --question "..." --context "..."
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_system.baseline import baseline_answer, build_qa_pipeline  # noqa: E402
from qa_system.config import QAConfig  # noqa: E402


def show(title, answers):
    print(f"\n{title}")
    print("-" * len(title))
    for c in answers:
        print(f"  {c['rank']}. {c['text']:<45} {c['confidence']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-model", default=QAConfig().baseline_model_name)
    p.add_argument("--model-dir", default="artifacts/qa-model")
    p.add_argument("--question", required=True)
    p.add_argument("--context", action="append", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--compare", action="store_true",
                   help="also run the fine-tuned checkpoint for comparison")
    args = p.parse_args()

    pipe = build_qa_pipeline(args.baseline_model)
    show(f"Hugging Face pipeline ({args.baseline_model})",
         baseline_answer(pipe, args.question, args.context, top_k=args.top_k))

    if args.compare:
        from qa_system.inference import RankedQAPipeline
        qa = RankedQAPipeline.from_pretrained(args.model_dir)
        show(f"Fine-tuned ({args.model_dir})",
             qa.answer(args.question, args.context, top_k=args.top_k)["answers"])


if __name__ == "__main__":
    main()
