#!/usr/bin/env python
"""Ranked answers from the command line.

    python scripts/run_predict.py --question "How long is the warranty?" \
        --context "The device carries a two-year limited warranty." \
        --context "Accessories are covered for ninety days." --top-k 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_system.inference import RankedQAPipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="artifacts/qa-model")
    p.add_argument("--question", required=True)
    p.add_argument("--context", action="append", default=[],
                   help="repeat the flag for several passages")
    p.add_argument("--context-file", action="append", default=[])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true", help="print the raw JSON response")
    args = p.parse_args()

    contexts = list(args.context)
    for path in args.context_file:
        with open(path, encoding="utf-8") as f:
            contexts.append(f.read())
    if not contexts:
        p.error("provide at least one --context or --context-file")

    qa = RankedQAPipeline.from_pretrained(args.model_dir)
    result = qa.answer(args.question, contexts, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"\nQ: {result['question']}")
    print(f"   {result['n_passages']} passage(s), {result['n_windows']} window(s), "
          f"{result['latency_ms']} ms\n")
    for cand in result["answers"]:
        src = "-" if cand["passage_index"] is None else f"p{cand['passage_index']}"
        support = f" (in {cand['support']} passages)" if cand.get("support", 1) > 1 else ""
        print(f"  {cand['rank']}. {cand['text']:<45} {cand['confidence']:.4f}  [{src}]{support}")


if __name__ == "__main__":
    main()
