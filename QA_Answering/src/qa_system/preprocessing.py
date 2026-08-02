"""Tokenisation and feature construction (Requirement 4.1 / 4.3).

Three entry points:

  prepare_train_features       raw example -> (input_ids, start_positions, end_positions)
  prepare_validation_features  raw example -> (input_ids, offset_mapping, example_id)
  featurise_single             (question, context) -> features for live inference

All three use the *same* sliding-window tokenisation, which is what keeps
training-time and serving-time behaviour identical.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import QAConfig


def _tokenise(questions, contexts, tokenizer, cfg: QAConfig):
    """Shared call: truncate the context only, and emit overlapping windows."""
    return tokenizer(
        questions,
        contexts,
        truncation="only_second",          # never truncate the question
        max_length=cfg.max_seq_length,
        stride=cfg.doc_stride,             # overlap between windows
        return_overflowing_tokens=True,    # -> one feature per window
        return_offsets_mapping=True,       # token -> character span
        padding="max_length" if cfg.pad_to_max_length else False,
    )


# --------------------------------------------------------------------------- #
# Training features
# --------------------------------------------------------------------------- #
def prepare_train_features(examples: Dict[str, List], tokenizer, cfg: QAConfig):
    """Convert character-level answer spans into token start/end positions.

    A window that does not fully contain the answer is labelled with the CLS
    index, i.e. treated as 'no answer' - this is exactly the signal that lets
    the model learn to abstain (SQuAD v2 style).
    """
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = _tokenise(questions, contexts, tokenizer, cfg)
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: List[int] = []
    end_positions: List[int] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)   # 0 = question, 1 = context, None = special
        answers = examples["answers"][sample_map[i]]

        # Unanswerable question -> point at CLS
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Boundaries of the context inside this window
        tok_start = 0
        while sequence_ids[tok_start] != 1:
            tok_start += 1
        tok_end = len(input_ids) - 1
        while sequence_ids[tok_end] != 1:
            tok_end -= 1

        # Answer not fully inside this window -> label as no-answer
        if not (offsets[tok_start][0] <= start_char and offsets[tok_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while tok_start <= tok_end and offsets[tok_start][0] <= start_char:
            tok_start += 1
        start_positions.append(tok_start - 1)

        while offsets[tok_end][1] >= end_char:
            tok_end -= 1
        end_positions.append(tok_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


# --------------------------------------------------------------------------- #
# Validation / prediction features
# --------------------------------------------------------------------------- #
def prepare_validation_features(examples: Dict[str, List], tokenizer, cfg: QAConfig):
    """Keep offsets + example id so predicted token spans map back to characters."""
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = _tokenise(questions, contexts, tokenizer, cfg)
    sample_map = tokenized.pop("overflow_to_sample_mapping")

    tokenized["example_id"] = []
    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        tokenized["example_id"].append(examples["id"][sample_map[i]])
        # Mask offsets that do not belong to the context so they can never be
        # selected as part of an answer.
        tokenized["offset_mapping"][i] = [
            off if sequence_ids[k] == 1 else None
            for k, off in enumerate(tokenized["offset_mapping"][i])
        ]
    return tokenized


# --------------------------------------------------------------------------- #
# Live inference features (no `datasets` dependency)
# --------------------------------------------------------------------------- #
def featurise_single(question: str, context: str, tokenizer, cfg: QAConfig) -> Tuple[Any, List]:
    """Tokenise one (question, context) pair for serving.

    Returns the encoding (tensors ready for the model) and the per-window
    offset mapping with non-context tokens masked to None.
    """
    enc = tokenizer(
        question.lstrip(),
        context,
        truncation="only_second",
        max_length=cfg.max_seq_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    offsets_per_window: List[List] = []
    n_windows = enc["input_ids"].shape[0]
    for i in range(n_windows):
        seq_ids = enc.sequence_ids(i)
        offsets_per_window.append(
            [
                tuple(off) if seq_ids[k] == 1 else None
                for k, off in enumerate(enc["offset_mapping"][i].tolist())
            ]
        )

    model_inputs = {
        k: v for k, v in enc.items()
        if k in ("input_ids", "attention_mask", "token_type_ids")
    }
    return model_inputs, offsets_per_window


def describe_windowing(question: str, context: str, tokenizer, cfg: QAConfig) -> Dict[str, Any]:
    """Diagnostics for the report: how a long passage is split into windows."""
    enc = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=cfg.max_seq_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    windows = []
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        ctx_offsets = [o for k, o in enumerate(enc["offset_mapping"][i]) if seq_ids[k] == 1]
        windows.append(
            {
                "window": i,
                "n_tokens": len(enc["input_ids"][i]),
                "char_start": ctx_offsets[0][0],
                "char_end": ctx_offsets[-1][1],
            }
        )
    return {
        "n_context_chars": len(context),
        "n_windows": len(windows),
        "windows": windows,
    }
