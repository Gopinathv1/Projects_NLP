# Extractive QA with Ranked Answers — Technical Report

**Assessment ref:** NX-NLP-2026-018 · **Candidate:** _<your name>_ · **Date:** _<date>_

> Fill bracketed placeholders and the results table from your run (`reports/eval_report.json`).

## 1. Problem
Return the answer to a question from one or more passages, using a pretrained transformer, as a
**ranked list of candidates each with a confidence score**, most likely first — and handle the case
where no passage supports an answer.

## 2. Approach
Extractive span prediction: a pretrained encoder plus a linear head giving per-token start/end
scores. Chosen over generation because the answer must come *directly* from the context — a span
head cannot invent text, which matters for manuals, guidelines and policies.

Two inference paths, as the brief requires: an **out-of-the-box Hugging Face pipeline** (answers with
zero training, and the honest baseline every fine-tuned number is measured against) and a
**fine-tuned** model trained on SQuAD v2.0. v2 is used because its unanswerable questions teach the
model to abstain rather than force a confident wrong span onto an irrelevant passage.

## 3. Confidence and ranking (the core of this brief)
Within one forward pass, `start_logit + end_logit` ranks spans correctly, but it is unbounded and
un-normalised, so it cannot be compared **across passages**. Because answers are pooled from several
passages into one ranked list, each window's logits are masked to legal positions and softmaxed, and
a span is scored `P(start=i)·P(end=j)` — a value in `[0,1]`, comparable everywhere, and the same
convention the HF pipeline uses. Duplicate answers across passages are merged, their confidence
combined and a `support` count recorded, so a fact appearing in two independent sources outranks a
single stronger mention.

## 4. Preprocessing
Question and context tokenised as a pair (`truncation="only_second"`), long passages split into
overlapping windows (`max_seq_length=384`, `stride=128`). Character answer spans are converted to
token positions via `offset_mapping`; windows not fully containing the answer, and all unanswerable
questions, are labelled at `[CLS]`.

## 5. Fine-tuning
AdamW, lr 3e-5, 10% warm-up then linear decay, gradient clipping 1.0, AMP on CUDA, 2 epochs, native
PyTorch loop (version-stable, unlike `TrainingArguments`). Evaluation after every epoch reports span
**and** ranking metrics, so a model that ranks worse while EM improves is caught immediately.

## 6. Evaluation
- **Span quality:** Exact Match / F1 on the rank-1 answer, split HasAns / NoAns.
- **Ranking quality:** Hit@k (correct answer in the top k), Recall@k (a *useful* answer, F1 ≥ 0.5),
  MRR (how high the first correct answer sits). This is what criterion 6.1 measures; EM only sees
  rank 1.
- **Calibration:** observed accuracy per confidence bin — does 0.8 confidence mean ~80% correct?

### Results
| Metric | Value |
|---|---|
| Exact Match (rank-1) | _fill in_ |
| F1 (rank-1) | _fill in_ |
| Hit@1 / Hit@3 / Hit@5 | _fill in_ |
| MRR (exact) | _fill in_ |
| Baseline vs fine-tuned (MRR) | _fill in_ |

Config: _<model, #train examples, epochs, hardware>_.

## 7. Application (modular Flask)
An application-factory package: `create_app()` wires an env-driven **config**, two **blueprints**
(HTML `web`, JSON `api`), a **service layer** that owns the model (lazy, cached, thread-safe, the
only torch touchpoint), isolated **validation**, and centralised **error handlers** with one JSON
error shape. Routes stay thin. Because the factory accepts an injected service, the whole HTTP stack
is tested with a stub model, no download. The UI takes several passages, a top-k slider, an engine
toggle, and highlights the supporting span.

## 8. Robustness and generalisation
Verified in the notebook: answers buried past the truncation point (recovered via later windows),
messy text with irregular whitespace/markup, noisy distractor passages, and hand-written questions
over an unseen document. _Add two concrete examples from your run._

## 9. Limitations and next steps
1. **No retrieval** — the user supplies passages; a real KB needs BM25/dense retrieval feeding the
   already-built multi-passage reader.
2. **Cross-encoder re-ranker** — the Hit@1→Hit@5 gap is recoverable headroom.
3. **Calibration** — the softmax product ranks well but isn't a true probability; fit a calibrator.
4. **Bigger backbone** — DeBERTa-v3/ELECTRA-large typically adds 6–10 F1 through the same code.
5. **Serving** — ONNX + int8 for ~3× CPU latency at a fraction of a point of F1.

## 10. Libraries
PyTorch · Transformers · Datasets · NumPy · pandas · tqdm · Flask · Matplotlib · pytest · gunicorn.
