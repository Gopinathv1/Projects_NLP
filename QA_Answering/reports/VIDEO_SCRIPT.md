# Explanation video — script (target 6–7 minutes)

Required coverage: (a) approach and main choices, (b) preprocessing → answer ranking, (c) one
question end to end. Screen-record the notebook and the Flask app.

**0:00–0:40 · Framing.** Question + one or more passages → ranked answers with confidence scores,
most likely first, plus a clear "no answer" when nothing fits. Two choices up front: extractive not
generative (the answer must come from the context; a span head can't hallucinate), and SQuAD v2 not
v1.1 (unanswerable questions make abstention learnable).

**0:40–1:40 · Code tour.** Show `src/qa_system/` — one module per stage. Then show `app/qa_web/`: the
application factory, blueprints (web/api), and the service layer that's the only place torch is
imported. Say why: the HTTP layer is decoupled and testable with a stub, no model needed.

**1:40–2:40 · Out-of-the-box first (Req. 4.3).** Run the baseline pipeline cell — ranked answers with
no training. Then the multi-passage example where an answer appearing in two passages outranks a
lone one, and explain duplicate aggregation.

**2:40–3:40 · Preprocessing.** Windowing cell — a long passage becomes N overlapping windows; explain
why overlap matters. Alignment cell — character spans → token positions via offset_mapping, verified
lossless before training. The `[CLS]` labelling rule is how "no answer" is taught.

**3:40–4:40 · Scoring and ranking (Req. 4.4).** This is the heart of the brief. Show the candidate
list with both the probability and the raw logit sum. Explain why probability, not logit sum:
comparability across passages. Show the cross-passage ranked list and the aggregation on/off toggle.

**4:40–5:40 · Evaluation.** EM/F1 for rank-1, then Hit@k / MRR for the ranking — explain that EM only
sees position 1, so ranking needs its own metrics. Show the calibration bars. Show the baseline vs
fine-tuned comparison and read your numbers.

**5:40–7:00 · One question end to end (part c).** Open the Flask app. Load the two-passage manual
sample, ask "How long is the warranty?", show the ranked answers, the confidence bars, and click the
top answer to highlight the span in its passage. Move the top-k slider. Flip the engine toggle to the
baseline and back. Ask "Who is the CFO?" against a passage that doesn't contain it and show the "no
answer" behaviour. Close with the two next steps: a retriever in front, and a cross-encoder re-ranker.
