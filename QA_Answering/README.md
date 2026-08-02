# Extractive Question Answering — ranked answers with confidence scores

Given a **question** and **one or more context passages**, the system returns a **ranked list of
candidate answers**, each a span copied from a passage and carrying a **confidence score**, most
likely answer first. It also answers **out of the box** via a Hugging Face pipeline and supports
**fine-tuning** your own model. The service is a **modular Flask** application.

Built against brief `NX-NLP-2026-018` (Verbalis AI, NLP Engineer take-home).

---

## 1. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Answer immediately, with NO training, using the Hugging Face pipeline
python scripts/run_baseline.py \
  --question "How long is the warranty?" \
  --context "The device carries a two-year limited warranty." \
  --context "Accessories are covered for ninety days."

# Fine-tune (small run: ~20 min on a T4)
python scripts/run_train.py --max-train-samples 8000 --max-eval-samples 1500

# Evaluate: span quality (EM/F1) AND ranking quality (Hit@k, MRR)
python scripts/run_eval.py --model-dir artifacts/qa-model

# Ranked answers from the CLI, over several passages
python scripts/run_predict.py --top-k 5 \
  --question "Where is the head office?" \
  --context "Head office is in Bengaluru." \
  --context "All functions report to the Bengaluru head office."

# Launch the Flask web app
python app/wsgi.py                     # http://127.0.0.1:5000
```

The web app runs **before any training**: if no fine-tuned checkpoint exists at `QA_MODEL_DIR`, it
falls back to a ready-made SQuAD model, so the endpoint is never dead.

---

## 2. Project layout

```
extractive-qa-ranked/
├── src/qa_system/              # the ML library — importable, testable
│   ├── config.py               # QAConfig: one dataclass drives everything (incl. top_k)
│   ├── data.py                 # HF hub datasets + local SQuAD-format JSON
│   ├── preprocessing.py        # sliding window, char->token span alignment
│   ├── modeling.py             # model/tokenizer build, save, reload
│   ├── baseline.py             # out-of-the-box HF question-answering pipeline (Req. 4.3)
│   ├── train.py                # native PyTorch fine-tuning loop
│   ├── postprocessing.py       # span extraction, confidence scoring, cross-passage ranking
│   ├── metrics.py              # EM/F1 + Hit@k, Recall@k, MRR, calibration
│   ├── evaluation.py           # batched inference + full metric report
│   └── inference.py            # RankedQAPipeline — the serving object
│
├── app/                        # MODULAR FLASK application (Req. 4.6)
│   ├── wsgi.py                 # entrypoint for `python app/wsgi.py` and gunicorn
│   └── qa_web/                 # the Flask package
│       ├── __init__.py         # create_app() application factory
│       ├── settings.py         # env-driven Dev/Prod/Test config classes
│       ├── errors.py           # ApiError + centralised error handlers
│       ├── logging_utils.py    # logging configuration
│       ├── samples.py          # demo passages
│       ├── blueprints/
│       │   ├── web.py          # HTML route (single-page UI)
│       │   └── api.py          # JSON API: /api/health, /api/info, /api/answer
│       ├── services/
│       │   ├── qa_service.py   # model lifecycle + engine dispatch (only torch touchpoint)
│       │   └── validation.py   # request parsing / guard rails
│       ├── templates/index.html
│       └── static/             # styles.css, app.js
│
├── scripts/                    # run_train · run_eval · run_predict · run_baseline
├── notebooks/                  # end-to-end walkthrough (Jupyter / Colab)
├── tests/                      # unit + API contract tests (offline)
├── data/                       # sample SQuAD-format files
└── reports/                    # REPORT.md, VIDEO_SCRIPT.md, generated metrics
```

The notebook and the Flask service both **import** `qa_system` — no logic is duplicated.

---

## 3. Why the Flask app is structured this way

The brief grades code quality (6.5) and asks for a Flask app (4.6). A single-file Flask script would
satisfy neither well, so the app is a **package built around an application factory**:

- **`create_app(config_name, qa_service=None)`** constructs and wires everything. It takes an
  optional injected service, which is what lets the test suite build a real app around a **stub
  model** and exercise the entire HTTP stack in milliseconds — no torch, no download.
- **Blueprints** split the surface: `web` (HTML) and `api` (JSON, under `/api`). Routes stay thin —
  they validate, call the service, and return JSON.
- **A service layer** (`QAService`) owns the model: lazy loading, caching, thread-safe first-load,
  and dispatch between the fine-tuned engine and the baseline pipeline. **It is the only place that
  imports torch**, so the web layer is decoupled from the ML stack.
- **Validation** is its own module raising typed `ApiError`s; **error handling** is centralised so
  every JSON error has one stable shape (`{error, code, status}`).
- **Settings** are environment-driven config classes (Dev/Prod/Test), the twelve-factor way.

### API

```bash
curl -s http://127.0.0.1:5000/api/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long is the warranty?",
       "contexts": ["The XR-500 carries a two-year limited warranty.",
                    "Accessories are covered for ninety days."],
       "top_k": 3,
       "engine": "finetuned"}'
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Single-page UI: multiple passages, top-k slider, engine toggle, span highlighting |
| `/api/health` | GET | Liveness + whether a model is loaded |
| `/api/info` | GET | Active model and decoding settings |
| `/api/answer` | POST | `{question, contexts[] or context, top_k, engine}` → ranked answers |

`engine` is `"finetuned"` (this project's model) or `"baseline"` (stock HF pipeline).

### Production

```bash
pip install gunicorn
gunicorn --chdir app wsgi:app -b 0.0.0.0:5000 -w 2
```

### Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `QA_ENV` | `dev` | `dev` / `prod` / `test` config profile |
| `QA_MODEL_DIR` | `artifacts/qa-model` | fine-tuned checkpoint directory |
| `QA_FALLBACK_MODEL` | `distilbert-base-cased-distilled-squad` | used when the checkpoint is absent |
| `QA_BASELINE_MODEL` | `distilbert-base-cased-distilled-squad` | model behind the "baseline" engine |
| `QA_EAGER_LOAD` | `0` | `1` loads the model at boot instead of first request |
| `QA_HOST` / `QA_PORT` | `127.0.0.1` / `5000` | bind address |
| `QA_MAX_PASSAGES` | `20` | request guard rail |

---

## 4. How it works

**Preprocessing (4.1).** Question and context are tokenised as a pair with `truncation="only_second"`
so the question is never cut. Long passages become overlapping windows (`max_seq_length=384`,
`stride=128`), so nothing is silently truncated and a boundary-straddling answer stays whole in one
window.

**Representation (4.2).** A pretrained transformer produces a contextual embedding per token. Swap
`--model-name` for `bert-base-uncased`, `roberta-base`, `deberta-v3-base` — no other change.

**Out-of-the-box + fine-tuning (4.3).** `qa_system.baseline` wraps the HF `question-answering`
pipeline (adding multi-passage support and duplicate aggregation); `qa_system.train` fine-tunes with
a native PyTorch loop. The baseline is also the honest reference point: a fine-tune that can't beat
it isn't worth shipping.

**Answer extraction, confidence, ranking (4.4).** Per window, logits are masked to legal positions
and softmaxed, and each span scores `P(start)·P(end)` — a value in `[0,1]` that is **comparable
across passages**, which a raw logit sum is not. Candidates from every window and passage are pooled;
identical answers are merged (their confidence combined, `support` counting how many passages
produced them), so a fact corroborated by two documents outranks a lone mention. The list is returned
in descending confidence.

**Save / reload + user-set top-k (4.5).** `save_artifacts` writes weights, tokenizer and
`qa_config.json` to one directory; reloading restores identical behaviour. `top_k` is a request
parameter (and a config default), clamped to `max_top_k`.

**Evaluation.** Span quality — Exact Match / F1 on the rank-1 answer, split HasAns/NoAns. Ranking
quality — **Hit@k, Recall@k, MRR**, which is what criterion 6.1 ("most likely answer first") actually
measures; EM alone only sees rank 1. Plus a **calibration** table: is a 0.8-confidence answer right
~80% of the time?

---

## 5. Results

Fill in from `reports/eval_report.json`. Reference for the small config (`distilbert-base-uncased`,
8k examples, 2 epochs, SQuAD v2 dev subset):

| Metric | Value |
|---|---|
| Exact Match (rank-1) | _fill in_ |
| F1 (rank-1) | _fill in_ |
| Hit@1 / Hit@5 | _fill in_ |
| MRR (exact) | _fill in_ |

The gap between Hit@1 and Hit@5 is the headroom a re-ranker would recover.

---

## 6. Tests

```bash
pytest tests/ -q
```

`test_metrics.py` and `test_postprocessing.py` cover scoring and ranking (numpy only, offline).
`test_validation.py` and `test_api.py` cover the Flask layer with an **injected stub service** — the
full factory → blueprint → service → JSON path plus error handling, no model needed.
`test_preprocessing.py` downloads a tokenizer and skips if offline.

---

## 7. Libraries

PyTorch · Hugging Face Transformers · Hugging Face Datasets · NumPy · pandas · tqdm · Flask ·
Matplotlib (notebook) · pytest (tests). Optional: gunicorn (production serving).
