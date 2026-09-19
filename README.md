# Autonomous AI Customer Support Agent

End-to-end AmazonHelp support agent for the **Hiver SDE Intern** take-home:
intent classification → train-only historical retrieval → grounded reply drafting → AUTO_HANDLE / ESCALATE with explicit reason codes.

**Honesty contract:** every metric below states its data source. Heuristic agreement is not human accuracy. Curated safety-suite metrics are not production benchmarks.

---

## A. Project overview

This repository turns the public *Customer Support on Twitter* dataset into a reproducible support agent focused on **AmazonHelp**.

It is designed to be:
- runnable from this README
- leakage-safe (conversation-level splits; train-only retrieval)
- interview-explainable (small modules, explicit policy)
- honest about what has and has not been human-evaluated

---

## B. Problem framing

False auto-handling (sending a wrong/hallucinated reply) is more costly than over-escalation. The system therefore uses an **escalation-first** safety policy and only auto-handles when intent confidence, retrieval evidence, and risk checks all pass.

---

## C. Architecture

```text
Incoming customer message
        │
        ▼
Preprocess / clean
        │
        ▼
Intent classifier (TF-IDF + calibrated logistic regression)
  → intent name + confidence
        │
        ▼
Historical retrieval (TF-IDF nearest neighbors, TRAIN split only)
  → top-k evidence cases + similarity
        │
        ▼
Escalation policy (ordered safety checks)
  → AUTO_HANDLE or ESCALATE + reason_code
        │
        ▼
Grounded reply generator (Gemini, or MOCK_LLM)
  → Pydantic SupportResponseSchema
```

---

## D. Dataset and AmazonHelp selection

| Item | Value | Source |
|------|-------|--------|
| Dataset | Customer Support on Twitter (`twcs.csv`) | Kaggle `thoughtvector/customer-support-on-twitter` |
| Total tweets | 2,811,774 | `results/dataset_profile.json` |
| Selected brand | AmazonHelp | programmatic volume / depth / diversity |
| AmazonHelp conversations | 82,493 | processed splits |
| Train / Val / Test | 65,994 / 8,249 / 8,250 | conversation-level 80/10/10, seed=42 |

Place the raw file at:

```text
data/raw/twcs.csv
```

(The raw CSV is gitignored; do not commit it.)

---

## E. Intent taxonomy

Eight data-derived support intents (`data/golden/taxonomy.yaml`):

| Intent | Meaning |
|--------|---------|
| `shipping_delay` | Late / stuck tracking |
| `missing_item` | Delivered box incomplete |
| `order_cancellation` | Cancel before/during fulfillment |
| `refund_return_request` | Return / refund |
| `account_access_issue` | Login / password / lockout |
| `payment_billing_issue` | Double charge / billing error |
| `product_defect_damage` | Broken / defective item |
| `general_inquiry_feedback` | Other / unclear |

**Training labels today are heuristic keyword labels**, not human gold.

---

## F. Data splitting / leakage prevention

- Splits are at **conversation ID** level (never tweet-level random mix).
- Retrieval index is built from **train only**.
- Golden candidates are sampled from **test/val only**.
- Automated check: `evaluation/leakage_check.py` (also run via `evaluation.run_all`).
- Current result: **0 train/test conversation ID overlap**.

---

## G. Retrieval methodology

- Vectorizer: TF-IDF (1–2 grams)
- Index: sklearn `NearestNeighbors` (cosine)
- Corpus: `data/processed/train.jsonl` only (~65,994 cases)
- Queries for retrieval eval: held-out `test.jsonl` (8,250)

**Proxy metrics (not IR Recall@K):**
- `IntentMatch@1` / `IntentMatch@5`: whether a retrieved neighbor shares the query’s **heuristic** intent
- These are topical-grouping diagnostics, **not** true recall (no gold relevant-doc IDs exist)

---

## H. Escalation policy

Ordered checks in `src/escalation/policy.py`:

1. `INSUFFICIENT_CONTEXT`
2. `SENSITIVE_REQUEST`
3. `ACCOUNT_SPECIFIC_ACTION_REQUIRED`
4. `LOW_INTENT_CONFIDENCE` (default threshold 0.70)
5. `NO_RELEVANT_HISTORICAL_EVIDENCE` (default similarity threshold 0.65)
6. else `AUTO_HANDLE`

Thresholds are safety defaults — **not** tuned to inflate auto-handle rate.

---

## I. Gemini generation

Real LLM path uses **Google Gemini** (`google-generativeai`) with structured JSON matching `SupportResponseSchema`.

Rules enforced in prompts:
- ground on retrieved historical cases
- do not invent order/account details
- do not claim actions unless supported
- respect pre-decided escalation

---

## J. Mock mode

```env
MOCK_LLM=true
```

Uses a deterministic mock generator/judge for offline tests and evaluation without API keys. Mock replies are prefixed with `[MOCK]`.

```env
MOCK_LLM=false
GEMINI_API_KEY=...
```

Uses Gemini for generation/judging.

---

## K. Installation

```bash
git clone <repo-url>
cd AI_Customer_Support_Agent
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## L. Environment configuration

Copy the safe example:

```bash
cp .env.example .env
```

`.env.example` contains placeholders only:

```env
GEMINI_API_KEY=your_gemini_api_key_here
MOCK_LLM=true
```

`.env` is gitignored. Never commit real keys. OpenAI is **not** used.

---

## M. Exact commands to reproduce results

### 1) Dataset prepare (requires `data/raw/twcs.csv`)

```bash
python -m src.data.prepare
```

### 2) Golden candidate generation (~200, not human-reviewed)

```bash
python -m src.data.label
python -m src.data.label --status
```

### 3) Tests

```bash
python -m pytest
```

### 4) Full automated evaluation

```bash
set MOCK_LLM=true   # Windows PowerShell: $env:MOCK_LLM="true"
python -m evaluation.run_all
```

### 5) Full held-out pipeline dump (optional / slower; writes 8,250 lines)

```bash
python -m evaluation.pipeline_eval
python -m evaluation.failure_analysis
```

Artifacts land in `results/`.

---

## N. Tests

```bash
python -m pytest
```

Covers conversations, intent, retrieval leakage terminology, escalation, generation schema, pipeline wiring, failure-analysis prioritization, and mock/Gemini mode selection.

---

## O. Automated evaluation (current evidence)

| Metric | Value | Dataset / Benchmark Type |
|--------|--------|--------------------------|
| **Conversation-level Leakage** | `PASS — 0 overlap` | REAL DATASET — all 82,493 conversations |
| **Duplicate-text contamination** | 10 messages (independent convs) | INFORMATIONAL — trivially generic messages |
| **Majority Baseline Accuracy** | 0.748 | HEURISTIC labels on test split (NOT human gold) |
| **Majority Baseline Macro F1** | 0.107 | HEURISTIC labels on test split |
| **TF-IDF LogReg Accuracy** | 0.9445 | HEURISTIC labels on test split |
| **TF-IDF LogReg Macro F1** | 0.8792 | HEURISTIC labels on test split |
| **Final Classifier Accuracy** | 0.9509 | HEURISTIC labels on test split (8,250 queries) |
| **Final Classifier Macro F1** | 0.8966 | HEURISTIC labels on test split |
| **Retrieval Corpus Size** | 65,994 resolved train conversations | REAL DATASET — train split ONLY |
| **Held-out Query Count** | 8,250 | REAL DATASET — test split |
| **Retrieval Train/Test Overlap** | 0 conversations | FIXED (was 8,250 — data leakage) |
| **IntentMatch@1** | 0.7589 | PROXY metric (NOT Recall@1) — heuristic labels |
| **IntentMatch@5** | 0.9038 | PROXY metric (NOT Recall@5) — heuristic labels |
| **Avg Top-1 Retrieval Similarity** | 0.4681 | REAL DATASET — TF-IDF cosine, no self-query |
| **Escalation Auto-Handle Rate** | 37.5% (8-case suite) | CURATED SAFETY SUITE — 8 cases only |
| **Escalation False Auto-Handle** | 0/5 risk cases | CURATED SAFETY SUITE — NOT production |
| **LLM Judge Discrimination Test** | Mean 8.0/10 across 4 cases | SYNTHETIC TEST — judge calibration ONLY |
| **Actual Agent Reply Quality** | NOT YET MEASURED | Requires real pipeline outputs + human ratings |
| **Human / LLM Agreement** | **BLOCKED — REQUIRES HUMAN INPUT** | Missing `human_ratings.json` |
| **Golden Benchmark (Intent)** | **BLOCKED — REQUIRES HUMAN INPUT** | Need 150–250 reviewed examples |
| **Human-reviewed golden examples** | **0** | `data/golden/golden_set.jsonl` |

---

## P. Human golden-set workflow (PENDING human work)

Assignment requires **150–250 human-labelled** examples. Candidates exist (~200) but are **not** human-reviewed.

```bash
python -m src.data.label                 # regenerate candidates if needed
python -m src.data.label --review        # interactive CLI
python -m src.data.label --promote       # write human-reviewed rows into golden_set.jsonl
python -m src.data.label --status
```

Guidelines: `data/golden/labeling_guidelines.md`

Automated evaluation **refuses** to treat heuristic seeds as human labels (`is_human_reviewed` must be true and `human_verified_intent` set).

---

## Q. LLM judge methodology

5-dimension rubric (0–2 each → 0–10 total): Correctness, Groundedness, Relevance, Completeness, Professionalism.

`evaluation/llm_judge.py` currently runs a **discrimination test** on 4 controlled replies to verify the judge penalizes hallucinations/poor replies. That mean score is **not** agent quality.

---

## R. Human-vs-LLM agreement methodology (BLOCKED until ratings exist)

1. Prepare shared examples (same reply text for both raters):

```bash
python -m evaluation.human_judge_agreement --prepare-examples
```

2. Create `data/golden/human_ratings.json` with real human 0–10 scores on those replies.

3. Re-run:

```bash
python -m evaluation.human_judge_agreement
```

Until the file exists, results report **BLOCKED — REQUIRES HUMAN INPUT**. No fake agreement values are generated.

---

## S. Baselines

Evaluated on the **same** held-out test split and **same** heuristic label source as the final classifier:

1. Majority class baseline
2. TF-IDF + Logistic Regression baseline
3. Final calibrated intent classifier

Human-golden evaluation is a **separate** track unlocked only after ≥150 human-reviewed labels.

---

## T. Failure analysis

Built from real `results/pipeline_test_outputs.jsonl` with prioritized categories:

1. `INTENT_MISMATCH_VS_HEURISTIC` (diagnostic vs heuristic — not human gold)
2. `LOW_INTENT_CONFIDENCE_ESCALATION` (expected safety)
3. `INSUFFICIENT_CONTEXT_ESCALATION` (expected safety)
4. `SENSITIVE_REQUEST_ESCALATION` (expected safety)
5. `ACCOUNT_SPECIFIC_ACTION_ESCALATION` / `LOW_RETRIEVAL_SIMILARITY`

Escalation reason codes take priority over raw similarity (fixes mislabelling low-confidence cases as retrieval failures). Not every escalation is called a model failure. Hypotheses are stored separately without invented frequencies.

---

## U. Limitations

- Heuristic training/eval labels create circular agreement risk.
- Public tweets often end in DM redirects; “resolutions” are incomplete.
- TF-IDF retrieval is lexical, not semantic.
- Escalation curated suite is tiny (8 cases).
- No live order/account APIs.
- Human golden set and human/LLM agreement are still pending.

---

## V. What is misleading about my headline number?

1. **95.09% intent accuracy** is agreement with **heuristic** labels on the held-out test split — **not** human-labelled accuracy. Training labels use the same keyword rules, so high scores partly measure rule reconstruction.
2. **IntentMatch@K** is a proxy topical match — **not** Recall@K.
3. **0% false auto-handling** is on an **8-case curated safety suite**, not production traffic.
4. **8.0/10 LLM judge mean** is a **4-case discrimination test**, not measured agent reply quality.
5. Historical retrieval similarity can look healthy while retrieved “resolutions” are only “please DM us” triage text.

---

## W. One-more-week plan

1. Complete 150–250 human golden labels and report separate human-intent metrics.
2. Collect ≥30 human reply ratings on shared pipeline outputs; unlock agreement stats.
3. Add dense retrieval / cross-encoder re-ranker; keep train-only indexing.
4. Add a clarification turn before permanent `INSUFFICIENT_CONTEXT` escalation.
5. Judge a sampled set of real pipeline outputs (not only discrimination cases).

---

## X. Decision log

See [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) (15 engineering decisions actually reflected in the implementation).


---

## Y. Report links

- Technical report: [`report/report.md`](report/report.md)
- Decision log: [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md)
- Final audit: [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md)
- Implementation status: [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)
- Labeling guidelines: [`data/golden/labeling_guidelines.md`](data/golden/labeling_guidelines.md)
- Taxonomy: [`data/golden/taxonomy.yaml`](data/golden/taxonomy.yaml)

### Citations

1. Thought Vector / Kaggle — *Customer Support on Twitter* dataset.
2. Pedregosa et al. — scikit-learn (TF-IDF, logistic regression, nearest neighbors).
3. Google — Gemini API (`google-generativeai`) for structured generation/judging.
4. Pydantic — structured output validation.
5. Hiver SDE Intern take-home assignment brief (problem requirements).
