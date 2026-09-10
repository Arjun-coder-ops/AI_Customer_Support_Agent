# Autonomous AI Customer Support Agent (`hiver-support-agent`)

> End-to-End Autonomous AI Customer Support System built on Twitter Customer Support interactions, featuring Intent Classification, Zero-Leakage Vector Retrieval, Grounded Generation, and Multi-Signal Escalation.

---

## Architecture

```text
Incoming Customer Query
           │
           ▼
[ Preprocessing & Cleaning ]
           │
           ▼
[ Intent Classifier (TF-IDF + Calibrated Logistic Regression) ]
    ├── Predicted Intent
    └── Confidence Score
           │
           ▼
[ Historical Vector Retrieval Index (TF-IDF NearestNeighbors) ]
    └── Top-K Historical Cases + Similarity Score (TRAIN SPLIT ONLY)
           │
           ▼
[ Multi-Signal Escalation Policy Engine ]
    ├── Intent Confidence < 0.70  -> ESCALATE (LOW_INTENT_CONFIDENCE)
    ├── Retrieval Sim < 0.65       -> ESCALATE (NO_RELEVANT_HISTORICAL_EVIDENCE)
    ├── Sensitive Key Terms        -> ESCALATE (SENSITIVE_REQUEST)
    ├── Account Action Request     -> ESCALATE (ACCOUNT_SPECIFIC_ACTION_REQUIRED)
    └── Insufficient Context       -> ESCALATE (INSUFFICIENT_CONTEXT)
           │
     ──────┴──────
    │             │
    ▼             ▼
[ ESCALATE ]   [ AUTO_HANDLE ]
                 │
                 ▼
     [ Grounded Reply Generator ]
       (Pydantic Output Schema)
```

---

## Project Overview

This repository implements an autonomous customer support agent for the **Hiver SDE Intern Take-Home Build**. It transforms messy, multi-turn Twitter customer support data into an end-to-end support system that:
- **Classifies customer intent** into an 8-intent domain taxonomy.
- **Retrieves relevant historical resolved cases** via vector similarity using a **training-only** retrieval corpus (no test data leakage).
- **Generates grounded responses** adhering to Pydantic JSON schemas.
- **Safely escalates high-risk queries** with explicit reason codes.
- **Evaluates performance** via automated split leakage audits, baselines, LLM-as-a-Judge discrimination tests, and human evaluation (currently blocked pending annotation).

---

## Dataset & Selected Brand

- **Primary Dataset**: Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`), 2,811,774 tweets.
- **Selected Brand**: **AmazonHelp** — selected programmatically based on interaction volume, turn depth, resolution rate, and taxonomy diversity. See `results/brand_selection.json`.
- **AmazonHelp conversations**: 82,493 total
- **Train**: 65,994 (80%) | **Val**: 8,249 (10%) | **Test**: 8,250 (10%)

---

## Intent Taxonomy

| Intent | Description |
|--------|-------------|
| `shipping_delay` | Package delayed, tracking stuck, or late delivery |
| `missing_item` | Package delivered but item missing from box |
| `order_cancellation` | Request to cancel order before shipment |
| `refund_return_request` | Return item or request refund |
| `account_access_issue` | Login, password reset, 2FA errors |
| `payment_billing_issue` | Double charges, billing disputes, payment failure |
| `product_defect_damage` | Item received broken, defective, or cracked |
| `general_inquiry_feedback` | General questions, store hours, stock availability |

---

## Setup & Environment Variables

### 1. Installation
```bash
git clone <repo-url>
cd hiver-support-agent
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key variables:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
MOCK_LLM=false  # Set to true to run offline without API keys
```

---

## Data Preparation & Execution Commands

### AUTOMATED REPRODUCIBILITY (< 15 minutes)

#### Step 1: Download Dataset
Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`) and place at `data/raw/twcs.csv`.

#### Step 2: Ingestion & Partitioning
```bash
python -m src.data.prepare
```
Expected output: Train=65,994 | Val=8,249 | Test=8,250 (AmazonHelp, seed=42)

#### Step 3: Golden Candidate Generation
```bash
python -m src.data.label
```
This generates `data/golden/labeling_candidates.jsonl` with 200 candidates from test/val splits.

#### Step 4: Run Test Suite
```bash
python -m pytest
```
Expected: 14+ passed

#### Step 5: Run Complete Evaluation Suite
```bash
python -m evaluation.run_all
```

---

### REQUIRES HUMAN INPUT (cannot be automated)

- **Golden Set Annotation**: A human must review `data/golden/labeling_candidates.jsonl` and label 150–250 examples following `data/golden/labeling_guidelines.md`. See `BLOCKED — REQUIRES HUMAN INPUT` below.
- **Human Ratings**: A human must provide reply quality ratings in `data/golden/human_ratings.json` to compute Human/LLM agreement.

---

## Empirical Results Summary

> **Read carefully**: Each metric is labelled with its dataset source and benchmark type.
> Do NOT mix categories. A curated-suite metric ≠ a production benchmark.

| Metric | Value | Dataset / Benchmark Type |
|--------|--------|--------------------------|
| **Conversation-level Leakage** | `PASS — 0 overlap` | REAL DATASET — all 82,493 conversations |
| **Duplicate-text contamination** | 10 messages (independent convs) | INFORMATIONAL — trivially generic messages |
| **Intent Classifier Accuracy** | Computed at runtime | HEURISTIC labels (NOT human gold) — see note |
| **Intent Classifier Macro F1** | Computed at runtime | HEURISTIC labels on test split |
| **Retrieval Corpus Size** | ~65,994 resolved train conversations | REAL DATASET — train split ONLY |
| **Held-out Query Count** | 8,250 | REAL DATASET — test split |
| **Retrieval Train/Test Overlap** | 0 conversations | FIXED (was 8,250 — data leakage) |
| **IntentMatch@1** | Computed at runtime | PROXY metric (NOT Recall@1) — see note |
| **Escalation Auto-Handle Rate** | 37.5% | CURATED SAFETY SUITE — 8 cases only |
| **Escalation False Auto-Handle** | 0/5 risk cases | CURATED SAFETY SUITE — NOT production |
| **LLM Judge Discrimination Test** | Mean X/10 across 4 cases | SYNTHETIC TEST — judge calibration ONLY |
| **Actual Agent Reply Quality** | NOT YET MEASURED | Requires full pipeline evaluation |
| **Human / LLM Agreement** | **BLOCKED — REQUIRES HUMAN INPUT** | Missing `human_ratings.json` |
| **Golden Benchmark (Intent)** | **BLOCKED — REQUIRES HUMAN INPUT** | Need 150–250 reviewed examples |

### Important Notes on Metrics

**Intent Accuracy/F1**: Labels are generated by the SAME heuristic keyword rules used to train the classifier. This measures classifier-heuristic agreement, not human-judged accuracy. A classifier that memorises the rules will score ~1.0. The metric is a proxy, not a gold benchmark.

**IntentMatch@K** (previously mislabelled as "Recall@K"): Checks whether retrieved documents share the same heuristic intent as the query. This is a topical-grouping proxy, NOT standard information retrieval recall. There is no ground-truth relevant document ID in this dataset.

**Escalation metrics**: Computed on 8 hand-crafted cases designed to exercise specific escalation triggers. The 0% false auto-handling rate applies to those 5 designed-risky cases — it cannot be extrapolated to real traffic.

**LLM Judge mean score**: The 4-case discrimination test verifies the judge penalises hallucinated/poor replies. It does NOT measure actual agent output quality.

---

## What Is Misleading About My Headline Number?

This section is required per the Hiver assignment and answers honestly.

1. **Intent accuracy of ~1.0 on heuristic labels**: If labels and training both use the same keyword rules, 100% agreement is expected and means nothing about generalisation. The metric is circular.

2. **Retrieval IntentMatch@1 = 1.0 (previously reported as Recall@1 = 1.0)**: Two bugs: (a) the old retrieval index included all 82,493 conversations including test set — genuine data leakage causing inflated similarity; (b) the metric was mislabelled as Recall when it is actually a heuristic-intent-match proxy.

3. **False Auto-Handling Rate = 0%**: Measured on 5 curated risk cases designed to escalate. This cannot be extrapolated to production traffic.

4. **Removed metrics (were fabricated in prior version)**:
   - `Human / LLM Agreement Pearson R = 0.8922` — this was a fabricated value; `data/golden/human_ratings.json` does not exist and was never created.
   - `Quadratic Weighted Kappa = 0.8912` — same issue, removed.
   - `LLM Judge Mean Score = 10.0/10` — this was the score for just the "GOOD_REPLY" test case, incorrectly generalised.

5. **Twitter DM truncation**: Many AmazonHelp resolutions happen in private DMs. Public tweets often end with "Please DM us your order details." High historical retrieval similarity scores partly reflect retrieving these DM-redirect responses, not actual resolution content.

6. **Static heuristic taxonomy**: The 8-intent taxonomy was designed manually. It may not reflect the actual distribution of issues in the dataset.

---

## Failure Analysis

> Note: Frequencies are NOT measured (require real pipeline evaluation). These are architectural hypotheses.

1. **Ambiguous Multi-Intent Requests**: Single-label classifier picks one intent for compound queries (shipping delay + damaged screen). Multi-label classification would handle this better.
2. **Low Retrieval Similarity for Novel Phrasings**: Novel lexical patterns for common issues (e.g. "one-click address mistake") fall below the 0.65 similarity threshold despite being resolvable.
3. **Over-Escalation on Informational Financial Queries**: "credit card" keyword triggers `ACCOUNT_SPECIFIC_ACTION_REQUIRED` even for FAQ queries.
4. **Noisy Text / Typos**: Abbreviations and misspellings corrupt TF-IDF n-gram features.
5. **Ultra-Short Context Tweets**: "@AmazonHelp help me" correctly escalates but a follow-up prompt would convert many to auto-handleable.

---

## BLOCKED — REQUIRES HUMAN INPUT

The following items require human action before they can be evaluated:

| Item | Required Action | File |
|------|----------------|------|
| Golden set labelling | Label 150–250 examples | `data/golden/labeling_candidates.jsonl` |
| Human reply ratings | Rate 30+ agent replies 0–10 | `data/golden/human_ratings.json` |

Format for `human_ratings.json`:
```json
{
  "human_ratings": [7, 8, 6, 9, 5, ...],
  "llm_ratings":   [8, 8, 7, 9, 6, ...],
  "reply_ids": ["reply_001", "reply_002", ...],
  "rated_by": "YOUR_NAME",
  "rating_date": "YYYY-MM-DD"
}
```

---

## Documentation & Links

- **Full Technical Report**: [`report/report.md`](report/report.md)
- **Engineering Decision Log**: [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md)
- **Final Audit**: [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md)
- **Implementation Status**: [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)
- **Taxonomy Definitions**: [`data/golden/taxonomy.yaml`](data/golden/taxonomy.yaml)
- **Labeling Guidelines**: [`data/golden/labeling_guidelines.md`](data/golden/labeling_guidelines.md)
