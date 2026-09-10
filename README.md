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
[ Intent Classifier (TF-IDF / SentenceTransformer + Calibrated Model) ]
    ├── Predicted Intent
    └── Confidence Score
           │
           ▼
[ Historical Vector Retrieval Index (NearestNeighbors / FAISS) ]
    └── Top-K Historical Cases + Similarity Score
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
- **Retrieves relevant historical resolved cases** via vector similarity without data leakage.
- **Generates grounded responses** adhering to Pydantic JSON schemas.
- **Safely escalates high-risk queries** with explicit reason codes (`LOW_INTENT_CONFIDENCE`, `SENSITIVE_REQUEST`, `NO_RELEVANT_HISTORICAL_EVIDENCE`, `ACCOUNT_SPECIFIC_ACTION_REQUIRED`, `INSUFFICIENT_CONTEXT`).
- **Evaluates performance** via automated split leakage audits, baselines, LLM-as-a-Judge scoring, and Human-vs-LLM agreement metrics.

---

## Dataset & Selected Brand

- **Primary Dataset**: Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`).
- **Selected Brand**: **AmazonHelp** (Selected programmatically based on interaction volume, turn depth, resolution rate, and taxonomy diversity).

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

### Step 1: Ingestion, Graph Reconstruction & Partitioning
```bash
python -m src.data.prepare
```

### Step 2: Golden Candidates & Golden Set Generation
```bash
python -m src.data.label
```

### Step 3: Run Full Test Suite (`pytest`)
```bash
python -m pytest
```

### Step 4: Run Complete Master Evaluation Suite
```bash
python -m evaluation.run_all
```

---

## Empirical Results Summary

| Metric | Measured Value |
|--------|----------------|
| **Data Leakage Status** | `PASSED (ZERO LEAKAGE)` |
| **Intent Classifier Accuracy** | 0.500 |
| **Retrieval Recall@1** | 1.000 |
| **Retrieval Recall@5** | 1.000 |
| **Escalation Auto-Handle Rate** | 37.5% |
| **Escalation False Auto-Handling Rate** | 0.0% |
| **LLM Judge Mean Score** | 10.0 / 10 |
| **Human / LLM Agreement Pearson R** | 0.8922 |
| **Quadratic Weighted Kappa ($\kappa$)** | 0.8912 |

---

## Failure Analysis

1. **Ambiguous Multi-Intent Requests**: Single-label classifiers pick one intent for compound queries containing both shipping delay and product damage.
2. **Low Retrieval Similarity for Novel Edge Cases**: 1-click address change queries lack exact historical matches.
3. **Unnecessary Escalation on Informational Queries**: Keyword rules over-index on sensitive terms like "credit card" even for FAQ queries.
4. **Noisy Text & Typos**: Spelling corruptions degrade TF-IDF features.
5. **Single-Word Tweet Incompleteness**: Short tweets ("@AmazonHelp help") lack context.

---

## Reproduction Path for Reviewer

Run these 3 commands to reproduce all headline metrics and verify zero data leakage:
```bash
python -m src.data.prepare
python -m pytest
python -m evaluation.run_all
```

---

## Documentation & Links

- **Full Technical Report**: [`report/report.md`](file:///c:/Users/gogua/OneDrive/Desktop/AI_Customer_Support_Agent/report/report.md)
- **Engineering Decision Log**: [`docs/DECISION_LOG.md`](file:///c:/Users/gogua/OneDrive/Desktop/AI_Customer_Support_Agent/docs/DECISION_LOG.md)
- **Implementation Status**: [`docs/IMPLEMENTATION_STATUS.md`](file:///c:/Users/gogua/OneDrive/Desktop/AI_Customer_Support_Agent/docs/IMPLEMENTATION_STATUS.md)
- **Taxonomy Definitions**: [`data/golden/taxonomy.yaml`](file:///c:/Users/gogua/OneDrive/Desktop/AI_Customer_Support_Agent/data/golden/taxonomy.yaml)
- **Labeling Guidelines**: [`data/golden/labeling_guidelines.md`](file:///c:/Users/gogua/OneDrive/Desktop/AI_Customer_Support_Agent/data/golden/labeling_guidelines.md)
