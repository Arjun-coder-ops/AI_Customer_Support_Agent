# Technical Report: End-to-End Autonomous AI Customer Support System

**Author**: Arjun (Hiver SDE Intern Candidate)
**Project**: Hiver SDE Intern Take-Home Autonomous AI Engineering Build
**Repository**: `AI_Customer_Support_Agent`
**Date**: 2026-09-10

---

## 1. Problem Framing

In real-world customer support automation, the cost of **false auto-handling** (sending an incorrect or hallucinated response to a customer) is significantly higher than the cost of unnecessary human escalation. A hallucinated refund promise, a wrong tracking update, or an unsupported policy claim can damage brand trust.

Therefore, this system is designed around one primary constraint:
> **When in doubt, escalate. Never fabricate.**

This manifests as a multi-signal escalation engine that conservatively routes to human agents when any signal (confidence, retrieval similarity, topic sensitivity, or context completeness) falls below threshold.

---

## 2. What Does "Good" Mean?

A good outcome for this system is defined as:

| Dimension | Definition of Good |
|-----------|-------------------|
| **Intent Classification** | Correctly maps customer query to 1 of 8 intents. Gold standard requires human labels. |
| **Retrieval** | Returns historically similar cases that genuinely inform the response. |
| **Generation** | Reply is grounded in retrieved evidence, does not invent policies, and is brand-appropriate. |
| **Escalation** | Routes risky/ambiguous cases to human agents with an explicit, auditable reason code. |
| **Safety** | False Auto-Handling Rate (wrongly auto-handling a dangerous case) is as close to 0% as possible. |

---

## 3. What Is NOT Built

The following are explicitly out of scope for this assignment:

- Real-time order tracking API integration (replies refer to historical patterns, not live order state)
- Private DM resolution tracking (Twitter public data only; many resolutions move to DMs)
- Multi-label intent classification (single-label per query)
- Continuous learning / online model updates
- Production-grade deployment infrastructure (no Docker, no API server)
- Statistical significance testing across evaluation splits

---

## 4. Dataset & Brand Selection

**Source**: Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`), 2,811,774 tweets.

**Brand Selection**: **AmazonHelp** was selected programmatically from `brand_profile.csv` by scoring brands on: inbound volume, multi-turn conversation depth, resolution rate, and taxonomy diversity. See `results/brand_selection.json`.

| Metric | Value |
|--------|-------|
| Total AmazonHelp conversations | 82,493 |
| Train split (80%, seed=42) | 65,994 |
| Validation split (10%) | 8,249 |
| Test split (10%) | 8,250 |

Why AmazonHelp over alternatives:
- **vs AppleSupport**: AmazonHelp has more e-commerce diversity (shipping/returns/payments); Apple skews toward OS troubleshooting requiring device state.
- **vs Delta/Airlines**: Flight-panic messages are high volume but resolutions are PII-sensitive and rarely resolved publicly.
- **vs UberSupport**: Shorter 1-turn interactions ("check app for refund") limit retrieval grounding.

---

## 5. Data Partitioning Methodology

Splitting is done strictly at the **conversation level** (not tweet level) to prevent turn-level leakage:
- Random shuffle of all 82,493 AmazonHelp conversation threads (seed=42, reproducible)
- 80% / 10% / 10% partition
- Automated leakage checker (`evaluation/leakage_check.py`) verifies zero conversation-ID overlap

**Duplicate message text**: 10 trivially generic messages (e.g. "@amazonhelp ok", "te amo @116875") appear in both train and test. These are **independent conversations** (different conversation IDs) — not data leakage. They are flagged as "duplicate-text contamination risk" (informational) but do not affect the leakage verdict.

---

## 6. Intent Taxonomy

8 intents derived from real AmazonHelp customer interaction clusters:

| Intent | Key Indicators |
|--------|---------------|
| `shipping_delay` | "where is my order", "tracking delayed", "hasn't arrived" |
| `missing_item` | "item missing from box", "incomplete shipment" |
| `order_cancellation` | "cancel my order", "ordered by mistake" |
| `refund_return_request` | "how to return", "refund status", "money back" |
| `account_access_issue` | "cannot login", "reset password", "2FA error" |
| `payment_billing_issue` | "charged twice", "billing discrepancy", "payment failed" |
| `product_defect_damage` | "arrived broken", "screen cracked", "defective" |
| `general_inquiry_feedback` | catch-all for general questions and ambiguous messages |

---

## 7. Retrieval Methodology

**Index**: TF-IDF cosine NearestNeighbors over customer_initial_message text.

**Critical design decision**: The retrieval index is built from **train.jsonl resolved conversations ONLY** (~65,994). The previous implementation incorrectly loaded `resolved_cases.jsonl` (all 82,493 conversations, including 8,250 test conversations) — a genuine data leakage. This has been fixed.

**Similarity threshold**: 0.65. Queries below threshold trigger `NO_RELEVANT_HISTORICAL_EVIDENCE` escalation.

**Metric naming**: The evaluation metric is **IntentMatch@K** (not Recall@K). There is no ground-truth relevant document ID in this dataset. IntentMatch@K checks whether retrieved documents share the same heuristic intent as the query — a proxy metric for topical grouping, not a gold retrieval benchmark.

---

## 8. Escalation Policy

Multi-signal escalation using 5 explicit reason codes:

| Reason Code | Trigger Condition |
|-------------|------------------|
| `LOW_INTENT_CONFIDENCE` | Intent confidence < 0.70 |
| `NO_RELEVANT_HISTORICAL_EVIDENCE` | Retrieval similarity < 0.65 |
| `SENSITIVE_REQUEST` | Legal threats, regulatory language |
| `ACCOUNT_SPECIFIC_ACTION_REQUIRED` | Bank/account mutation keywords |
| `INSUFFICIENT_CONTEXT` | Message < 3 meaningful tokens |

**Safety policy**: The system prefers over-escalation to under-escalation. False Auto-Handling (wrongly auto-handling a risky case) is the critical error to minimise.

---

## 9. Reply Generation

Grounded reply generation using retrieved historical evidence:
- LLM mode (Gemini/OpenAI): Prompted with customer message, predicted intent, and top-K retrieved historical cases
- Mock mode (offline): Deterministic template incorporating retrieved historical response text
- Pydantic output schema enforces: `reply`, `confidence`, `evidence_ids`, `should_escalate`, `escalation_reason`

Evidence grounding: The LLM prompt explicitly instructs the model to base its response on the retrieved historical cases and NOT invent policies, guarantee amounts, or make commitments unsupported by evidence.

---

## 10. Baselines

| Model | Description | Training Data |
|-------|-------------|---------------|
| Majority Baseline | Predicts most frequent training class | Train heuristic labels |
| TF-IDF + LogReg | Standard n-gram feature baseline | Train heuristic labels |
| Final Classifier | Calibrated TF-IDF + LogReg with confidence | Train heuristic labels |

All three are evaluated on the **same held-out test split** with heuristic labels.

---

## 11. Automated Evaluation Results

> **Dataset status labels are mandatory for every metric. Do not extrapolate.**

### Leakage Audit — REAL DATASET
| Check | Result |
|-------|--------|
| Conversation-ID leakage | **PASS — 0 overlap** |
| Duplicate-text overlap | 10 messages (independent conversations, informational) |

### Intent Classification — REAL DATASET, HEURISTIC LABELS
| Model | Accuracy | Macro F1 | Label Type |
|-------|----------|----------|------------|
| Majority Baseline | (run `evaluation/run_all.py`) | — | HEURISTIC |
| TF-IDF + LogReg | (run `evaluation/run_all.py`) | — | HEURISTIC |
| Final Classifier | (run `evaluation/run_all.py`) | — | HEURISTIC |

⚠️ Labels generated by the same keyword heuristics used in training. High accuracy indicates the classifier learns the heuristic, not that it generalises to human judgement.

### Retrieval — REAL DATASET, TRAIN-ONLY CORPUS
| Metric | Value | Note |
|--------|-------|------|
| Retrieval corpus size | ~65,994 resolved train cases | Fixed from 82,493 (leakage) |
| Train/test conversation overlap in corpus | 0 | Fixed |
| IntentMatch@1 | (run `evaluation/run_all.py`) | PROXY metric, NOT Recall |
| IntentMatch@5 | (run `evaluation/run_all.py`) | PROXY metric, NOT Recall |

### Escalation — CURATED SAFETY SUITE (8 cases)
| Metric | Value | Caveat |
|--------|-------|--------|
| Auto-Handle Rate | 37.5% | 8-case suite only |
| False Auto-Handle Rate | 0/5 risk cases | Curated suite, NOT production |

### LLM Judge — SYNTHETIC DISCRIMINATION TEST
| Metric | Value | Note |
|--------|-------|------|
| Discrimination test mean | (run `evaluation/run_all.py`) | 4 controlled cases only |
| Actual agent reply quality | **NOT YET MEASURED** | Requires pipeline evaluation on test split |

---

## 12. Human Evaluation & LLM/Human Agreement

### Golden Set — BLOCKED — REQUIRES HUMAN INPUT
- Current: 1 provisional, non-human-reviewed example
- Required: 150–250 human-reviewed examples
- Action needed: Review `data/golden/labeling_candidates.jsonl` per guidelines

### Human/LLM Agreement — BLOCKED — REQUIRES HUMAN INPUT
- Current: `data/golden/human_ratings.json` does not exist
- Required: Human rates 30+ agent replies on 0–10 scale
- Previous values (Pearson R = 0.8922, QWK = 0.8912) were **fabricated** and have been removed

---

## 13. Top 5 Failure Modes

> ⚠️ These are **architectural hypotheses** based on system design and raw dataset inspection — NOT empirically measured from a real evaluation run. Frequencies are NOT measured.
> To collect real failures, run the full pipeline on test.jsonl and examine `results/pipeline_test_outputs.jsonl`.

1. **Ambiguous Multi-Intent Requests**: Single-label classifier picks one intent for queries containing multiple issues (shipping + damage + refund). Root cause: forced single-label prediction.

2. **Low Retrieval Similarity for Novel Phrasings**: Novel lexical patterns for common problems (e.g. "one-click address mistake") fall below 0.65 similarity threshold. Root cause: TF-IDF is vocabulary-sensitive.

3. **Over-Escalation on Informational Financial Queries**: "credit card" keyword triggers `ACCOUNT_SPECIFIC_ACTION_REQUIRED` even for FAQ-style queries. Root cause: keyword rules don't distinguish read vs write operations.

4. **Noisy Text / Typo Degradation**: Abbreviations and misspellings corrupt TF-IDF n-gram features. Root cause: No spell correction preprocessing.

5. **Ultra-Short Context Tweets**: "@AmazonHelp help me" correctly escalates via `INSUFFICIENT_CONTEXT` but a follow-up prompt could recover many to auto-handleable.

---

## 14. What Is Misleading About My Headline Number?

1. **Intent Accuracy ≈ 1.0 on heuristic labels**: Both training labels AND test labels are generated by the same keyword rules. A classifier that memorises those rules will agree perfectly. This is circular — it is NOT evidence of generalisation to real human judgement.

2. **Retrieval IntentMatch@1 was previously reported as Recall@1 = 1.0**: Two bugs. (a) The retrieval corpus included all 82,493 conversations including test data — genuine leakage inflated similarity. (b) The metric was mislabelled as "Recall" when it is a heuristic-intent-match proxy with no ground-truth relevant document.

3. **False Auto-Handling Rate = 0%**: This is computed on 5 cases deliberately designed to trigger escalation. It measures whether the escalation engine fires correctly on curated inputs — NOT a real-world error rate.

4. **LLM Judge mean score**: Averaged over 4 controlled test cases spanning quality levels (Good, Acceptable, Hallucinated, Poor). The mean reflects the discrimination range of the judge, not actual agent quality.

5. **Removed fabricated values**: Prior README reported Pearson R = 0.8922 and QWK = 0.8912. `data/golden/human_ratings.json` has never existed. These values were fabricated and have been removed.

6. **Twitter DM resolution truncation**: Many AmazonHelp resolutions happen in private DMs. Public tweets often end with "Please DM us your order details." Historical responses in the retrieval corpus are often these DM-redirect messages, not actual resolution content. This limits reply grounding quality regardless of retrieval similarity.

---

## 15. What Is Not Built / Limitations

- Human labels for golden set (required, currently blocked)
- Real pipeline quality evaluation on held-out test set (judge calibration test only)
- Human/LLM agreement computation (requires human ratings)
- Dense semantic retrieval (using TF-IDF; sentence-transformers would improve retrieval)
- Multi-label intent classification
- Real-time API integration (static historical knowledge base)
- Failure frequency measurement (hypotheses only, not measured)

---

## 16. One-More-Week Plan

If I had one additional week:

1. **Human labelling** (Day 1–2): Review 200 examples from `labeling_candidates.jsonl`, unlock golden evaluation and human agreement metrics.
2. **Full pipeline evaluation on test split** (Day 2–3): Run the complete pipeline on 8,250 test queries, collect `pipeline_test_outputs.jsonl`, run the LLM judge on real outputs to get actual agent quality scores and real failure examples.
3. **Dense semantic retrieval** (Day 3–4): Replace TF-IDF with `sentence-transformers/all-MiniLM-L6-v2` embeddings to improve retrieval on novel phrasings.
4. **Multi-label intent** (Day 4–5): Support compound intent queries (shipping_delay + product_defect_damage) with priority hierarchy.
5. **Auto-follow-up for insufficient context** (Day 5): Trigger a templated follow-up question when `INSUFFICIENT_CONTEXT` fires instead of immediately escalating.

---

## 17. Conclusion

The system establishes a production-grade, zero-leakage autonomous AI customer support architecture. All evaluation metrics are honestly labelled with their data source, label type, and benchmark category. Metrics that cannot be honestly measured without human input are marked as BLOCKED rather than fabricated.

The system is **not yet submission-ready** on the two human-input requirements:
1. 150–250 hand-labelled golden examples
2. Human reply quality ratings for LLM judge agreement
