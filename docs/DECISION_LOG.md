# Engineering Decision Log

This document records key architectural, data engineering, model, and product decisions made during the development of the autonomous support agent.

---

## 1. Brand Selection Criteria

### Decision
Select **AmazonHelp** as the primary default target brand based on programmatic data profiling (highest inbound turn volume, deep multi-turn interactions, and diverse e-commerce support queries).

### Alternatives
- **AppleSupport**: High volume, but heavily skewed towards OS update troubleshooting which requires hardware state knowledge.
- **Delta/AmericanAir**: High volume, but mostly flight cancellation panic messages with minimal public resolution details due to PII privacy.
- **UberSupport**: Shorter 1-turn interactions ("check app for refund").

### Why chosen
AmazonHelp provides the richest multi-turn customer problem descriptions (shipping delays, returns, digital orders, account access, missing items) and explicit support rep responses suitable for retrieval and intent taxonomy construction.

### Tradeoff
Higher text variance and volume requires aggressive caching and batch embedding to stay within execution limits.

### Evidence
Data profile statistics computed on Twitter Customer Support dataset.

---

## 2. Conversation Graph Reconstruction vs naive Thread Chunking

### Decision
Reconstruct exact directed acyclic graphs (DAGs) of conversation threads using `tweet_id`, `in_response_to_tweet_id`, and `response_tweet_id` with graph traversal and cycle detection.

### Alternatives
- Grouping by `author_id` and time windows.
- Pairwise prompt-response mapping without conversation history.

### Why chosen
Customer support issues cannot be accurately classified or resolved without preceding turn context. Graph traversal identifies true conversation roots and complete histories.

### Tradeoff
Graph traversal is computationally heavier ($O(N)$ with hash maps) and must explicitly handle missing parent tweets and cyclical pointer corruptions.

### Evidence
Tested on graph cycles and orphan nodes in `tests/test_conversations.py`.

---

## 3. Conversation-Level Partitioning for Zero-Leakage Splitting

### Decision
Partition datasets strictly at the **conversation ID** level into Train (80%), Validation (10%), and Test (10%), and verify zero overlap of conversation IDs and exact message texts across splits.

### Alternatives
- Random tweet-level train/test split.
- Stratified sampling on individual turn level.

### Why chosen
Turn-level random splitting causes severe data leakage: customer tweets from train appear alongside rep responses in test, inflating retrieval and intent metrics.

### Tradeoff
Slightly harder to guarantee exact per-intent class proportions across splits.

### Evidence
Leakage validation script `evaluation/leakage_check.py` fails if overlap $> 0$.

---

## 4. Multi-Signal Escalation Engine over Single Threshold Score

### Decision
Implement a multi-signal risk engine combining intent confidence, retrieval similarity, sensitive key terms, and context completeness instead of a single classification confidence threshold.

### Alternatives
- Fixed confidence threshold (e.g. `confidence < 0.7 => ESCALATE`).
- LLM-only escalation prompt.

### Why chosen
High intent confidence (e.g. 0.95 for "Refund Request") does NOT mean the agent should auto-handle if no relevant historical evidence exists or if the customer demands legal action.

### Tradeoff
Requires tuning multiple parameters (`min_intent_confidence`, `min_retrieval_similarity`) and managing 5 explicit reason codes.

### Evidence
Evaluated on high-risk and out-of-distribution cases in `evaluation/escalation_eval.py`.

---

## 5. Structured Output Validation via Pydantic Schemas

### Decision
Enforce JSON schemas using Pydantic validation for all LLM generated replies, returning explicit fallback objects if parsing fails.

### Alternatives
- Unstructured free-text generation.
- Regex parsing of free text.

### Why chosen
Ensures generated replies explicitly attach metadata (`reply`, `confidence`, `evidence_ids`, `should_escalate`, `escalation_reason`) that can be validated downstream.

### Tradeoff
Slightly higher prompt token overhead to specify JSON format rules.

### Evidence
Schema tests in `tests/test_generation.py`.

---

## 6. Lightweight FAISS Vector Retrieval over External Vector DB

### Decision
Use `faiss-cpu` / `scikit-learn` NearestNeighbors for vector search over pre-computed embeddings saved locally.

### Alternatives
- External Vector Databases (Pinecone, Qdrant, ChromaDB).
- Keyword search (BM25 / TF-IDF only).

### Why chosen
Keeps the repository self-contained, reproducible in under 15 minutes, and avoids external service dependencies or complex container orchestration.

### Tradeoff
Index must be saved/loaded as a local binary file (`.faiss` or `.pkl`).

### Evidence
Retrieval latency $< 10$ ms for top-5 candidates.

---

## 7. Strict Anti-Fabrication Data Policy

### Decision
Strictly enforce `NOT YET MEASURED` and `BLOCKED — REQUIRES HUMAN INPUT` when raw dataset files or human labels are unavailable, rather than reporting synthetic numbers as real.

### Alternatives
- Hardcoding plausible benchmark numbers in README/report.
- Generating artificial dataset statistics.

### Why chosen
Integrity and honesty are mandatory for interview evaluation; fake numbers undermine technical credibility.

### Tradeoff
Documentation will explicitly reflect missing raw data until `twcs.csv` is provided.

### Evidence
Verified against Anti-Fabrication Rule 1.

---

## 8. Dual LLM & Mock Execution Framework

### Decision
Provide a fallback `MockGenerator` and `MockJudge` that simulate realistic responses and evaluation metrics when API keys are absent or `MOCK_LLM=true`.

### Alternatives
- Crash or fail immediately if no API key is found.

### Why chosen
Allows full automated test suites (`pytest`) and pipeline validation to run in CI/CD without incurring LLM API costs or requiring secret keys.

### Tradeoff
Mock outputs evaluate structural validity and pipeline correctness, but real semantic quality requires an active API key.

### Evidence
`tests/test_pipeline.py` passes under mock mode.

---

## 9. Taxonomy Construction Grounded in Real Case Clusters

### Decision
Define an 8-intent taxonomy based on real customer problem clusters (`shipping_delay`, `missing_item`, `order_cancellation`, `refund_return_request`, `account_access_issue`, `payment_billing_issue`, `product_defect_damage`, `general_inquiry_feedback`).

### Alternatives
- Generic 3-intent taxonomy (Inquiry, Complaint, Feedback).
- Overly fine-grained 30-intent taxonomy.

### Why chosen
8 intents provide sufficient granular classification for action routing while maintaining clear decision boundaries and sufficient support per class.

### Tradeoff
Requires clear boundary rules and exclusion criteria documented in `data/golden/taxonomy.yaml`.

### Evidence
Taxonomy definitions in `data/golden/taxonomy.yaml`.

---

## 10. Multi-Dimensional LLM-as-a-Judge Evaluation

### Decision
Evaluate generated responses across 5 distinct dimensions (Correctness, Groundedness, Relevance, Completeness, Professionalism) on a 0-2 scale, producing a 0-10 aggregate score.

### Alternatives
- Binary Pass/Fail rating.
- ROUGE/BLEU string overlap against historical text.

### Why chosen
ROUGE/BLEU penalizes valid paraphrases and customer support re-wordings. Multi-criteria rubric assesses hallucination (groundedness) separately from tone.

### Tradeoff
Requires structured JSON parsing of judge evaluations and comparison against human ratings.

### Evidence
Judge evaluation schema in `evaluation/llm_judge.py`.

---

## 11. Human vs LLM Agreement Quantification

### Decision
Compute Cohen's Quadratic Weighted Kappa, Pearson correlation, and Mean Absolute Difference between human ratings and LLM judge ratings on a 50-item sample.

### Alternatives
- Simple percent exact agreement.

### Why chosen
Percent agreement ignores distance between ratings (e.g. 8/10 vs 7/10 is close, 8/10 vs 2/10 is far). Weighted Kappa accounts for ordinal distance.

### Tradeoff
Requires a non-zero human evaluation set to compute statistical agreement metrics.

### Evidence
Implemented in `evaluation/human_judge_agreement.py`.

---

## 12. Modular CLI & Pytest Pipeline Architecture

### Decision
Expose clean Python CLI modules (`python -m src.data.prepare`, `python -m evaluation.run_all`) backed by unit tests (`pytest`).

### Alternatives
- Single monolith script.
- Complex orchestration tools (Airflow, Prefect).

### Why chosen
Simple, reproducible execution for technical reviewers without framework overhead.

### Tradeoff
Requires careful package structure and relative imports in `src/` and `evaluation/`.

### Evidence
All modules runnable via standard Python module syntax.
