# Hiver Take-Home Final Audit Report

**Audit Timestamp**: 2026-09-19
**Evaluator**: Technical Audit (post-real-dataset integration)
**Repository Path**: `C:\Users\gogua\OneDrive\Desktop\AI_Customer_Support_Agent`

---

## 1. Repository Status

All code, pipelines, tests, and evaluation runners are **fully built and operational**.
The repository uses the **REAL dataset** (`data/raw/twcs.csv`, 2,811,774 tweets).

---

## 2. Dataset Status

**REAL DATASET** — `twcs.csv` is present and ingested.

| Metric | Value |
|--------|-------|
| Total tweets loaded | 2,811,774 |
| Conversation threads reconstructed | 798,197 |
| Selected brand | AmazonHelp |
| AmazonHelp total conversations | 82,493 |
| Train split (80%, seed=42) | 65,994 |
| Validation split (10%) | 8,249 |
| Test split (10%) | 8,250 |

Data pipeline: `python -m src.data.prepare`
Evidence artifact: `data/processed/dataset_profile.json`

---

## 3. Brand Selection

**Selected Brand**: `AmazonHelp`
**Selection basis**: Programmatically identified via `brand_profile.csv` scoring on: inbound volume, multi-turn depth, resolution rate, and taxonomy coverage. Alternatives (Apple, Delta, Uber) ranked lower due to PII sensitivity, OS-specific bias, or shallow turn depth.
**Evidence**: `results/brand_selection.json`

---

## 4. Golden Set Status

`BLOCKED — REQUIRES HUMAN INPUT`

| Item | Status |
|------|--------|
| Candidate count | 200 extracted into `data/golden/labeling_candidates.jsonl` (from test/val splits only) |
| Human-reviewed count | **0** (was 0, previously 1 provisional only) |
| Required minimum | 150 human-reviewed examples |
| Review status | `is_human_reviewed: false` — all provisional |
| Labeling guidelines | `data/golden/labeling_guidelines.md` (comprehensive workflow provided) |

**Action required**: Human must annotate 150–250 examples following `data/golden/labeling_guidelines.md`.

---

## 5. Data Leakage Audit

`PASS (ZERO CONVERSATION-LEVEL LEAKAGE)`

| Check | Result |
|-------|--------|
| `train_val_id_overlap` | 0 |
| `train_test_id_overlap` | 0 |
| `train_golden_id_overlap` | 0 |
| `val_test_id_overlap` | 0 |

**Duplicate-text contamination**: 10 trivially generic messages (e.g. "@amazonhelp ok", "te amo @116875") appear in both train and test. These are **independent conversations** (different conversation IDs). Assessed as `INDEPENDENT_CONVERSATIONS` — not conversation leakage. Flagged as informational.

**ISSUE 3 FIX**: The previous checker reported `has_leakage=True` on these 10 messages, incorrectly failing the leakage check. The updated checker distinguishes conversation-ID leakage (critical) from duplicate-text overlap (informational). `has_leakage` now correctly reflects only conversation-ID overlap.

Evidence artifact: `evaluation/leakage_check.py`, `results/final_results.json`

---

## 6. Retrieval Data Leakage — FIXED

`PREVIOUSLY FAILED — NOW FIXED`

| Item | Before Fix | After Fix |
|------|-----------|-----------|
| Retrieval index source | `resolved_cases.jsonl` (all 82,493 conversations) | `train.jsonl` (65,994 conversations only) |
| Test conversations in retrieval index | **8,250 (LEAKAGE)** | **0 (FIXED)** |
| Reported metric | `recall_at_1/3/5` (mislabelled) | `intent_match_at_1/3/5` (correct) |

**ISSUE 1 FIX**: `evaluation/retrieval_eval.py` now loads from `train.jsonl` only, not `resolved_cases.jsonl`. Train/test conversation overlap in retrieval corpus is verified to be 0.

**ISSUE 2 FIX**: Metrics renamed from `recall_at_k` to `intent_match_at_k`. There is no ground-truth relevant document ID in this dataset. IntentMatch@K is a heuristic proxy metric, not standard retrieval recall.

---

## 7. Intent Classification

**ISSUE 4 FIX**: The previous implementation fell back to `X_test = X_train` when `golden_set.jsonl` was empty (1 provisional example). This caused accuracy = 1.0 by evaluating the model on its own training data.

The fixed implementation evaluates on the **held-out test split** (`test.jsonl`) with heuristic labels.

**Label source**: HEURISTIC keyword rules — both training labels AND test labels use the same keyword mapper. This creates a circular evaluation: accuracy measures "does the classifier agree with the rules?" not "does the classifier match human judgement?"

| Model | Dataset | Label Type | Benchmark |
|-------|---------|-----------|-----------|
| Majority Baseline | REAL (test split, ~8,250 queries) | HEURISTIC | PROXY |
| TF-IDF + LogReg | REAL (test split, ~8,250 queries) | HEURISTIC | PROXY |
| Final Classifier | REAL (test split, ~8,250 queries) | HEURISTIC | PROXY |

Gold benchmark: **BLOCKED — REQUIRES HUMAN INPUT** (0 human-reviewed examples vs 150 required).

---

## 8. Generation

**Status**: VERIFIED & OPERATIONAL

**Grounding**: Evidence from retrieved historical cases is passed to the LLM prompt. The system prompt instructs the model not to invent policies or guarantee amounts unsupported by evidence. Mock mode uses first retrieved historical response as grounding text.

**Output schema** (Pydantic-enforced): `reply`, `confidence`, `evidence_ids`, `should_escalate`, `escalation_reason`.

**Gap**: `evidence_used` / `grounding_status` tracking fields are not yet implemented (would require pipeline-level output logging for failure analysis).

---

## 9. LLM Judge Evaluation

**ISSUE 7 FIX**: The evaluation is now explicitly labelled as a `JUDGE_DISCRIMINATION_TEST` — NOT a measure of real agent reply quality.

| Field | Value |
|-------|-------|
| `evaluation_type` | `JUDGE_DISCRIMINATION_TEST` |
| `dataset_status` | `SYNTHETIC/CONTROLLED TEST — 4 hand-crafted cases` |
| `actual_agent_quality` | `NOT YET MEASURED` |

The 4 controlled test cases (Good, Acceptable, Hallucinated, Poor) verify the judge discriminates correctly between quality levels. The mean score from these 4 cases **must not** be reported as agent output quality.

Previous error: Old reports stated "LLM Judge Mean Score = 10.0/10" — this was the GOOD_REPLY case score, incorrectly averaged with quality extremes.

---

## 10. Human / LLM Agreement

`BLOCKED — REQUIRES HUMAN INPUT`

- `data/golden/human_ratings.json` does not exist.
- Agreement metrics (Pearson R, QWK) are `NOT YET MEASURED`.
- **Removed fabricated values**: Previous README contained `Pearson R = 0.8922` and `QWK = 0.8912` — these were never computed from real data and have been removed from all documentation.

---

## 11. Escalation Policy

**ISSUE 8 FIX**: Evaluation now explicitly labelled as `CURATED SAFETY SUITE`.

| Metric | Value | Context |
|--------|-------|---------|
| Total test cases | 8 | Curated hand-crafted, NOT real traffic |
| Auto-Handle Rate | 37.5% | 8-case suite only |
| False Auto-Handle Rate | 0/5 risk cases | Curated suite, NOT production |
| Reason codes verified | 5 of 5 | All escalation triggers fire correctly |

The 0% false auto-handling rate applies only to the 5 deliberately designed risky cases. It cannot be extrapolated to real-world traffic without a large annotated escalation benchmark.

---

## 12. Failure Analysis

**ISSUE 9 FIX**: The previous failure analysis contained fabricated percentage frequencies (e.g. "14.2% of evaluated error cases") and invented pipeline outputs.

**Current status**: `BLOCKED — REQUIRES REAL EVALUATION OUTPUT`

5 failure mode **hypotheses** are documented based on architectural analysis and raw dataset inspection — clearly labelled as `source: ILLUSTRATIVE_HYPOTHESIS`. Frequencies are `NOT MEASURED`.

To collect real failures: run the full pipeline on `test.jsonl`, save outputs to `results/pipeline_test_outputs.jsonl`, then re-run failure analysis.

---

## 13. Tests

**37 tests passing** (up from 14).

Added test coverage for:
- ISSUE 1: Train-only retrieval corpus (zero test-conv overlap)
- ISSUE 2: `intent_match_at_k` keys present, `recall_at_k` keys absent
- ISSUE 3: Independent-conversation duplicate text does NOT trigger `has_leakage=True`
- ISSUE 4: Evaluation uses test split, not training data
- ISSUE 5: Golden status BLOCKED when < 150 human-reviewed examples
- ISSUE 6: Human agreement BLOCKED when ratings file missing; no fabricated values
- ISSUE 7: Judge labelled as discrimination test; `actual_agent_quality = NOT YET MEASURED`

---

## 14. Reproducibility

### Automated (< 15 minutes):
```bash
python -m src.data.prepare        # ingestion, split, profile
python -m src.data.label           # golden candidates from test/val
python -m pytest                   # 37 tests
python -m evaluation.run_all       # full evaluation suite
```

### Requires Human Input (cannot be automated):
- Annotate 150–250 golden examples from `data/golden/labeling_candidates.jsonl`
- Provide 30+ human reply ratings in `data/golden/human_ratings.json`

---

## 15. Files Changed in This Audit

| File | Change |
|------|--------|
| `evaluation/retrieval_eval.py` | Load train.jsonl only; rename recall→intent_match; add leakage check |
| `evaluation/leakage_check.py` | Distinguish conv-ID leakage from duplicate-text; fix has_leakage logic |
| `evaluation/intent_eval.py` | Evaluate on test split not training data; label type field; golden guard |
| `evaluation/escalation_eval.py` | Label as CURATED SAFETY SUITE with caveat |
| `evaluation/llm_judge.py` | Label as JUDGE_DISCRIMINATION_TEST; add actual_agent_quality field |
| `evaluation/failure_analysis.py` | Remove fabricated percentages; mark as hypotheses; add real-failure collector |
| `evaluation/run_all.py` | Updated field names; honest console output with metric labels |
| `evaluation/human_judge_agreement.py` | Unchanged — already correctly blocks without human ratings |
| `README.md` | Remove Pearson/QWK fabrications; add metric labels; "What is misleading" section |
| `report/report.md` | Full rewrite with all 17 required sections; honest metrics |
| `docs/FINAL_AUDIT.md` | This document (full rewrite) |
| `data/golden/labeling_guidelines.md` | Comprehensive annotation workflow |
| `src/data/golden_validator.py` | New: validate_golden_set() blocks evaluation until 150+ examples |
| `tests/test_leakage.py` | Comprehensive tests covering Issue 3 fix |
| `tests/test_retrieval.py` | Tests covering Issues 1 & 2 |
| `tests/test_intent.py` | Tests covering Issue 4 fix |
| `tests/test_judge_and_agreement.py` | New: tests for Issues 6 & 7 |

---

## 16. Submission Readiness Verdict

`NOT READY TO SUBMIT` — 2 mandatory human-input items remain:

| Required Item | Status | Blocker |
|--------------|--------|---------|
| 150–250 hand-labelled golden examples | **0 reviewed** | YOU must annotate `labeling_candidates.jsonl` |
| Human reply quality ratings | **Missing** | YOU must provide `human_ratings.json` |

All code, tests, pipelines, and automated evaluations are complete and passing. The system will be ready to submit once the two human annotation tasks are completed.

---

## 17. Trustworthy Metrics Checklist

| Metric | Trustworthy? | Why |
|--------|-------------|-----|
| Conversation-level leakage = PASS | ✅ Yes | Zero conversation-ID overlap verified |
| Retrieval corpus = train-only | ✅ Yes | Fixed from leakage; overlap verified = 0 |
| IntentMatch@K | ⚠️ Proxy only | No ground-truth relevant document; heuristic label |
| Intent Accuracy/F1 | ⚠️ Proxy only | Circular: same heuristic rules for train and test labels |
| Escalation False Auto-Handle | ⚠️ Suite only | 8 curated cases, not real traffic |
| Judge discrimination test | ✅ Valid | 4 controlled cases; judge correctly ranks quality |
| Actual agent quality | ❌ Not measured | Requires pipeline evaluation on real test outputs |
| Human/LLM agreement | ❌ Blocked | No human ratings file |
| Golden benchmark | ❌ Blocked | 0 human-reviewed examples |
| Failure mode frequencies | ❌ Not measured | Architectural hypotheses only |
