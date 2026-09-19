# Engineering Decision Log

Decisions actually represented by the current implementation/process. 15 items.

---

## 1. Why AmazonHelp

**Decision:** Focus on AmazonHelp.  
**Why:** Highest usable multi-turn volume and e-commerce intent diversity in the Twitter support corpus.  
**Evidence:** `results/brand_selection.json`, `results/dataset_profile.json`.

---

## 2. Conversation-graph reconstruction

**Decision:** Rebuild threads via tweet reply pointers with cycle handling.  
**Why:** Intent and resolution context require full conversation roots, not isolated tweets.  
**Evidence:** `src/data/conversations.py`, `tests/test_conversations.py`.

---

## 3. Conversation-level train/val/test split

**Decision:** 80/10/10 split by conversation ID (seed 42).  
**Why:** Tweet-level splits leak customer text across train and test and inflate retrieval.  
**Evidence:** `evaluation/leakage_check.py` → 0 conversation-ID overlap.

---

## 4. Heuristic taxonomy bootstrap

**Decision:** Bootstrap 8 intents + keyword provisional labels for training.  
**Why:** No human labels existed at build time; taxonomy is still data-derived and documented.  
**Tradeoff:** Heuristic eval accuracy is circular until human golden labels exist.  
**Evidence:** `data/golden/taxonomy.yaml`, `evaluation/intent_eval.py`.

---

## 5. Train-only retrieval corpus

**Decision:** Index only `train.jsonl` resolved cases.  
**Why:** Prevents test responses from becoming retrieval evidence (leakage).  
**Evidence:** `evaluation/retrieval_eval.py`, `evaluation/pipeline_eval.py` leakage asserts.

---

## 6. TF-IDF retrieval (not dense embeddings by default)

**Decision:** TF-IDF + cosine nearest neighbors for the submission path.  
**Why:** Fast, dependency-light, reproducible offline; dense models remain a one-week upgrade.  
**Evidence:** `src/retrieval/index.py`.

---

## 7. Confidence threshold 0.70

**Decision:** Escalate when intent confidence < 0.70.  
**Why:** Prefer false escalations over confident wrong auto-handles.  
**Not tuned** to maximize auto-handle rate.  
**Evidence:** `configs/config.yaml`, `src/escalation/policy.py`.

---

## 8. Retrieval similarity threshold 0.65

**Decision:** Escalate when top evidence similarity < 0.65 (`NO_RELEVANT_HISTORICAL_EVIDENCE`).  
**Why:** Weak historical grounding should not produce assertive auto-replies.  
**Evidence:** escalation policy + curated suite eval.

---

## 9. Escalation-first multi-signal policy

**Decision:** Ordered checks: context → sensitive → account action → confidence → retrieval → auto-handle.  
**Why:** High intent confidence alone is insufficient for legal/account/low-evidence cases.  
**Evidence:** `src/escalation/policy.py`, `tests/test_escalation.py`.

---

## 10. Gemini structured generation + MOCK_LLM

**Decision:** Real LLM = Gemini only; offline path = deterministic mock.  
**Why:** Assignment needs real generation, but CI/reviewers must run without secrets. OpenAI removed as unnecessary.  
**Evidence:** `src/generation/generator.py`, `.env.example`.

---

## 11. Human evaluation design (blocked until real labels)

**Decision:** Candidates + CLI labeling; refuse to treat heuristic seeds as human; agreement requires `human_ratings.json`.  
**Why:** Anti-fabrication — do not invent golden labels or agreement stats.  
**Evidence:** `src/data/label.py`, `evaluation/human_judge_agreement.py`.

---

## 12. Explicit leakage controls in eval

**Decision:** Automated leakage check + pipeline_eval pre/post index overlap asserts.  
**Why:** Prior bug indexed all conversations including test; must not recur.  
**Evidence:** `evaluation/leakage_check.py`, `evaluation/pipeline_eval.py`.

---

## 13. No production infrastructure

**Decision:** No Docker/UI/queues/cloud deploy in this submission.  
**Why:** Scope is a reproducible, interview-modifiable research prototype.  
**Evidence:** repository layout / README scope statement.

---

## 14. IntentMatch@K naming (not Recall@K)

**Decision:** Report IntentMatch@K as a proxy topical metric.
**Why:** Dataset has no gold relevant-document IDs; calling it Recall would overclaim.
**Evidence:** `evaluation/retrieval_eval.py`, README metric notes.

---

## 15. Distinguishing conversation-level leakage from duplicate-text contamination

**Decision:** The leakage checker reports two separate verdicts: (a) `conversation_id_leakage` (critical — grounds the `has_leakage` flag), and (b) `duplicate_text_contamination_risk` (informational — when identical message strings appear across independent conversations).

**Why this is non-obvious:** A naive leakage check that unions all duplicate evidence — ID overlap OR text overlap — would incorrectly fail on trivially generic tweets like "@amazonhelp ok" or "te amo". These strings occur in multiple independent conversations by chance (different conversation_ids, different timestamps, different resolutions). Treating them as leakage would produce a false FAIL verdict, obscure genuinely important ID-level leakage, and potentially lead to incorrectly discarding those conversations from evaluation.

**Alternatives considered:**
- Fail on ANY text overlap → false positive on generic messages; not useful.
- Ignore text overlap entirely → misses real contamination if short generic messages have disproportionate class representation in training.

**Why chosen:** Keep the critical conversation-ID check as the primary verdict. Separately surface duplicate-text overlap as an informational diagnostic, including a recommendation to filter messages shorter than 5 tokens for intent evaluation.

**Tradeoff:** Requires human to interpret the contamination risk note; it is not automatically resolved.

**Evidence:** `evaluation/leakage_check.py`, `tests/test_leakage.py` (test_leakage_check_duplicate_text_independent_conversations).

