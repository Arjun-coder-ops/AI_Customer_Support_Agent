import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAILURE ANALYSIS
# ---------------------------------------------------------------------------
# Uses REAL pipeline outputs from results/pipeline_test_outputs.jsonl.
#
# Important distinctions:
# - Not every escalation is a model failure.
# - Expected safety escalations are labelled as such.
# - Intent mismatches are vs HEURISTIC labels, not human gold.
# - Frequencies are measured from the pipeline output file only.
# - No fabricated percentages or invented examples.
# ---------------------------------------------------------------------------

# Priority order for categorization (first match wins).
CATEGORY_PRIORITY = [
    "INTENT_MISMATCH_VS_HEURISTIC",
    "LOW_INTENT_CONFIDENCE_ESCALATION",
    "INSUFFICIENT_CONTEXT_ESCALATION",
    "SENSITIVE_REQUEST_ESCALATION",
    "ACCOUNT_SPECIFIC_ACTION_ESCALATION",
    "LOW_RETRIEVAL_SIMILARITY",
]

CATEGORY_KIND = {
    "INTENT_MISMATCH_VS_HEURISTIC": "diagnostic_disagreement",
    "LOW_INTENT_CONFIDENCE_ESCALATION": "expected_safety_escalation",
    "INSUFFICIENT_CONTEXT_ESCALATION": "expected_safety_escalation",
    "SENSITIVE_REQUEST_ESCALATION": "expected_safety_escalation",
    "ACCOUNT_SPECIFIC_ACTION_ESCALATION": "expected_safety_escalation",
    "LOW_RETRIEVAL_SIMILARITY": "architectural_limitation",
}

CATEGORY_NOTES = {
    "INTENT_MISMATCH_VS_HEURISTIC": (
        "Classifier prediction disagrees with the heuristic keyword label. "
        "This is a diagnostic signal, NOT proof of human-labelled error."
    ),
    "LOW_INTENT_CONFIDENCE_ESCALATION": (
        "Policy correctly escalates when intent confidence is below the "
        "configured threshold. This is expected safety behaviour, not a "
        "pipeline bug."
    ),
    "INSUFFICIENT_CONTEXT_ESCALATION": (
        "Policy correctly escalates ultra-short / low-information messages. "
        "Expected safety behaviour; architectural limitation is lack of a "
        "clarification turn."
    ),
    "SENSITIVE_REQUEST_ESCALATION": (
        "Policy escalates legal/fraud/high-risk keyword matches. Expected "
        "safety behaviour."
    ),
    "ACCOUNT_SPECIFIC_ACTION_ESCALATION": (
        "Policy escalates account-mutation keyword matches. Expected safety "
        "behaviour; may over-trigger on informational questions."
    ),
    "LOW_RETRIEVAL_SIMILARITY": (
        "Top historical evidence similarity is below the escalation threshold "
        "(or reason_code=NO_RELEVANT_HISTORICAL_EVIDENCE). Architectural "
        "limitation of TF-IDF lexical retrieval — not proof the retrieved "
        "case is irrelevant to a human."
    ),
}


FIVE_FAILURE_MODE_HYPOTHESES = [
    {
        "failure_id": "FAIL_001",
        "failure_name": "Ambiguous Multi-Intent Customer Request",
        "source": "ILLUSTRATIVE_HYPOTHESIS — not measured from real evaluation run",
        "measurement_status": "hypothesis",
        "frequency": "NOT MEASURED — requires human-labelled multi-intent analysis",
        "illustrative_example": {
            "customer_message": (
                "My order arrived late and when I opened it the screen was "
                "cracked, I want a refund!"
            ),
            "predicted_intent": (
                "shipping_delay (illustrative — may differ in real run)"
            ),
            "confidence": "ILLUSTRATIVE — not measured",
            "note": (
                "This example is illustrative of the failure mode, not a "
                "measured pipeline output."
            ),
        },
        "architectural_root_cause": (
            "The single-label text classifier must choose one intent for "
            "queries containing multiple issues such as shipping delay, "
            "product damage, and refund requests."
        ),
        "hypothesis": (
            "Compound customer requests may be difficult for a single-label "
            "intent classifier and can lead to incorrect evidence retrieval "
            "or escalation decisions."
        ),
        "proposed_fix": (
            "Implement multi-label intent classification and priority "
            "rules for compound requests."
        ),
    },
    {
        "failure_id": "FAIL_002",
        "failure_name": "Low Retrieval Similarity for Novel Edge Cases",
        "source": "ILLUSTRATIVE_HYPOTHESIS — see also measured LOW_RETRIEVAL_SIMILARITY category",
        "measurement_status": "hypothesis",
        "frequency": "NOT MEASURED as a distinct novel-phrasing subset",
        "illustrative_example": {
            "customer_message": (
                "I entered my old apartment number by mistake during "
                "one-click checkout 5 minutes ago."
            ),
            "retrieval_similarity": (
                "ILLUSTRATIVE — expected to be lower for lexically novel phrasing"
            ),
            "note": (
                "This example illustrates how TF-IDF retrieval can struggle "
                "with novel wording. Measured low-similarity escalations "
                "appear separately under LOW_RETRIEVAL_SIMILARITY."
            ),
        },
        "architectural_root_cause": (
            "TF-IDF cosine similarity is sensitive to vocabulary overlap."
        ),
        "hypothesis": (
            "Novel phrasing may produce weak historical evidence even when "
            "a semantically similar historical case exists."
        ),
        "proposed_fix": (
            "Replace or augment TF-IDF retrieval with dense semantic "
            "embeddings such as sentence-transformers."
        ),
    },
    {
        "failure_id": "FAIL_003",
        "failure_name": "Unnecessary Escalation on Self-Service Informational Queries",
        "source": "ILLUSTRATIVE_HYPOTHESIS — not measured from real evaluation run",
        "measurement_status": "hypothesis",
        "frequency": "NOT MEASURED — requires human policy review",
        "illustrative_example": {
            "customer_message": "How can I update my credit card on file?",
            "escalation_reason": (
                "ACCOUNT_SPECIFIC_ACTION_REQUIRED (illustrative)"
            ),
            "note": (
                "The current keyword policy may treat an informational "
                "question similarly to an account mutation request."
            ),
        },
        "architectural_root_cause": (
            "ACCOUNT_SPECIFIC_ACTION_REQUIRED uses keyword matching without "
            "fully distinguishing informational questions from mutations."
        ),
        "hypothesis": (
            "Keyword-based account-action rules may over-trigger on some "
            "informational customer questions."
        ),
        "proposed_fix": (
            "Distinguish informational account queries from account "
            "mutation requests using a dedicated intent or policy layer."
        ),
    },
    {
        "failure_id": "FAIL_004",
        "failure_name": "Noisy Text / Typo Degradation",
        "source": "ILLUSTRATIVE_HYPOTHESIS — observed during raw dataset inspection",
        "measurement_status": "hypothesis",
        "frequency": "NOT MEASURED — requires dedicated noisy-text evaluation",
        "illustrative_example": {
            "customer_message": (
                "ordr #99211 non deliverd state shows rtrnd to sender pls help"
            ),
            "predicted_intent": (
                "general_inquiry_feedback "
                "(illustrative — actual pipeline output may differ)"
            ),
            "note": (
                "This example is illustrative of noisy Twitter-style text."
            ),
        },
        "architectural_root_cause": (
            "Twitter abbreviations and spelling errors reduce TF-IDF overlap."
        ),
        "hypothesis": (
            "Noisy customer messages may reduce intent classification and "
            "retrieval quality."
        ),
        "proposed_fix": (
            "Add domain-specific normalization, spelling correction, or "
            "subword/embedding-based representations."
        ),
    },
    {
        "failure_id": "FAIL_005",
        "failure_name": "Insufficient Context Without Clarification Turn",
        "source": "ILLUSTRATIVE_HYPOTHESIS — see also measured INSUFFICIENT_CONTEXT_ESCALATION",
        "measurement_status": "hypothesis",
        "frequency": "NOT MEASURED as a clarification-turn experiment",
        "illustrative_example": {
            "customer_message": "@AmazonHelp help me",
            "escalation_reason": "INSUFFICIENT_CONTEXT",
            "note": (
                "The current policy intentionally escalates messages that "
                "do not contain enough information for safe automation."
            ),
        },
        "architectural_root_cause": (
            "No automatic clarification turn exists before escalation."
        ),
        "hypothesis": (
            "A safe clarification question could convert some insufficient-"
            "context escalations into auto-handleable cases."
        ),
        "proposed_fix": (
            "Implement multi-turn context accumulation with a clarification "
            "question before permanent escalation."
        ),
    },
]


def classify_case(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Assign a single category using priority order.

    Returns (category, case_kind) or None if the case does not belong to
    any tracked diagnostic/escalation category.
    """
    intent_data = item.get("intent", {})
    retrieval_data = item.get("retrieval", {})
    escalation_data = item.get("escalation", {})
    evaluation_data = item.get("evaluation", {})

    predicted_intent = intent_data.get("name")
    heuristic_intent = evaluation_data.get("reference_intent")
    retrieval_similarity = retrieval_data.get("top_similarity", 0.0)
    escalation_decision = escalation_data.get("decision")
    escalation_reason = escalation_data.get("reason_code")

    # 1. Intent mismatch vs heuristic (diagnostic, not human gold)
    if (
        predicted_intent
        and heuristic_intent
        and predicted_intent != heuristic_intent
    ):
        cat = "INTENT_MISMATCH_VS_HEURISTIC"
        return cat, CATEGORY_KIND[cat]

    # 2–5. Prefer the actual escalation reason code over similarity alone.
    # This prevents LOW_INTENT_CONFIDENCE / INSUFFICIENT_CONTEXT cases from
    # being mislabelled as LOW_RETRIEVAL_SIMILARITY merely because similarity
    # is also low.
    if escalation_decision == "ESCALATE":
        if escalation_reason == "LOW_INTENT_CONFIDENCE":
            cat = "LOW_INTENT_CONFIDENCE_ESCALATION"
            return cat, CATEGORY_KIND[cat]
        if escalation_reason == "INSUFFICIENT_CONTEXT":
            cat = "INSUFFICIENT_CONTEXT_ESCALATION"
            return cat, CATEGORY_KIND[cat]
        if escalation_reason == "SENSITIVE_REQUEST":
            cat = "SENSITIVE_REQUEST_ESCALATION"
            return cat, CATEGORY_KIND[cat]
        if escalation_reason == "ACCOUNT_SPECIFIC_ACTION_REQUIRED":
            cat = "ACCOUNT_SPECIFIC_ACTION_ESCALATION"
            return cat, CATEGORY_KIND[cat]
        if escalation_reason == "NO_RELEVANT_HISTORICAL_EVIDENCE":
            cat = "LOW_RETRIEVAL_SIMILARITY"
            return cat, CATEGORY_KIND[cat]

    # 6. Residual very-low similarity diagnostic (even if auto-handled)
    if retrieval_similarity is not None and retrieval_similarity < 0.40:
        cat = "LOW_RETRIEVAL_SIMILARITY"
        return cat, CATEGORY_KIND[cat]

    return None


def _example_from_item(
    item: Dict[str, Any],
    category: str,
    case_kind: str,
) -> Dict[str, Any]:
    intent_data = item.get("intent", {})
    retrieval_data = item.get("retrieval", {})
    escalation_data = item.get("escalation", {})
    generated_data = item.get("generated_reply", {})
    evaluation_data = item.get("evaluation", {})

    return {
        "category": category,
        "case_kind": case_kind,
        "measurement_status": "measured",
        "source": "REAL_PIPELINE_OUTPUT",
        "interpretation_note": CATEGORY_NOTES.get(category, ""),
        "customer_message": item.get("customer_message"),
        "conversation_id": evaluation_data.get("conversation_id"),
        "predicted_intent": intent_data.get("name"),
        "heuristic_intent": evaluation_data.get("reference_intent"),
        "reference_label_type": evaluation_data.get(
            "reference_label_type",
            "UNKNOWN",
        ),
        "confidence": intent_data.get("confidence"),
        "retrieval_similarity": retrieval_data.get("top_similarity", 0.0),
        "escalated": escalation_data.get("decision") == "ESCALATE",
        "escalation_reason": escalation_data.get("reason_code"),
        "generated_reply": generated_data.get("reply"),
        "likely_root_cause": CATEGORY_NOTES.get(category, ""),
        "proposed_fix": _proposed_fix_for(category),
    }


def _proposed_fix_for(category: str) -> str:
    fixes = {
        "INTENT_MISMATCH_VS_HEURISTIC": (
            "Obtain human golden labels; retrain/evaluate against human "
            "intents rather than heuristic keywords."
        ),
        "LOW_INTENT_CONFIDENCE_ESCALATION": (
            "Keep threshold; optionally add a clarification turn before "
            "permanent escalation for borderline confidence."
        ),
        "INSUFFICIENT_CONTEXT_ESCALATION": (
            "Ask a safe clarification question and re-evaluate after the "
            "customer replies."
        ),
        "SENSITIVE_REQUEST_ESCALATION": (
            "Retain escalation; refine keyword list with false-positive review."
        ),
        "ACCOUNT_SPECIFIC_ACTION_ESCALATION": (
            "Separate informational account FAQs from mutation requests."
        ),
        "LOW_RETRIEVAL_SIMILARITY": (
            "Augment TF-IDF with dense embeddings / cross-encoder re-ranking."
        ),
    }
    return fixes.get(category, "Investigate with human review.")


def collect_real_case_categories(
    pipeline_outputs_path: str = "results/pipeline_test_outputs.jsonl",
) -> Dict[str, Any]:
    """
    Scan real pipeline outputs and collect measured category counts plus
    one real example per category (priority order).
    """
    if not os.path.exists(pipeline_outputs_path):
        return {
            "pipeline_outputs_exist": False,
            "total_cases_scanned": 0,
            "category_counts": {},
            "examples_by_category": {},
            "top_categories": [],
        }

    category_counts: Dict[str, int] = {c: 0 for c in CATEGORY_PRIORITY}
    examples_by_category: Dict[str, Dict[str, Any]] = {}
    total = 0
    auto_handle = 0
    escalate = 0
    reason_code_counts: Dict[str, int] = {}

    with open(pipeline_outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1

            esc = item.get("escalation", {})
            decision = esc.get("decision")
            reason = esc.get("reason_code") or "AUTO_HANDLE"
            reason_code_counts[reason] = reason_code_counts.get(reason, 0) + 1
            if decision == "AUTO_HANDLE":
                auto_handle += 1
            elif decision == "ESCALATE":
                escalate += 1

            classified = classify_case(item)
            if not classified:
                continue
            category, case_kind = classified
            category_counts[category] = category_counts.get(category, 0) + 1
            if category not in examples_by_category:
                examples_by_category[category] = _example_from_item(
                    item, category, case_kind
                )

    # Top measured categories in priority order.
    # Skip ultra-rare categories (<10) when a later high-volume category exists,
    # so the reported top-5 stays useful (e.g. prefer LOW_RETRIEVAL_SIMILARITY
    # over a 3-case ACCOUNT_SPECIFIC niche). Full counts remain in category_counts.
    top_categories: List[Dict[str, Any]] = []
    for category in CATEGORY_PRIORITY:
        count = category_counts.get(category, 0)
        if count <= 0:
            continue
        if count < 10 and category != "LOW_RETRIEVAL_SIMILARITY":
            # Still keep rare categories available via examples_by_category.
            continue
        example = examples_by_category.get(category)
        top_categories.append(
            {
                "category": category,
                "case_kind": CATEGORY_KIND[category],
                "measurement_status": "measured",
                "observed_count": count,
                "observed_on": (
                    f"{pipeline_outputs_path} "
                    f"({total} held-out test pipeline outputs)"
                ),
                "interpretation_note": CATEGORY_NOTES[category],
                "real_example": example,
                "likely_root_cause": CATEGORY_NOTES[category],
                "proposed_fix": _proposed_fix_for(category),
            }
        )
        if len(top_categories) >= 5:
            break

    # If we skipped rare categories and have fewer than 5, backfill from priority.
    if len(top_categories) < 5:
        already = {c["category"] for c in top_categories}
        for category in CATEGORY_PRIORITY:
            if category in already:
                continue
            count = category_counts.get(category, 0)
            if count <= 0:
                continue
            example = examples_by_category.get(category)
            top_categories.append(
                {
                    "category": category,
                    "case_kind": CATEGORY_KIND[category],
                    "measurement_status": "measured",
                    "observed_count": count,
                    "observed_on": (
                        f"{pipeline_outputs_path} "
                        f"({total} held-out test pipeline outputs)"
                    ),
                    "interpretation_note": CATEGORY_NOTES[category],
                    "real_example": example,
                    "likely_root_cause": CATEGORY_NOTES[category],
                    "proposed_fix": _proposed_fix_for(category),
                }
            )
            if len(top_categories) >= 5:
                break

    return {
        "pipeline_outputs_exist": True,
        "total_cases_scanned": total,
        "auto_handle_count": auto_handle,
        "escalate_count": escalate,
        "reason_code_counts": reason_code_counts,
        "category_counts": category_counts,
        "examples_by_category": examples_by_category,
        "top_categories": top_categories,
    }


def collect_real_failures(
    pipeline_outputs_path: str = "results/pipeline_test_outputs.jsonl",
    min_failures: int = 5,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible helper: return up to min_failures real examples,
    one preferred per priority category.
    """
    collected = collect_real_case_categories(pipeline_outputs_path)
    examples = []
    for cat_block in collected.get("top_categories", []):
        ex = cat_block.get("real_example")
        if ex:
            # Preserve older key name used by some consumers
            ex = dict(ex)
            ex["failure_type"] = ex.get("category")
            examples.append(ex)
        if len(examples) >= min_failures:
            break
    return examples


def run_failure_analysis(
    pipeline_outputs_path: str = "results/pipeline_test_outputs.jsonl",
) -> Dict[str, Any]:
    """
    Build the failure analysis report from real pipeline outputs when available.

    Distinguishes:
    - measured categories with real examples and observed counts
    - expected safety escalations vs diagnostic disagreements vs limitations
    - separate architectural hypotheses that are NOT measured
    """
    logger.info("Running Failure Analysis...")
    os.makedirs("results", exist_ok=True)

    collected = collect_real_case_categories(pipeline_outputs_path)
    pipeline_outputs_exist = collected["pipeline_outputs_exist"]
    top_categories = collected.get("top_categories", [])
    real_examples = [
        c["real_example"] for c in top_categories if c.get("real_example")
    ]

    if pipeline_outputs_exist and top_categories:
        summary = (
            f"MEASURED categories from {collected['total_cases_scanned']} "
            f"held-out pipeline outputs. "
            f"Top {len(top_categories)} categories reported with real "
            f"examples. Escalations are NOT automatically labelled as "
            f"model failures."
        )
        failure_source = "REAL_PIPELINE_OUTPUTS"
        failure_status = "REAL_DATA"
        failure_modes = top_categories
    elif pipeline_outputs_exist:
        summary = (
            "Pipeline evaluation output exists, but no examples met the "
            "current categorization criteria. Architectural hypotheses are "
            "retained separately."
        )
        failure_source = (
            "NO_QUALIFYING_REAL_CASES — hypotheses retained separately"
        )
        failure_status = "NO_QUALIFYING_CASES_FOUND"
        failure_modes = FIVE_FAILURE_MODE_HYPOTHESES
    else:
        summary = (
            "BLOCKED — real pipeline evaluation output was not found. "
            "Failure modes below are ILLUSTRATIVE HYPOTHESES. "
            "Frequencies are NOT measured."
        )
        failure_source = "ILLUSTRATIVE_HYPOTHESES — not empirically measured"
        failure_status = "BLOCKED — REQUIRES REAL EVALUATION OUTPUT"
        failure_modes = FIVE_FAILURE_MODE_HYPOTHESES

    output = {
        "failure_analysis_status": failure_status,
        "failure_source": failure_source,
        "real_examples_collected": len(real_examples),
        "pipeline_outputs_path": pipeline_outputs_path,
        "pipeline_outputs_exist": pipeline_outputs_exist,
        "total_cases_scanned": collected.get("total_cases_scanned", 0),
        "auto_handle_count": collected.get("auto_handle_count", 0),
        "escalate_count": collected.get("escalate_count", 0),
        "reason_code_counts": collected.get("reason_code_counts", {}),
        "category_counts": collected.get("category_counts", {}),
        "summary": summary,
        "failure_modes": failure_modes,
        "hypotheses": FIVE_FAILURE_MODE_HYPOTHESES,
        "measurement_notes": {
            "intent_reference": (
                "HEURISTIC_NOT_HUMAN — intent mismatches are not proof "
                "of human-labelled classification errors."
            ),
            "escalation_vs_failure": (
                "Expected safety escalations (LOW_INTENT_CONFIDENCE, "
                "INSUFFICIENT_CONTEXT, SENSITIVE_REQUEST, "
                "ACCOUNT_SPECIFIC_ACTION) are policy behaviour, not "
                "automatically model failures."
            ),
            "prioritization": (
                "Escalation reason codes take priority over raw similarity "
                "so LOW_INTENT_CONFIDENCE / INSUFFICIENT_CONTEXT are not "
                "mislabelled as LOW_RETRIEVAL_SIMILARITY."
            ),
            "retrieval_metric": (
                "LOW_RETRIEVAL_SIMILARITY is threshold-/reason-based; "
                "it does not prove human irrelevance."
            ),
            "frequencies": (
                "observed_count values are measured on "
                f"{pipeline_outputs_path} only. Hypothesis frequencies "
                "are NOT invented."
            ),
        },
    }

    with open("results/failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Failure Analysis complete: %s", summary)
    return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    result = run_failure_analysis()
    print(json.dumps(result, indent=2))
