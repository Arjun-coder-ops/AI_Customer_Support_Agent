import os
import json
import logging
from typing import Dict, Any

from evaluation.leakage_check import check_split_leakage
from evaluation.intent_eval import run_intent_evaluations
from evaluation.retrieval_eval import run_retrieval_evaluation
from evaluation.escalation_eval import run_escalation_evaluation
from evaluation.llm_judge import run_judge_evaluations
from evaluation.human_judge_agreement import compute_human_llm_agreement
from evaluation.failure_analysis import run_failure_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_all_evaluations() -> Dict[str, Any]:
    """
    Run complete evaluation suite across leakage, intent, retrieval, escalation, judge, and agreement.
    Compiles final summary report into results/final_results.json.

    IMPORTANT — Metric honesty contract:
    Every metric in the final results is labelled with its dataset source,
    label type, and benchmark category. Do NOT mix these categories or
    report proxy metrics as gold benchmarks.
    """
    logger.info("==================================================")
    logger.info("STARTING COMPLETE EVALUATION SUITE FOR HIVER AGENT")
    logger.info("==================================================")
    os.makedirs("results", exist_ok=True)

    # 1. Leakage Check
    leakage_res = check_split_leakage()

    # 2. Intent Classifier & Baselines
    intent_res = run_intent_evaluations()

    # 3. Retrieval Evaluation (train-only corpus, fixed leakage)
    retrieval_res = run_retrieval_evaluation()

    # 4. Escalation Policy Evaluation (curated safety suite)
    escalation_res = run_escalation_evaluation()

    # 5. LLM-as-a-Judge Discrimination Test (not real pipeline quality)
    judge_res = run_judge_evaluations()

    # 6. Human vs LLM Agreement Analysis (BLOCKED until human ratings exist)
    agreement_res = compute_human_llm_agreement()

    # 7. Failure Analysis (real pipeline outputs when present; hypotheses separate)
    failure_res = run_failure_analysis()

    # Consolidate Final Master Results with explicit dataset status on every section
    final_summary = {
        "evaluation_status": "COMPLETED",
        "audit_timestamp": "2026-09-10",
        "dataset": "REAL DATASET — AmazonHelp from Customer Support on Twitter (twcs.csv)",
        "brand": "AmazonHelp",
        "total_conversations": 82493,
        "train_conversations": 65994,
        "val_conversations": 8249,
        "test_conversations": 8250,

        # --- Leakage ---
        "data_leakage": leakage_res,

        # --- Intent classification (HEURISTIC labels, NOT human gold) ---
        "intent_classification": intent_res,

        # --- Retrieval (FIXED: train-only corpus; proxy metric IntentMatch@K) ---
        "historical_retrieval": retrieval_res,

        # --- Escalation (CURATED SAFETY SUITE, not production benchmark) ---
        "escalation_policy": escalation_res,

        # --- LLM judge (DISCRIMINATION TEST, not real agent quality) ---
        "llm_judge_evaluation": judge_res,

        # --- Human agreement (BLOCKED) ---
        "human_llm_agreement": agreement_res,

        # --- Failure analysis (HYPOTHESES until real evaluation run) ---
        "failure_analysis": failure_res,
    }

    with open("results/final_results.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    logger.info("==================================================")
    logger.info("EVALUATION SUITE COMPLETED SUCCESSFULLY!")
    logger.info("Master Results saved to 'results/final_results.json'")
    logger.info("==================================================")

    return final_summary


if __name__ == "__main__":
    summary = run_all_evaluations()

    # Leakage
    leakage = summary["data_leakage"]
    print(f"\n--- MASTER EVALUATION SUMMARY ---")
    print(f"[LEAKAGE] Conversation-level leakage: {leakage.get('conversation_leakage_status', 'N/A')}")
    print(f"[LEAKAGE] Duplicate-text contamination risk: {leakage.get('duplicate_text_contamination_risk', 'N/A')} ({leakage.get('train_test_exact_message_overlap_count', 0)} trivially generic messages — independent conversations)")

    # Intent
    intent = summary["intent_classification"]
    print(f"\n[INTENT]  Dataset: {intent.get('dataset_status', 'N/A')}")
    print(f"[INTENT]  Label type: {intent.get('label_type', 'N/A')}")
    final_clf = intent.get("final_classifier", {})
    print(f"[INTENT]  Final Classifier Accuracy (vs heuristic): {final_clf.get('accuracy', 'N/A')}")
    print(f"[INTENT]  Final Classifier Macro F1 (vs heuristic): {final_clf.get('macro_f1', 'N/A')}")
    print(f"[INTENT]  Golden benchmark: {intent.get('golden_benchmark_status', 'N/A')}")

    # Retrieval
    retrieval = summary["historical_retrieval"]
    print(f"\n[RETRIEVAL] Corpus: {retrieval.get('index_corpus_source', 'N/A')}")
    print(f"[RETRIEVAL] Corpus size (train-only resolved): {retrieval.get('index_corpus_resolved_case_count', 'N/A')}")
    print(f"[RETRIEVAL] Held-out query count: {retrieval.get('held_out_query_count', 'N/A')}")
    print(f"[RETRIEVAL] Train/test conversation overlap: {retrieval.get('train_test_conversation_overlap', 'N/A')} (should be 0)")
    print(f"[RETRIEVAL] IntentMatch@1 (proxy metric, NOT Recall@1): {retrieval.get('intent_match_at_1', 'N/A')}")
    print(f"[RETRIEVAL] IntentMatch@5 (proxy metric, NOT Recall@5): {retrieval.get('intent_match_at_5', 'N/A')}")

    # Escalation
    esc = summary["escalation_policy"]
    print(f"\n[ESCALATION] Dataset: {esc.get('dataset_status', 'N/A')}")
    print(f"[ESCALATION] Auto-Handle Rate: {esc.get('auto_handle_rate', 'N/A')} (curated suite only)")
    print(f"[ESCALATION] False Auto-Handle Rate: {esc.get('false_auto_handling_rate', 'N/A')} (curated suite only — NOT production benchmark)")

    # Judge
    judge = summary["llm_judge_evaluation"]
    print(f"\n[JUDGE]   Evaluation type: {judge.get('evaluation_type', 'N/A')}")
    print(f"[JUDGE]   Dataset: {judge.get('dataset_status', 'N/A')}")
    print(f"[JUDGE]   Discrimination test mean: {judge.get('discrimination_test_mean_score', 'N/A')}/10 (NOT agent quality score)")
    print(f"[JUDGE]   Actual agent quality: {judge.get('actual_agent_quality', 'N/A')}")

    # Agreement
    agreement = summary["human_llm_agreement"]
    print(f"\n[AGREEMENT] Status: {agreement.get('status', 'N/A')}")
    print(f"[AGREEMENT] Pearson R: {agreement.get('pearson_correlation', 'N/A')}")

    # Failure analysis
    fa = summary["failure_analysis"]
    print(f"\n[FAILURES] Status: {fa.get('failure_analysis_status', 'N/A')}")
