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
    """
    logger.info("==================================================")
    logger.info("STARTING COMPLETE EVALUATION SUITE FOR HIVER AGENT")
    logger.info("==================================================")
    os.makedirs("results", exist_ok=True)

    # 1. Leakage Check
    leakage_res = check_split_leakage()

    # 2. Intent Classifier & Baselines
    intent_res = run_intent_evaluations()

    # 3. Retrieval Evaluation
    retrieval_res = run_retrieval_evaluation()

    # 4. Escalation Policy Evaluation
    escalation_res = run_escalation_evaluation()

    # 5. LLM-as-a-Judge Evaluation
    judge_res = run_judge_evaluations()

    # 6. Human vs LLM Agreement Analysis
    agreement_res = compute_human_llm_agreement()

    # 7. Failure Analysis
    failure_res = run_failure_analysis()

    # Consolidate Final Master Results
    final_summary = {
        "evaluation_status": "COMPLETED",
        "data_leakage": leakage_res,
        "intent_classification": intent_res,
        "historical_retrieval": retrieval_res,
        "escalation_policy": escalation_res,
        "llm_judge_evaluation": judge_res,
        "human_llm_agreement": agreement_res,
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
    print("\n--- MASTER EVALUATION SUMMARY ---")
    print(f"Data Leakage Status: {summary['data_leakage']['leakage_status']}")
    print(f"Intent Classifier Accuracy: {summary['intent_classification']['final_classifier']['accuracy']}")
    print(f"Retrieval Recall@1: {summary['historical_retrieval']['recall_at_1']}")
    print(f"Escalation Auto-Handle Rate: {summary['escalation_policy']['auto_handle_rate']}")
    print(f"Escalation False Auto-Handling Rate: {summary['escalation_policy']['false_auto_handling_rate']}")
    print(f"LLM Judge Mean Score: {summary['llm_judge_evaluation']['mean_total_score']}/10")
    print(f"Human/LLM Agreement Pearson R: {summary['human_llm_agreement']['pearson_correlation']}")
