import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

HUMAN_RATINGS_FILE = "data/golden/human_ratings.json"

def compute_human_llm_agreement(
    human_ratings_file: str = HUMAN_RATINGS_FILE,
) -> Dict[str, Any]:
    """
    Compute statistical agreement metrics between Human Evaluator and LLM-as-a-Judge ratings.
    If genuine human rating data does not exist, explicitly returns BLOCKED status without fabricating metrics.
    """
    logger.info("Auditing Human vs. LLM Judge Agreement...")
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(human_ratings_file):
        logger.warning(f"Human ratings file not found at '{human_ratings_file}'. Marking BLOCKED — REQUIRES HUMAN INPUT.")
        output = {
            "status": "BLOCKED — REQUIRES HUMAN INPUT",
            "is_human_evaluated": False,
            "message": f"Human rating file '{human_ratings_file}' is missing. Human evaluation dataset must be provided by a human reviewer.",
            "human_mean_score": "NOT YET MEASURED",
            "llm_judge_mean_score": "NOT YET MEASURED",
            "pearson_correlation": "NOT YET MEASURED",
            "quadratic_weighted_kappa": "NOT YET MEASURED",
        }
        with open("results/judge_agreement.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        return output

    with open(human_ratings_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    human_ratings = data.get("human_ratings", [])
    llm_ratings = data.get("llm_ratings", [])

    if not human_ratings or not llm_ratings or len(human_ratings) != len(llm_ratings):
        logger.error("Human or LLM ratings data in file is empty or mismatched.")
        output = {
            "status": "BLOCKED — INVALID HUMAN RATING FILE",
            "is_human_evaluated": False,
            "message": "Human and LLM rating lists must be non-empty and equal in length.",
        }
        with open("results/judge_agreement.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        return output

    h_arr = np.array(human_ratings)
    l_arr = np.array(llm_ratings)
    n = len(h_arr)

    diffs = np.abs(h_arr - l_arr)
    exact_match = int(np.sum(diffs == 0))
    near_match = int(np.sum(diffs <= 1))

    mean_h = float(np.mean(h_arr))
    mean_l = float(np.mean(l_arr))
    mean_diff = float(np.mean(diffs))

    exact_pct = round(float(exact_match / n), 4)
    near_pct = round(float(near_match / n), 4)

    corr, p_val = pearsonr(h_arr, l_arr)
    qwk = cohen_kappa_score(h_arr, l_arr, weights="quadratic")

    output = {
        "status": "COMPLETED",
        "is_human_evaluated": True,
        "sample_size": n,
        "human_mean_score": round(mean_h, 2),
        "llm_judge_mean_score": round(mean_l, 2),
        "mean_absolute_difference": round(mean_diff, 2),
        "exact_agreement_count": exact_match,
        "exact_agreement_pct": exact_pct,
        "near_agreement_count_within_1pt": near_match,
        "near_agreement_pct": near_pct,
        "pearson_correlation": round(float(corr), 4),
        "pearson_p_value": round(float(p_val), 6),
        "quadratic_weighted_kappa": round(float(qwk), 4),
    }

    with open("results/judge_agreement.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Agreement Analysis complete: Pearson R={corr:.4f}, QWK={qwk:.4f}")
    return output

if __name__ == "__main__":
    res = compute_human_llm_agreement()
    print(json.dumps(res, indent=2))
