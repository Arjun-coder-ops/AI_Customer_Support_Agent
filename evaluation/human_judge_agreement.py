import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

HUMAN_RATINGS_FILE = "data/golden/human_ratings.json"
JUDGE_SHARED_EXAMPLES_FILE = "data/golden/judge_agreement_examples.jsonl"


HOWTO_CREATE_HUMAN_RATINGS = """
HOW TO CREATE data/golden/human_ratings.json
===========================================

1. Select the SAME examples the judge will score.
   Preferred source: data/golden/judge_agreement_examples.jsonl
   (create with: python -m evaluation.human_judge_agreement --prepare-examples)

   Each example must include:
     - example_id
     - customer_message
     - generated_reply
     - predicted_intent
     - retrieved_evidence

2. Rate each generated_reply on the same 0–10 total scale used by the LLM judge
   (Correctness + Groundedness + Relevance + Completeness + Professionalism).

3. Write data/golden/human_ratings.json with this schema:

{
  "rated_by": "YOUR_NAME",
  "rating_date": "YYYY-MM-DD",
  "rubric": "0-10 total across correctness/groundedness/relevance/completeness/professionalism",
  "examples": [
    {
      "example_id": "agr_001",
      "customer_message": "...",
      "generated_reply": "...",
      "human_total_score": 7,
      "llm_total_score": null
    }
  ]
}

Notes:
- llm_total_score may be null; the agreement script will fill it by
  running the judge on the SAME generated_reply text.
- Do NOT invent ratings.
- Until this file exists with real human scores, agreement remains BLOCKED.
""".strip()


def prepare_agreement_examples(
    pipeline_outputs_path: str = "results/pipeline_test_outputs.jsonl",
    output_path: str = JUDGE_SHARED_EXAMPLES_FILE,
    n: int = 30,
    seed: int = 42,
) -> int:
    """
    Sample shared examples (same customer message + same generated reply)
    for human rating and LLM judge scoring.
    """
    if not os.path.exists(pipeline_outputs_path):
        print(
            f"BLOCKED: {pipeline_outputs_path} not found. "
            "Run `python -m evaluation.pipeline_eval` first."
        )
        return 0

    rows: List[Dict[str, Any]] = []
    with open(pipeline_outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        print("No pipeline outputs available.")
        return 0

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(rows), size=min(n, len(rows)), replace=False)

    examples = []
    for i, idx in enumerate(sorted(idxs), start=1):
        item = rows[int(idx)]
        examples.append(
            {
                "example_id": f"agr_{i:03d}",
                "conversation_id": item.get("evaluation", {}).get("conversation_id"),
                "customer_message": item.get("customer_message"),
                "predicted_intent": item.get("intent", {}).get("name"),
                "retrieved_evidence": item.get("retrieval", {}).get("evidence_cases", []),
                "generated_reply": item.get("generated_reply", {}).get("reply"),
                "source": "REAL_PIPELINE_TEST_OUTPUT",
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} shared examples to {output_path}")
    print(HOWTO_CREATE_HUMAN_RATINGS)
    return len(examples)


def compute_human_llm_agreement(
    human_ratings_file: str = HUMAN_RATINGS_FILE,
    use_mock_judge: Optional[bool] = True,
) -> Dict[str, Any]:
    """
    Compute agreement between human ratings and LLM-as-a-Judge on the SAME replies.

    If human_ratings.json does not exist, returns BLOCKED without fabricating metrics.
    """
    logger.info("Auditing Human vs. LLM Judge Agreement...")
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(human_ratings_file):
        logger.warning(
            "Human ratings file not found at '%s'. Marking BLOCKED.",
            human_ratings_file,
        )
        output = {
            "status": "BLOCKED — REQUIRES HUMAN INPUT",
            "is_human_evaluated": False,
            "message": (
                f"Human rating file '{human_ratings_file}' is missing. "
                "Human evaluation must be provided by a human reviewer."
            ),
            "how_to_create": HOWTO_CREATE_HUMAN_RATINGS,
            "shared_examples_file": JUDGE_SHARED_EXAMPLES_FILE,
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

    # Preferred schema: examples[{example_id, generated_reply, human_total_score, ...}]
    examples = data.get("examples")
    human_ratings: List[float] = []
    llm_ratings: List[float] = []

    if examples:
        from evaluation.llm_judge import LLMJudgeEvaluator

        judge = LLMJudgeEvaluator(use_mock=use_mock_judge)
        for ex in examples:
            human_score = ex.get("human_total_score")
            if human_score is None:
                continue
            llm_score = ex.get("llm_total_score")
            if llm_score is None:
                judged = judge.evaluate_reply(
                    customer_message=ex.get("customer_message", ""),
                    retrieved_evidence=ex.get("retrieved_evidence", []),
                    generated_reply=ex.get("generated_reply", ""),
                    predicted_intent=ex.get("predicted_intent", "general_inquiry_feedback"),
                )
                llm_score = judged.get("total_score")
            if llm_score is None:
                continue
            human_ratings.append(float(human_score))
            llm_ratings.append(float(llm_score))
    else:
        # Legacy parallel arrays (must be same length / same order / same replies)
        human_ratings = [float(x) for x in data.get("human_ratings", [])]
        llm_ratings = [float(x) for x in data.get("llm_ratings", [])]

    if not human_ratings or not llm_ratings or len(human_ratings) != len(llm_ratings):
        logger.error("Human or LLM ratings data is empty or mismatched.")
        output = {
            "status": "BLOCKED — INVALID HUMAN RATING FILE",
            "is_human_evaluated": False,
            "message": (
                "Human and LLM rating lists must be non-empty and equal in length. "
                "Prefer the examples[] schema so both raters score the SAME reply."
            ),
            "how_to_create": HOWTO_CREATE_HUMAN_RATINGS,
        }
        with open("results/judge_agreement.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        return output

    try:
        from scipy.stats import pearsonr
        from sklearn.metrics import cohen_kappa_score
    except ImportError as e:
        output = {
            "status": "BLOCKED — MISSING DEPENDENCY",
            "message": f"Need scipy/sklearn for agreement stats: {e}",
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

    if n >= 2 and float(np.std(h_arr)) > 0 and float(np.std(l_arr)) > 0:
        corr, p_val = pearsonr(h_arr, l_arr)
        corr_v = round(float(corr), 4)
        p_v = round(float(p_val), 6)
    else:
        corr_v = "NOT COMPUTABLE — need >=2 samples with variance"
        p_v = "NOT COMPUTABLE"

    # QWK requires integer ordinal labels
    try:
        qwk = cohen_kappa_score(
            np.rint(h_arr).astype(int),
            np.rint(l_arr).astype(int),
            weights="quadratic",
        )
        qwk_v = round(float(qwk), 4)
    except Exception:
        qwk_v = "NOT COMPUTABLE"

    output = {
        "status": "COMPLETED",
        "is_human_evaluated": True,
        "sample_size": n,
        "evaluation_source": human_ratings_file,
        "human_mean_score": round(mean_h, 2),
        "llm_judge_mean_score": round(mean_l, 2),
        "mean_absolute_difference": round(mean_diff, 2),
        "exact_agreement_count": exact_match,
        "exact_agreement_pct": exact_pct,
        "near_agreement_count_within_1pt": near_match,
        "near_agreement_pct": near_pct,
        "pearson_correlation": corr_v,
        "pearson_p_value": p_v,
        "quadratic_weighted_kappa": qwk_v,
    }

    with open("results/judge_agreement.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Agreement Analysis complete: Pearson R=%s, QWK=%s",
        corr_v,
        qwk_v,
    )
    return output


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-examples",
        action="store_true",
        help="Sample shared examples for human+LLM rating",
    )
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    if args.prepare_examples:
        prepare_agreement_examples(n=args.n)
    else:
        res = compute_human_llm_agreement()
        print(json.dumps(res, indent=2))
