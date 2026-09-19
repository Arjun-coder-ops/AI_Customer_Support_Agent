
import json
import logging
from pathlib import Path

from src.pipeline import SupportAgentPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TRAIN_PATH = Path("data/processed/train.jsonl")
TEST_PATH = Path("data/processed/test.jsonl")
OUTPUT_PATH = Path("results/pipeline_test_outputs.jsonl")


def load_jsonl(path: Path):
    """Load non-empty JSONL records from a file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def heuristic_intent(message: str) -> str:
    """
    Evaluation-only reference label.

    IMPORTANT:
    This is NOT a human gold label.
    It mirrors the heuristic taxonomy currently used by
    the dataset preparation/evaluation pipeline.
    """
    msg = message.lower()

    if any(k in msg for k in ["delay", "where is", "tracking", "late"]):
        return "shipping_delay"

    if any(k in msg for k in ["missing", "incomplete"]):
        return "missing_item"

    if any(k in msg for k in ["cancel", "stop"]):
        return "order_cancellation"

    if any(k in msg for k in ["refund", "return"]):
        return "refund_return_request"

    if any(k in msg for k in ["lock", "login", "password"]):
        return "account_access_issue"

    if any(k in msg for k in ["charge", "paid", "double"]):
        return "payment_billing_issue"

    if any(k in msg for k in ["damage", "defect", "broken"]):
        return "product_defect_damage"

    return "general_inquiry_feedback"


def build_retrieval_cases(train_cases):
    """
    Convert processed training conversations into the schema expected
    by HistoricalCaseRetrievalIndex.

    IMPORTANT:
    Only TRAINING conversations are passed into the retrieval index.
    The held-out test set is never used as retrieval evidence.
    """
    retrieval_cases = []

    for case in train_cases:
        customer_message = case.get("customer_initial_message", "")
        historical_response = case.get("support_final_response", "")

        if not customer_message:
            continue

        retrieval_cases.append(
            {
                "case_id": case.get("conversation_id"),
                "conversation_id": case.get("conversation_id"),
                "customer_message": customer_message,
                "historical_response": historical_response,
                "turn_count": case.get("turn_count", 1),
            }
        )

    if not retrieval_cases:
        raise ValueError(
            "No valid retrieval cases were created from train.jsonl. "
            "Expected 'customer_initial_message' in training records."
        )

    logger.info(
        "Prepared %d training cases for historical retrieval.",
        len(retrieval_cases),
    )

    return retrieval_cases


def main():
    # ---------------------------------------------------------
    # 1. Validate required files
    # ---------------------------------------------------------
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {TEST_PATH}")

    # ---------------------------------------------------------
    # 2. Load train and held-out test data
    # ---------------------------------------------------------
    logger.info("Loading training data...")
    train_cases = list(load_jsonl(TRAIN_PATH))

    logger.info("Loading held-out test data...")
    test_cases = list(load_jsonl(TEST_PATH))

    logger.info("Train cases: %d", len(train_cases))
    logger.info("Test cases: %d", len(test_cases))

    # ---------------------------------------------------------
    # 3. Convert training records for retrieval
    # ---------------------------------------------------------
    retrieval_cases = build_retrieval_cases(train_cases)

    # ---------------------------------------------------------
    # 4. Create pipeline in MOCK mode
    # ---------------------------------------------------------
    # We intentionally use Mock LLM here.
    #
    # This evaluates:
    #   - intent classification
    #   - historical retrieval
    #   - escalation policy
    #   - end-to-end pipeline wiring
    #
    # without consuming Gemini API quota.
    pipeline = SupportAgentPipeline(
        brand_name="AmazonHelp",
        use_mock=True,
    )

    # ---------------------------------------------------------
    # 5. Train classifier + build retrieval index
    # ---------------------------------------------------------
    # CRITICAL:
    # Only training conversations are used for retrieval evidence.
    # The held-out test conversations remain completely excluded.
    train_ids = {
        c.get("conversation_id")
        for c in train_cases
        if c.get("conversation_id")
    }
    test_ids = {
        c.get("conversation_id")
        for c in test_cases
        if c.get("conversation_id")
    }
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(
            f"Train/test conversation leakage detected before indexing: "
            f"{len(overlap)} overlapping IDs."
        )

    retrieval_ids = {
        c.get("conversation_id") or c.get("case_id")
        for c in retrieval_cases
    }
    retrieval_test_overlap = retrieval_ids & test_ids
    if retrieval_test_overlap:
        raise RuntimeError(
            "Retrieval index would include test conversations "
            f"({len(retrieval_test_overlap)} IDs). Aborting."
        )

    pipeline.train_and_index(
        train_cases=train_cases,
        resolved_cases=retrieval_cases,
    )

    indexed_ids = {
        c.get("conversation_id") or c.get("case_id")
        for c in pipeline.retrieval_index.cases
    }
    if indexed_ids & test_ids:
        raise RuntimeError(
            "Post-index leakage check failed: retrieval index contains "
            "held-out test conversation IDs."
        )
    logger.info(
        "Leakage checks passed. Retrieval index size=%d (train-only).",
        len(pipeline.retrieval_index.cases),
    )

    # ---------------------------------------------------------
    # 6. Prepare output
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    # ---------------------------------------------------------
    # 7. Run complete pipeline on held-out test set
    # ---------------------------------------------------------
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for i, case in enumerate(test_cases, start=1):

            message = case.get("customer_initial_message", "").strip()

            if not message:
                skipped += 1
                continue

            # Run the actual support-agent pipeline.
            result = pipeline.process_message(
                customer_message=message,
                turn_count=case.get("turn_count", 1),
            )

            # -------------------------------------------------
            # Evaluation-only metadata
            # -------------------------------------------------
            #
            # These fields are NOT used by the pipeline itself.
            # They are stored so the evaluation scripts can compare
            # predictions against the held-out reference information.
            result["evaluation"] = {
                "source": "REAL_DATASET_TEST_SPLIT",
                "conversation_id": case.get("conversation_id"),
                "reference_intent": heuristic_intent(message),
                "reference_label_type": "HEURISTIC_NOT_HUMAN",
                "historical_support_response": case.get(
                    "support_final_response",
                    "",
                ),
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            processed += 1

            if processed % 250 == 0:
                logger.info(
                    "Processed %d/%d test cases...",
                    processed,
                    len(test_cases),
                )

    # ---------------------------------------------------------
    # 8. Final summary
    # ---------------------------------------------------------
    logger.info("Pipeline evaluation complete.")
    logger.info("Processed test cases: %d", processed)
    logger.info("Skipped test cases: %d", skipped)
    logger.info("Output: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()

