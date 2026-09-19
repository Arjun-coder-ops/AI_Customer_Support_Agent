import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISSUE 5 — Golden Set Status
# ---------------------------------------------------------------------------
# The golden set human-review track is BLOCKED until annotators finish.
# - ~200 candidates exist in labeling_candidates.jsonl
# - is_human_reviewed remains false until a human labels them
# - Hiver assignment requires 150–250 HAND-LABELLED examples
#
# Workflow:
# 1. python -m src.data.label
# 2. python -m src.data.label --review
# 3. python -m src.data.label --promote
# 4. Once 150+ examples are reviewed, human-golden evaluation unlocks
# ---------------------------------------------------------------------------

GOLDEN_SET_PATH = "data/golden/golden_set.jsonl"
MIN_HUMAN_REVIEWED = 150


def validate_golden_set(golden_path: str = GOLDEN_SET_PATH) -> Dict[str, Any]:
    """
    Validate the golden set and return its status.
    Returns BLOCKED status if insufficient human-reviewed examples.
    """
    total = 0
    human_reviewed = 0
    provisional = 0

    if os.path.exists(golden_path):
        for line in open(golden_path, "r", encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1
            if item.get("is_human_reviewed", False):
                human_reviewed += 1
            else:
                provisional += 1

    is_ready = human_reviewed >= MIN_HUMAN_REVIEWED

    return {
        "total_candidates": total,
        "human_reviewed_count": human_reviewed,
        "provisional_count": provisional,
        "required_minimum": MIN_HUMAN_REVIEWED,
        "is_evaluation_ready": is_ready,
        "status": (
            f"READY — {human_reviewed} human-reviewed examples (≥ {MIN_HUMAN_REVIEWED} required)"
            if is_ready
            else f"BLOCKED — REQUIRES HUMAN INPUT: {human_reviewed}/{MIN_HUMAN_REVIEWED} examples reviewed. "
                 f"Please annotate labeling_candidates.jsonl and promote reviewed examples to golden_set.jsonl."
        ),
    }


def require_human_golden_set(golden_path: str = GOLDEN_SET_PATH) -> None:
    """
    Raises RuntimeError if the golden set does not have sufficient human-reviewed examples.
    Use this guard in evaluation functions that claim to use a 'gold benchmark'.
    """
    status = validate_golden_set(golden_path)
    if not status["is_evaluation_ready"]:
        raise RuntimeError(
            f"GOLDEN SET EVALUATION BLOCKED: {status['status']}. "
            "Do not report this as a gold benchmark until the human review is complete."
        )
