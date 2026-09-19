"""
Human golden-set labeling workflow.

Commands:
  python -m src.data.label                  # generate ~200 candidates
  python -m src.data.label --review         # interactive human review CLI
  python -m src.data.label --promote        # promote reviewed candidates
  python -m src.data.label --status         # show review counts

DO NOT mark candidates as human-reviewed automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

from src.intent.taxonomy import IntentTaxonomy

logger = logging.getLogger(__name__)

CANDIDATES_PATH = "data/golden/labeling_candidates.jsonl"
GOLDEN_SET_PATH = "data/golden/golden_set.jsonl"
VALID_INTENTS = None  # filled at runtime from taxonomy


def _valid_intents() -> List[str]:
    global VALID_INTENTS
    if VALID_INTENTS is None:
        VALID_INTENTS = IntentTaxonomy().get_intent_names()
    return VALID_INTENTS


def _heuristic_intent(msg: str) -> str:
    msg_lower = msg.lower()
    if any(k in msg_lower for k in ["delay", "where is", "tracking", "late", "status"]):
        return "shipping_delay"
    if any(k in msg_lower for k in ["missing", "incomplete"]):
        return "missing_item"
    if any(k in msg_lower for k in ["cancel", "stop"]):
        return "order_cancellation"
    if any(k in msg_lower for k in ["refund", "return", "back"]):
        return "refund_return_request"
    if any(k in msg_lower for k in ["lock", "login", "password", "account"]):
        return "account_access_issue"
    if any(k in msg_lower for k in ["charge", "paid", "double", "billing"]):
        return "payment_billing_issue"
    if any(k in msg_lower for k in ["damage", "defect", "broken", "crack"]):
        return "product_defect_damage"
    return "general_inquiry_feedback"


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_golden_candidates(
    test_path: str = "data/processed/test.jsonl",
    val_path: str = "data/processed/val.jsonl",
    output_candidates_path: str = CANDIDATES_PATH,
    target_count: int = 200,
) -> int:
    """
    Extract stratified-ish candidate customer messages for human labeling.

    Uses ONLY held-out test/val splits to avoid train leakage.
    Candidates are NOT human-reviewed.
    """
    os.makedirs("data/golden", exist_ok=True)

    sources: List[Dict[str, Any]] = []
    for path in [test_path, val_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sources.append(json.loads(line))

    if not sources:
        logger.warning(
            "No test/val split files found. Cannot generate real candidates."
        )
        print(
            "BLOCKED: data/processed/test.jsonl and/or val.jsonl missing. "
            "Run `python -m src.data.prepare` after placing twcs.csv first."
        )
        return 0

    # Aim for diversity across heuristic intents without fabricating labels.
    intent_buckets: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in _valid_intents()
    }
    seen_messages = set()

    for idx, item in enumerate(sources):
        msg = item.get("customer_initial_message", "").strip()
        conv_id = item.get("conversation_id", f"conv_heldout_{idx}")
        if not msg or msg in seen_messages:
            continue
        seen_messages.add(msg)
        suggested = _heuristic_intent(msg)
        intent_buckets.setdefault(suggested, []).append(
            {
                "candidate_id": f"cand_{idx + 1:04d}",
                "conversation_id": conv_id,
                "customer_message": msg,
                "suggested_intent": suggested,
                "human_verified_intent": None,
                "is_human_reviewed": False,
                "review_status": "PENDING_HUMAN_REVIEW",
                "reviewer": None,
                "review_date": None,
                "escalation_decision": None,
                "notes": None,
            }
        )

    # Round-robin sample across intents for diversity.
    candidates: List[Dict[str, Any]] = []
    pointers = {k: 0 for k in intent_buckets}
    while len(candidates) < target_count:
        progressed = False
        for intent_name in _valid_intents():
            bucket = intent_buckets.get(intent_name, [])
            ptr = pointers[intent_name]
            if ptr < len(bucket):
                candidates.append(bucket[ptr])
                pointers[intent_name] = ptr + 1
                progressed = True
                if len(candidates) >= target_count:
                    break
        if not progressed:
            break

    # Re-number candidate IDs sequentially for readability.
    for i, c in enumerate(candidates, start=1):
        c["candidate_id"] = f"cand_{i:04d}"

    _write_jsonl(output_candidates_path, candidates)

    # Keep golden_set.jsonl as provisional seeds — NEVER mark as human-reviewed.
    golden_rows = []
    for c in candidates:
        golden_rows.append(
            {
                "id": c["candidate_id"],
                "conversation_id": c["conversation_id"],
                "customer_message": c["customer_message"],
                "suggested_intent": c["suggested_intent"],
                "intent": c["suggested_intent"],  # provisional only
                "human_verified_intent": None,
                "is_human_reviewed": False,
                "review_status": "PROVISIONAL_SEED_PENDING_HUMAN_AUDIT",
                "label_source": "HEURISTIC_SUGGESTION_NOT_HUMAN",
            }
        )
    _write_jsonl(GOLDEN_SET_PATH, golden_rows)

    logger.info(
        "Generated %s golden labeling candidates at '%s'",
        len(candidates),
        output_candidates_path,
    )
    print("\n--- GOLDEN SET CANDIDATE GENERATION COMPLETE ---")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Candidates File: {output_candidates_path}")
    print(f"Golden Set File: {GOLDEN_SET_PATH}")
    print("IMPORTANT: All candidates are NOT human-reviewed.")
    print("Next: python -m src.data.label --review")
    return len(candidates)


def status_report(
    candidates_path: str = CANDIDATES_PATH,
    golden_path: str = GOLDEN_SET_PATH,
) -> Dict[str, Any]:
    candidates = _load_jsonl(candidates_path)
    golden = _load_jsonl(golden_path)
    cand_reviewed = sum(1 for c in candidates if c.get("is_human_reviewed"))
    gold_reviewed = sum(1 for g in golden if g.get("is_human_reviewed"))
    report = {
        "candidates_total": len(candidates),
        "candidates_human_reviewed": cand_reviewed,
        "candidates_pending": len(candidates) - cand_reviewed,
        "golden_total": len(golden),
        "golden_human_reviewed": gold_reviewed,
        "required_minimum": 150,
        "target_range": "150-250",
        "status": (
            "READY_FOR_GOLD_EVAL"
            if gold_reviewed >= 150
            else "BLOCKED — REQUIRES HUMAN INPUT"
        ),
    }
    print(json.dumps(report, indent=2))
    return report


def interactive_review(
    candidates_path: str = CANDIDATES_PATH,
    reviewer: Optional[str] = None,
    limit: Optional[int] = None,
) -> int:
    """
    Interactive CLI for human labeling.

    Commands during review:
      <intent_name>  accept a taxonomy label
      s / suggested  accept the heuristic suggestion
      skip           leave pending
      quit           save and exit
    """
    candidates = _load_jsonl(candidates_path)
    if not candidates:
        print(f"No candidates found at {candidates_path}. Run without --review first.")
        return 0

    intents = _valid_intents()
    reviewer = reviewer or input("Reviewer name/initials: ").strip() or "anonymous"
    reviewed_now = 0
    pending = [c for c in candidates if not c.get("is_human_reviewed")]
    if limit is not None:
        pending = pending[:limit]

    print("\n=== HUMAN LABELING CLI ===")
    print(f"Pending: {len(pending)} | Valid intents:")
    for i, name in enumerate(intents, 1):
        print(f"  {i}. {name}")
    print("Commands: <intent>, s/suggested, skip, quit\n")

    for cand in pending:
        print("-" * 72)
        print(f"ID: {cand.get('candidate_id')} | conv: {cand.get('conversation_id')}")
        print(f"Message: {cand.get('customer_message')}")
        print(f"Suggested (heuristic, may be WRONG): {cand.get('suggested_intent')}")
        raw = input("Your label> ").strip()

        if raw.lower() in {"quit", "q", "exit"}:
            break
        if raw.lower() in {"skip", ""}:
            continue

        if raw.lower() in {"s", "suggested", "accept"}:
            label = cand.get("suggested_intent")
        elif raw.isdigit() and 1 <= int(raw) <= len(intents):
            label = intents[int(raw) - 1]
        else:
            label = raw

        if label not in intents:
            print(f"Invalid intent '{label}'. Leaving pending.")
            continue

        esc = input("Escalation decision [AUTO_HANDLE/ESCALATE/skip]: ").strip()
        notes = input("Notes (optional): ").strip()

        cand["human_verified_intent"] = label
        cand["is_human_reviewed"] = True
        cand["review_status"] = "HUMAN_REVIEWED"
        cand["reviewer"] = reviewer
        cand["review_date"] = date.today().isoformat()
        if esc.upper() in {"AUTO_HANDLE", "ESCALATE"}:
            cand["escalation_decision"] = esc.upper()
        cand["notes"] = notes or None
        reviewed_now += 1
        print(f"Saved as {label}.")

    _write_jsonl(candidates_path, candidates)
    print(f"\nReviewed this session: {reviewed_now}")
    print("Run `python -m src.data.label --promote` to update golden_set.jsonl")
    return reviewed_now


def promote_reviewed_candidates(
    candidates_path: str = CANDIDATES_PATH,
    golden_path: str = GOLDEN_SET_PATH,
) -> int:
    """
    Promote only genuinely human-reviewed candidates into golden_set.jsonl.

    Unreviewed provisional seeds remain marked is_human_reviewed=false.
    Heuristic suggestions are NEVER treated as human labels.
    """
    candidates = _load_jsonl(candidates_path)
    if not candidates:
        print(f"No candidates at {candidates_path}")
        return 0

    reviewed = [c for c in candidates if c.get("is_human_reviewed") is True]
    provisional = [c for c in candidates if not c.get("is_human_reviewed")]

    golden_rows: List[Dict[str, Any]] = []
    for c in reviewed:
        human_intent = c.get("human_verified_intent")
        if not human_intent:
            continue
        golden_rows.append(
            {
                "id": c["candidate_id"],
                "conversation_id": c["conversation_id"],
                "customer_message": c["customer_message"],
                "suggested_intent": c.get("suggested_intent"),
                "human_verified_intent": human_intent,
                "intent": human_intent,  # gold intent for evaluation
                "is_human_reviewed": True,
                "review_status": "HUMAN_REVIEWED",
                "label_source": "HUMAN",
                "reviewer": c.get("reviewer"),
                "review_date": c.get("review_date"),
                "escalation_decision": c.get("escalation_decision"),
                "notes": c.get("notes"),
            }
        )

    # Keep provisional seeds so the file still documents candidates, but
    # clearly mark them as non-human.
    for c in provisional:
        golden_rows.append(
            {
                "id": c["candidate_id"],
                "conversation_id": c["conversation_id"],
                "customer_message": c["customer_message"],
                "suggested_intent": c.get("suggested_intent"),
                "intent": c.get("suggested_intent"),
                "human_verified_intent": None,
                "is_human_reviewed": False,
                "review_status": "PROVISIONAL_SEED_PENDING_HUMAN_AUDIT",
                "label_source": "HEURISTIC_SUGGESTION_NOT_HUMAN",
            }
        )

    _write_jsonl(golden_path, golden_rows)
    n_human = sum(1 for g in golden_rows if g.get("is_human_reviewed"))
    print(f"Promoted {n_human} human-reviewed examples to {golden_path}")
    if n_human < 150:
        print(
            f"BLOCKED for gold evaluation: {n_human}/150 human-reviewed. "
            "Continue labeling."
        )
    else:
        print("Golden set meets the 150+ human-reviewed minimum.")
    return n_human


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Human golden-set labeling workflow"
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Interactive human review of pending candidates",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote human-reviewed candidates into golden_set.jsonl",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show review progress",
    )
    parser.add_argument("--reviewer", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max candidates to review in this session",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=200,
        help="Number of candidates to generate (default 200)",
    )
    args = parser.parse_args(argv)

    if args.status:
        status_report()
    elif args.review:
        interactive_review(reviewer=args.reviewer, limit=args.limit)
    elif args.promote:
        promote_reviewed_candidates()
    else:
        generate_golden_candidates(target_count=args.target_count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
