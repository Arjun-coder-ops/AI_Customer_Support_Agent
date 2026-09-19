"""
Interactive Human Annotation CLI for Golden Set Labelling.

Usage:
    python -m src.data.annotate_golden

This tool guides you through each candidate from data/golden/labeling_candidates.jsonl,
asks you to assign the correct intent, and saves your progress after every answer.

Progress is saved incrementally — you can safely Ctrl+C and resume later.
Completed examples are promoted into data/golden/golden_set.jsonl.

Rules:
- YOU are the only person who can assign labels.
- This tool will NOT auto-label or suggest final labels silently.
- It shows a SUGGESTED intent (heuristic) but you choose the final label.
- Your choice is recorded as human_verified_intent with is_human_reviewed=true.
"""
import os
import json
import sys
import textwrap
from datetime import date
from typing import Dict, Any, List, Optional

CANDIDATES_PATH = "data/golden/labeling_candidates.jsonl"
GOLDEN_SET_PATH = "data/golden/golden_set.jsonl"
PROGRESS_PATH = "data/golden/.annotation_progress.json"

INTENT_LABELS = [
    "shipping_delay",
    "missing_item",
    "order_cancellation",
    "refund_return_request",
    "account_access_issue",
    "payment_billing_issue",
    "product_defect_damage",
    "general_inquiry_feedback",
]

INTENT_DESCRIPTIONS = {
    "shipping_delay": "Order delayed / tracking stuck / package hasn't arrived",
    "missing_item": "Parcel delivered but one or more items missing from box",
    "order_cancellation": "Request to cancel an order before shipment",
    "refund_return_request": "Return product or request money back / refund status",
    "account_access_issue": "Login, password reset, 2FA, account locked",
    "payment_billing_issue": "Double charge, wrong fee, billing dispute, payment failure",
    "product_defect_damage": "Received broken, cracked, or defective item",
    "general_inquiry_feedback": "General question, restock, unclear intent, ambiguous",
}

SKIP_LABEL = "skip"


def load_candidates() -> List[Dict[str, Any]]:
    candidates = []
    if not os.path.exists(CANDIDATES_PATH):
        print(f"ERROR: {CANDIDATES_PATH} not found. Run `python -m src.data.label` first.")
        sys.exit(1)
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


def load_progress() -> Dict[str, Any]:
    """Load previously annotated IDs so we can skip them."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"annotated_ids": {}, "reviewer": None}


def save_progress(progress: Dict[str, Any]):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_golden_set() -> Dict[str, Any]:
    """Load existing golden set keyed by candidate_id."""
    golden = {}
    if os.path.exists(GOLDEN_SET_PATH):
        with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    golden[rec.get("id", rec.get("candidate_id", ""))] = rec
    return golden


def write_golden_set(golden: Dict[str, Any]):
    os.makedirs(os.path.dirname(GOLDEN_SET_PATH), exist_ok=True)
    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        for rec in golden.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def promote_to_golden(
    candidate: Dict[str, Any],
    human_intent: str,
    reviewer: str,
    notes: str,
    golden: Dict[str, Any],
):
    """Write a human-reviewed record into the golden set dict."""
    cid = candidate["candidate_id"]
    golden[cid] = {
        "id": cid,
        "conversation_id": candidate.get("conversation_id"),
        "customer_message": candidate.get("customer_message"),
        "suggested_intent": candidate.get("suggested_intent"),
        "human_verified_intent": human_intent,
        "is_human_reviewed": True,
        "reviewer": reviewer,
        "review_date": date.today().isoformat(),
        "notes": notes,
    }


def wrap(text: str, width: int = 80) -> str:
    return textwrap.fill(text, width=width)


def print_candidate(idx: int, total: int, candidate: Dict[str, Any]):
    """Print the candidate for human review."""
    print()
    print("=" * 70)
    print(f"  Example {idx} / {total}")
    print("=" * 70)
    print()
    print("Customer Message:")
    print()
    msg = candidate.get("customer_message", "")
    print(wrap(f'  "{msg}"'))
    print()
    print(f"Heuristic Suggestion: [{candidate.get('suggested_intent', 'unknown')}]")
    print("  (This is a KEYWORD GUESS — it may be WRONG. You choose the final label.)")
    print()
    print("Choose the correct intent:")
    print()
    for i, label in enumerate(INTENT_LABELS, 1):
        desc = INTENT_DESCRIPTIONS.get(label, "")
        print(f"  {i}. {label}")
        print(f"     {desc}")
    print()
    print("  9. skip  (unclear / ambiguous — will be excluded from golden set)")
    print()


def get_user_choice() -> Optional[str]:
    """Read and validate the user's choice."""
    valid = {str(i): INTENT_LABELS[i - 1] for i in range(1, 9)}
    valid["9"] = SKIP_LABEL
    # Also accept intent names directly
    for label in INTENT_LABELS:
        valid[label] = label
    valid["skip"] = SKIP_LABEL
    valid["s"] = SKIP_LABEL
    valid["q"] = "QUIT"
    valid["quit"] = "QUIT"
    valid["exit"] = "QUIT"

    while True:
        try:
            raw = input("  Your label (1–9, or intent name, or 'q' to quit and save): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\nInterrupted — saving progress...")
            return "QUIT"

        if raw in valid:
            return valid[raw]
        print(f"  Invalid input '{raw}'. Enter a number 1–9 or an intent name.")


def get_optional_notes() -> str:
    try:
        raw = input("  Optional notes (press Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    return raw


def print_status(progress: Dict[str, Any], candidates: List):
    annotated = progress.get("annotated_ids", {})
    reviewed = [v for v in annotated.values() if v != SKIP_LABEL]
    skipped = [v for v in annotated.values() if v == SKIP_LABEL]
    remaining = len(candidates) - len(annotated)
    print()
    print("=" * 70)
    print("  ANNOTATION PROGRESS")
    print("=" * 70)
    print(f"  Reviewed and labelled : {len(reviewed)}")
    print(f"  Skipped (unclear)     : {len(skipped)}")
    print(f"  Remaining             : {remaining}")
    print(f"  Required minimum      : 150 reviewed (not skipped)")
    print()
    if len(reviewed) >= 150:
        print("  [OK] GOLDEN SET READY — you have 150+ labelled examples.")
        print("     Run `python -m evaluation.run_all` to unlock gold metrics.")
    else:
        print(f"  [WAIT] Need {150 - len(reviewed)} more labels to reach minimum.")
    print("=" * 70)
    print()


def main():
    print()
    print("=" * 70)
    print("  HIVER CUSTOMER SUPPORT — INTENT ANNOTATION TOOL")
    print("=" * 70)
    print()
    print("  You will label each customer message with the correct intent.")
    print("  Your labels will be saved as the human-gold evaluation standard.")
    print()
    print("  Rules:")
    print("  - Enter a number (1–8) or intent name to assign a label.")
    print("  - Enter 9 or 'skip' if the message is too unclear.")
    print("  - Enter 'q' to quit and save progress at any time.")
    print("  - Progress is saved after EVERY response.")
    print()

    candidates = load_candidates()
    progress = load_progress()
    golden = load_golden_set()
    annotated_ids = progress.get("annotated_ids", {})

    # Get or confirm reviewer name
    reviewer = progress.get("reviewer")
    if not reviewer:
        try:
            reviewer = input("  Your name (for record-keeping): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Cancelled.")
            sys.exit(0)
        if not reviewer:
            reviewer = "anonymous"
        progress["reviewer"] = reviewer
        save_progress(progress)

    print(f"\n  Reviewer: {reviewer}")
    print_status(progress, candidates)

    # Filter out already-annotated candidates
    remaining_candidates = [
        c for c in candidates
        if c["candidate_id"] not in annotated_ids
    ]

    if not remaining_candidates:
        print("  All candidates have already been annotated!")
        print_status(progress, candidates)
        return

    print(f"  Starting annotation — {len(remaining_candidates)} remaining...")
    print("  (Ctrl+C or 'q' to quit and save at any time)")
    print()

    total = len(candidates)
    completed_this_session = 0

    for candidate in remaining_candidates:
        cid = candidate["candidate_id"]
        position = list(c["candidate_id"] for c in candidates).index(cid) + 1

        print_candidate(position, total, candidate)

        choice = get_user_choice()

        if choice == "QUIT":
            print("\n  Saving progress and exiting...")
            break

        if choice == SKIP_LABEL:
            print(f"  → Skipped (will not count toward golden set)")
            annotated_ids[cid] = SKIP_LABEL
            notes = ""
        else:
            notes = get_optional_notes()
            promote_to_golden(candidate, choice, reviewer, notes, golden)
            annotated_ids[cid] = choice
            reviewed_count = len([v for v in annotated_ids.values() if v != SKIP_LABEL])
            print(f"  → Saved: {choice}  (Total reviewed: {reviewed_count})")

        # Save after every response
        progress["annotated_ids"] = annotated_ids
        save_progress(progress)
        write_golden_set(golden)
        completed_this_session += 1

    print()
    print(f"  Session complete. Annotated {completed_this_session} examples this session.")
    print_status(progress, candidates)


if __name__ == "__main__":
    main()
