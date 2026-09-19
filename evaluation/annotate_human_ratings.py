"""
Interactive Human Rating CLI for LLM Judge Agreement Evaluation.

Usage:
    python -m evaluation.annotate_human_ratings

This tool:
1. Samples 30 AUTO_HANDLE pipeline outputs from results/pipeline_test_outputs.jsonl
2. Displays each generated reply with its context
3. Asks you to rate it on the SAME 5-dimension rubric the LLM judge uses
4. Saves progress incrementally
5. Produces data/golden/human_ratings.json with the required schema

RUBRIC (matches LLM judge exactly):
  Each dimension scored 0–2 (total 0–10):
  - Correctness (0-2): Is the reply factually correct and not misleading?
  - Groundedness (0-2): Is the reply based on evidence, not invented?
  - Relevance (0-2): Does the reply address what the customer asked?
  - Completeness (0-2): Does the reply cover the issue sufficiently?
  - Professionalism (0-2): Is the tone appropriate / brand-suitable?

Rules:
  - YOU are the only rater. Do not invent numbers.
  - Progress is saved after every rating.
  - You need at least 30 ratings to unlock agreement statistics.
"""
import os
import json
import sys
import textwrap
from datetime import date
from typing import Dict, Any, List, Optional

import numpy as np

PIPELINE_OUTPUTS = "results/pipeline_test_outputs.jsonl"
SHARED_EXAMPLES = "data/golden/judge_agreement_examples.jsonl"
HUMAN_RATINGS_FILE = "data/golden/human_ratings.json"
PROGRESS_PATH = "data/golden/.rating_progress.json"

MIN_RATINGS = 30
SAMPLE_SEED = 42
SAMPLE_N = 35  # slight buffer over the 30 minimum


RUBRIC = """
RATING RUBRIC (same as LLM Judge — score each dimension 0, 1, or 2):
  Correctness (0-2): Is the reply factually accurate and free from false claims?
    0 = Contains factual errors or harmful false information
    1 = Mostly correct but minor inaccuracy
    2 = Fully correct, no false claims

  Groundedness (0-2): Is the reply grounded in historical evidence, not invented?
    0 = Invents policies, prices, guarantees with no evidence basis
    1 = Mostly grounded but makes one unverified assertion
    2 = Fully grounded in retrieved evidence or safe DM redirect

  Relevance (0-2): Does the reply actually address the customer's issue?
    0 = Off-topic or completely unrelated
    1 = Partially addresses the question
    2 = Directly and clearly addresses the customer's issue

  Completeness (0-2): Is the reply complete enough to be useful?
    0 = Too short or missing key next steps
    1 = Has some useful content but incomplete
    2 = Provides sufficient information and clear next steps

  Professionalism (0-2): Is the tone professional and brand-appropriate?
    0 = Rude, dismissive, or unprofessional
    1 = Acceptable but could be warmer/clearer
    2 = Professional, polite, brand-appropriate tone
"""


def wrap(text: str, width: int = 80) -> str:
    return textwrap.fill(str(text), width=width)


def load_pipeline_outputs() -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(PIPELINE_OUTPUTS):
        print(f"ERROR: {PIPELINE_OUTPUTS} not found.")
        print("Run `python -m evaluation.pipeline_eval` first.")
        sys.exit(1)
    with open(PIPELINE_OUTPUTS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_examples(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """
    Sample n examples that have actual generated replies (non-null, non-empty).
    Prefer AUTO_HANDLE cases since they have actual generated text.
    Pad with ESCALATE cases if needed.
    """
    rng = np.random.default_rng(seed)

    def get_reply(row: Dict) -> Optional[str]:
        gr = row.get("generated_reply", "")
        if isinstance(gr, str):
            try:
                gr = json.loads(gr)
            except Exception:
                return gr if gr.strip() else None
        if isinstance(gr, dict):
            return gr.get("reply") or gr.get("generated_reply")
        return None

    usable = []
    for row in rows:
        reply = get_reply(row)
        if reply and len(reply.strip()) > 5:
            usable.append((row, reply))

    if not usable:
        print("ERROR: No usable generated replies found in pipeline outputs.")
        sys.exit(1)

    n_sample = min(n, len(usable))
    idxs = rng.choice(len(usable), size=n_sample, replace=False)

    examples = []
    for i, idx in enumerate(sorted(idxs), start=1):
        row, reply = usable[int(idx)]
        ev = row.get("retrieval", {})
        if isinstance(ev, dict):
            ev_cases = ev.get("evidence_cases", [])[:2]
        else:
            ev_cases = []
        examples.append({
            "example_id": f"agr_{i:03d}",
            "conversation_id": row.get("evaluation", {}).get("conversation_id") if isinstance(row.get("evaluation"), dict) else None,
            "customer_message": row.get("customer_message", ""),
            "predicted_intent": row.get("intent", {}).get("name") if isinstance(row.get("intent"), dict) else None,
            "intent_confidence": row.get("intent", {}).get("confidence") if isinstance(row.get("intent"), dict) else None,
            "escalation_decision": row.get("escalation", {}).get("decision") if isinstance(row.get("escalation"), dict) else None,
            "escalation_reason": row.get("escalation", {}).get("reason_code") if isinstance(row.get("escalation"), dict) else None,
            "generated_reply": reply,
            "evidence_snippets": [
                ev_c.get("customer_issue", ev_c.get("customer_message", ""))[:100]
                for ev_c in ev_cases if isinstance(ev_c, dict)
            ],
            "source": "REAL_PIPELINE_TEST_OUTPUT",
        })

    return examples


def load_shared_examples() -> List[Dict[str, Any]]:
    if not os.path.exists(SHARED_EXAMPLES):
        return []
    examples = []
    with open(SHARED_EXAMPLES, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_shared_examples(examples: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(SHARED_EXAMPLES), exist_ok=True)
    with open(SHARED_EXAMPLES, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_progress() -> Dict[str, Any]:
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"rated_ids": {}, "reviewer": None}


def save_progress(progress: Dict[str, Any]):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_human_ratings() -> Dict[str, Any]:
    if os.path.exists(HUMAN_RATINGS_FILE):
        with open(HUMAN_RATINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"rated_by": None, "rating_date": None, "rubric": "0–10 total: correctness+groundedness+relevance+completeness+professionalism (each 0–2)", "examples": []}


def save_human_ratings(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(HUMAN_RATINGS_FILE), exist_ok=True)
    with open(HUMAN_RATINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def rate_dimension(name: str, description: str) -> int:
    while True:
        try:
            raw = input(f"  {name} (0–2): ").strip()
        except (EOFError, KeyboardInterrupt):
            return -1
        if raw == "q" or raw == "quit":
            return -1
        try:
            val = int(raw)
            if 0 <= val <= 2:
                return val
            print("  Enter 0, 1, or 2.")
        except ValueError:
            print("  Enter 0, 1, or 2.")


def print_example(idx: int, total: int, ex: Dict[str, Any]):
    print()
    print("=" * 70)
    print(f"  Reply {idx} / {total}   [Example ID: {ex['example_id']}]")
    print("=" * 70)
    print()
    print("CUSTOMER MESSAGE:")
    print(wrap(f"  \"{ex.get('customer_message', 'N/A')}\""))
    print()
    print(f"PREDICTED INTENT : {ex.get('predicted_intent', 'N/A')}  (confidence: {ex.get('intent_confidence', 'N/A')})")
    print(f"ESCALATION       : {ex.get('escalation_decision', 'N/A')}  (reason: {ex.get('escalation_reason') or 'none'})")
    print()

    snippets = ex.get("evidence_snippets", [])
    if snippets:
        print("RETRIEVED EVIDENCE SNIPPETS:")
        for i, s in enumerate(snippets, 1):
            print(wrap(f"  [{i}] {s}"))
        print()

    print("GENERATED REPLY:")
    print(wrap(f"  {ex.get('generated_reply', 'N/A')}"))
    print()
    print("-" * 70)
    print(RUBRIC)
    print("-" * 70)
    print()


def print_status(progress: Dict, examples: List):
    rated = progress.get("rated_ids", {})
    remaining = len(examples) - len(rated)
    print()
    print("=" * 70)
    print("  RATING PROGRESS")
    print("=" * 70)
    print(f"  Rated so far   : {len(rated)}")
    print(f"  Remaining      : {remaining}")
    print(f"  Required min   : {MIN_RATINGS}")
    if len(rated) >= MIN_RATINGS:
        print("  [OK] MINIMUM REACHED — run `python -m evaluation.human_judge_agreement` to compute agreement.")
    else:
        print(f"  [WAIT] Need {MIN_RATINGS - len(rated)} more ratings.")
    print("=" * 70)
    print()


def main():
    print()
    print("=" * 70)
    print("  HIVER — HUMAN REPLY QUALITY RATING TOOL")
    print("=" * 70)
    print()
    print("  You will rate each generated reply on a 5-dimension rubric (0–10 total).")
    print("  These ratings will be compared to the LLM Judge to compute agreement.")
    print()
    print("  IMPORTANT: Each dimension is scored 0, 1, or 2.")
    print("  Total = correctness + groundedness + relevance + completeness + professionalism")
    print()
    print("  Enter 'q' at any time to quit and save progress.")
    print()

    # Load or create shared examples
    examples = load_shared_examples()
    if not examples:
        print("  Sampling examples from pipeline outputs...")
        rows = load_pipeline_outputs()
        examples = sample_examples(rows, SAMPLE_N, SAMPLE_SEED)
        save_shared_examples(examples)
        print(f"  Sampled {len(examples)} examples. Saved to {SHARED_EXAMPLES}")
        print()

    progress = load_progress()
    rated_ids = progress.get("rated_ids", {})

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
    print_status(progress, examples)

    human_ratings_data = load_human_ratings()
    human_ratings_data["rated_by"] = reviewer
    # Build a dict of existing rated examples keyed by example_id
    existing = {ex["example_id"]: ex for ex in human_ratings_data.get("examples", [])}

    remaining = [ex for ex in examples if ex["example_id"] not in rated_ids]

    if not remaining:
        print("  All examples already rated!")
        print_status(progress, examples)
        return

    print(f"  Starting ratings — {len(remaining)} remaining...")
    print("  (Enter 'q' to quit and save at any time)")

    completed_this_session = 0

    for ex in remaining:
        eid = ex["example_id"]
        total = len(examples)
        position = [e["example_id"] for e in examples].index(eid) + 1

        print_example(position, total, ex)

        scores = {}
        dimensions = [
            ("correctness", "Is the reply factually correct and free from false claims?"),
            ("groundedness", "Is the reply grounded in retrieved evidence, not invented?"),
            ("relevance", "Does the reply address the customer's actual issue?"),
            ("completeness", "Is the reply complete enough to be useful?"),
            ("professionalism", "Is the tone professional and brand-appropriate?"),
        ]

        quit_requested = False
        for dim_name, dim_desc in dimensions:
            score = rate_dimension(dim_name, dim_desc)
            if score == -1:
                quit_requested = True
                break
            scores[dim_name] = score

        if quit_requested:
            print("\n  Saving progress and exiting...")
            break

        total_score = sum(scores.values())
        print(f"\n  → Total: {total_score}/10  ({scores})")

        try:
            notes = input("  Optional notes (Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""

        # Save this rating
        rated_record = {
            "example_id": eid,
            "customer_message": ex.get("customer_message"),
            "generated_reply": ex.get("generated_reply"),
            "predicted_intent": ex.get("predicted_intent"),
            "escalation_decision": ex.get("escalation_decision"),
            "reviewer": reviewer,
            "review_date": date.today().isoformat(),
            "correctness": scores.get("correctness"),
            "groundedness": scores.get("groundedness"),
            "relevance": scores.get("relevance"),
            "completeness": scores.get("completeness"),
            "professionalism": scores.get("professionalism"),
            "human_total_score": total_score,
            "llm_total_score": None,  # to be filled by agreement script
            "notes": notes,
        }
        existing[eid] = rated_record
        rated_ids[eid] = total_score

        human_ratings_data["examples"] = list(existing.values())
        human_ratings_data["rating_date"] = date.today().isoformat()
        save_human_ratings(human_ratings_data)

        progress["rated_ids"] = rated_ids
        save_progress(progress)
        completed_this_session += 1

        reviewed_count = len(rated_ids)
        print(f"  Saved. Total rated: {reviewed_count}")

    print()
    print(f"  Session complete. Rated {completed_this_session} replies this session.")
    print_status(progress, examples)

    if len(rated_ids) >= MIN_RATINGS:
        print("  [OK] You now have enough ratings to compute Human/LLM agreement.")
        print("  Run:")
        print("    python -m evaluation.human_judge_agreement")
        print()


if __name__ == "__main__":
    main()
