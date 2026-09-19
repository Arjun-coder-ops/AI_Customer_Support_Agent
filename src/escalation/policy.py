import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Sensitive / high-risk key terms requiring human intervention
SENSITIVE_KEYWORDS = [
    "lawyer", "attorney", "legal action", "sue", "court",
    "stolen", "fraud", "police", "threat", "scam",
    "compensation", "bodily harm", "hacked", "identity theft",
]

ACCOUNT_ACTION_KEYWORDS = [
    "change address", "update credit card", "ssn", "social security",
    "close account", "delete account", "wire transfer", "bank routing",
]


def _contains_keyword(text: str, keyword: str) -> bool:
    """Match multi-word phrases or whole-word tokens (avoids 'ssn' in 'grossness')."""
    pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
    return re.search(pattern, text) is not None


class EscalationPolicyEngine:
    """
    Multi-signal escalation policy (interview-explainable).

    Evaluation order (first match wins):
      1. INSUFFICIENT_CONTEXT          — message too short to resolve safely
      2. SENSITIVE_REQUEST             — legal/fraud/high-risk language
      3. ACCOUNT_SPECIFIC_ACTION_REQUIRED — account mutation / credentials
      4. LOW_INTENT_CONFIDENCE         — classifier below confidence threshold
      5. NO_RELEVANT_HISTORICAL_EVIDENCE — missing or weak retrieval evidence
      6. AUTO_HANDLE                   — only if ALL checks pass

    Design principle: false auto-handling is more costly than over-escalation.
    Thresholds are safety defaults, not tuned to maximize auto-handle rate.
    """

    def __init__(
        self,
        min_intent_confidence: float = 0.70,
        min_retrieval_similarity: float = 0.65,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_cases: List[Dict[str, Any]],
        turn_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate multi-signal risk criteria.

        Returns:
          decision: 'AUTO_HANDLE' | 'ESCALATE'
          reason_code: string | None
          reason: string | None
        """
        msg_lower = customer_message.lower().strip()

        # 1. Short / empty message
        if len(msg_lower.split()) < 3:
            return {
                "decision": "ESCALATE",
                "reason_code": "INSUFFICIENT_CONTEXT",
                "reason": (
                    "Customer message is too short or lacks sufficient details "
                    "for automated resolution."
                ),
            }

        # 2. High-risk / sensitive keywords
        for kw in SENSITIVE_KEYWORDS:
            if _contains_keyword(msg_lower, kw):
                return {
                    "decision": "ESCALATE",
                    "reason_code": "SENSITIVE_REQUEST",
                    "reason": (
                        "Customer message contains sensitive or high-risk "
                        f"legal/fraud key term: '{kw}'."
                    ),
                }

        # 3. Account-specific action
        for kw in ACCOUNT_ACTION_KEYWORDS:
            if _contains_keyword(msg_lower, kw):
                return {
                    "decision": "ESCALATE",
                    "reason_code": "ACCOUNT_SPECIFIC_ACTION_REQUIRED",
                    "reason": (
                        "Request requires sensitive account action or "
                        f"credential change: '{kw}'."
                    ),
                }

        # 4. Low intent confidence
        if intent_confidence < self.min_intent_confidence:
            return {
                "decision": "ESCALATE",
                "reason_code": "LOW_INTENT_CONFIDENCE",
                "reason": (
                    f"Intent confidence ({intent_confidence:.2f}) is below "
                    f"safety threshold ({self.min_intent_confidence:.2f})."
                ),
            }

        # 5. Insufficient historical evidence
        if not evidence_cases:
            return {
                "decision": "ESCALATE",
                "reason_code": "NO_RELEVANT_HISTORICAL_EVIDENCE",
                "reason": "No historical resolved cases found in knowledge base.",
            }

        top_similarity = evidence_cases[0].get("similarity", 0.0)
        if top_similarity < self.min_retrieval_similarity:
            return {
                "decision": "ESCALATE",
                "reason_code": "NO_RELEVANT_HISTORICAL_EVIDENCE",
                "reason": (
                    f"Top historical evidence similarity ({top_similarity:.2f}) "
                    f"is below threshold ({self.min_retrieval_similarity:.2f})."
                ),
            }

        # 6. All safety checks passed
        return {
            "decision": "AUTO_HANDLE",
            "reason_code": None,
            "reason": (
                "Request satisfied all intent, retrieval similarity, and "
                "risk policy checks safely."
            ),
        }
