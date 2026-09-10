import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Sensitive / high-risk key terms requiring human intervention
SENSITIVE_KEYWORDS = [
    "lawyer", "attorney", "legal action", "sue", "court",
    "stolen", "fraud", "police", "threat", "scam",
    "compensation", "bodily harm", "hacked", "identity theft"
]

ACCOUNT_ACTION_KEYWORDS = [
    "change address", "update credit card", "ssn", "social security",
    "close account", "delete account", "wire transfer", "bank routing"
]

class EscalationPolicyEngine:
    """
    Multi-signal Escalation Policy Engine.
    Evaluates intent confidence, retrieval similarity, sensitive key terms, and account-level actions.
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
        Returns dict with:
        - decision: 'AUTO_HANDLE' | 'ESCALATE'
        - reason_code: string | None
        - reason: string | None
        """
        msg_lower = customer_message.lower().strip()

        # Check 1: Short / Empty message -> INSUFFICIENT_CONTEXT
        if len(msg_lower.split()) < 3:
            return {
                "decision": "ESCALATE",
                "reason_code": "INSUFFICIENT_CONTEXT",
                "reason": "Customer message is too short or lacks sufficient details for automated resolution.",
            }

        # Check 2: High-Risk / Sensitive keywords -> SENSITIVE_REQUEST
        for kw in SENSITIVE_KEYWORDS:
            if kw in msg_lower:
                return {
                    "decision": "ESCALATE",
                    "reason_code": "SENSITIVE_REQUEST",
                    "reason": f"Customer message contains sensitive or high-risk legal/fraud key term: '{kw}'.",
                }

        # Check 3: Account Specific Action -> ACCOUNT_SPECIFIC_ACTION_REQUIRED
        for kw in ACCOUNT_ACTION_KEYWORDS:
            if kw in msg_lower:
                return {
                    "decision": "ESCALATE",
                    "reason_code": "ACCOUNT_SPECIFIC_ACTION_REQUIRED",
                    "reason": f"Request requires sensitive account action or credential change: '{kw}'.",
                }

        # Check 4: Intent Classifier Confidence -> LOW_INTENT_CONFIDENCE
        if intent_confidence < self.min_intent_confidence:
            return {
                "decision": "ESCALATE",
                "reason_code": "LOW_INTENT_CONFIDENCE",
                "reason": f"Intent confidence ({intent_confidence:.2f}) is below safety threshold ({self.min_intent_confidence:.2f}).",
            }

        # Check 5: Historical Evidence Availability -> NO_RELEVANT_HISTORICAL_EVIDENCE
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
                "reason": f"Top historical evidence similarity ({top_similarity:.2f}) is below threshold ({self.min_retrieval_similarity:.2f}).",
            }

        # If all safety checks pass -> AUTO_HANDLE
        return {
            "decision": "AUTO_HANDLE",
            "reason_code": None,
            "reason": "Request satisfied all intent, retrieval similarity, and risk policy checks safely.",
        }
