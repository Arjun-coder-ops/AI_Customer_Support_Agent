import os
import json
import logging
from typing import Dict, Any, List, Optional

from src.generation.schemas import SupportResponseSchema
from src.generation.prompts import GROUNDED_REPLY_SYSTEM_PROMPT, GROUNDED_REPLY_USER_PROMPT

logger = logging.getLogger(__name__)


class GroundedReplyGenerator:
    """
    Grounded reply generator.

    Supported modes:
    - MOCK_LLM=true  -> deterministic mock generator (offline / evaluation)
    - MOCK_LLM=false + GEMINI_API_KEY -> Gemini generation

    OpenAI is intentionally not supported.
    """

    def __init__(self, brand_name: str = "AmazonHelp", use_mock: Optional[bool] = None):
        self.brand_name = brand_name
        self.gemini_key = os.getenv("GEMINI_API_KEY")

        if use_mock is not None:
            self.use_mock = use_mock
        else:
            mock_env = os.getenv("MOCK_LLM", "false").lower()
            self.use_mock = mock_env in ("true", "1", "yes") or not self.gemini_key

        if self.use_mock:
            logger.info("GroundedReplyGenerator running in MOCK mode.")
        else:
            logger.info("GroundedReplyGenerator initialized with Gemini.")

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_cases: List[Dict[str, Any]],
        should_escalate: bool = False,
        escalation_reason_code: Optional[str] = None,
    ) -> SupportResponseSchema:
        """Generate a grounded support reply using Gemini or the mock engine."""
        if self.use_mock:
            return self._generate_mock(
                customer_message,
                predicted_intent,
                intent_confidence,
                evidence_cases,
                should_escalate,
                escalation_reason_code,
            )

        try:
            return self._generate_llm(
                customer_message,
                predicted_intent,
                intent_confidence,
                evidence_cases,
                should_escalate,
                escalation_reason_code,
            )
        except Exception as e:
            logger.error("Gemini generation failed: %s. Falling back to Mock Generator.", e)
            return self._generate_mock(
                customer_message,
                predicted_intent,
                intent_confidence,
                evidence_cases,
                should_escalate,
                escalation_reason_code,
            )

    def _generate_mock(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_cases: List[Dict[str, Any]],
        should_escalate: bool,
        escalation_reason_code: Optional[str],
    ) -> SupportResponseSchema:
        """
        Deterministic mock generator for offline testing/evaluation.

        Behaviour is intentionally conservative:
        - Escalation replies cite the policy reason code.
        - Auto-handle replies paraphrase historical evidence only.
        - No invented order IDs, refund amounts, or action claims.
        """
        evidence_ids = [c["case_id"] for c in evidence_cases if "case_id" in c]
        intent_label = predicted_intent.replace("_", " ")

        if should_escalate:
            reply = (
                f"[MOCK] We have escalated your inquiry regarding '{intent_label}' "
                f"to a customer specialist for manual review. "
                f"(Reason: {escalation_reason_code})"
            )
            return SupportResponseSchema(
                reply=reply,
                confidence=round(float(intent_confidence), 2),
                evidence_ids=evidence_ids,
                should_escalate=True,
                escalation_reason=escalation_reason_code or "PRE_DECIDED_ESCALATION",
            )

        if evidence_cases:
            hist_reply = evidence_cases[0].get("historical_response", "").strip()
            # Ground on historical text; do not invent order/account details.
            reply = (
                f"[MOCK] Thanks for contacting us about your {intent_label} request. "
                f"Based on similar past cases: {hist_reply} "
                f"Please share any order details via DM if needed — "
                f"we cannot access account-specific information here."
            )
        else:
            reply = (
                "[MOCK] Thank you for reaching out. We do not have enough "
                "historical evidence to resolve this safely in automated mode. "
                "Please provide additional details or wait for a specialist."
            )

        return SupportResponseSchema(
            reply=reply,
            confidence=round(float(intent_confidence), 2),
            evidence_ids=evidence_ids,
            should_escalate=False,
            escalation_reason=None,
        )

    def _generate_llm(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_cases: List[Dict[str, Any]],
        should_escalate: bool,
        escalation_reason_code: Optional[str],
    ) -> SupportResponseSchema:
        if not self.gemini_key:
            raise ValueError(
                "GEMINI_API_KEY is required when MOCK_LLM is false. "
                "Set MOCK_LLM=true for offline evaluation."
            )

        evidence_lines = []
        for c in evidence_cases:
            evidence_lines.append(
                f"Case #{c.get('case_id')}: Customer: '{c.get('customer_issue')}' "
                f"-> Support: '{c.get('historical_response')}' "
                f"(Sim: {c.get('similarity')})"
            )
        evidence_text = (
            "\n".join(evidence_lines)
            if evidence_lines
            else "No relevant historical evidence found."
        )

        system_prompt = GROUNDED_REPLY_SYSTEM_PROMPT.format(brand_name=self.brand_name)
        user_prompt = GROUNDED_REPLY_USER_PROMPT.format(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            evidence_text=evidence_text,
            escalation_signal="ESCALATE" if should_escalate else "AUTO_HANDLE",
            escalation_reason_code=escalation_reason_code or "NONE",
        )

        import google.generativeai as genai

        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
        raw_text = response.text

        clean_json_str = raw_text.strip()
        if clean_json_str.startswith("```"):
            lines = clean_json_str.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_json_str = "\n".join(lines).strip()

        data = json.loads(clean_json_str)

        # Enforce escalation policy decision over model disagreement.
        if should_escalate:
            data["should_escalate"] = True
            data["escalation_reason"] = (
                escalation_reason_code or data.get("escalation_reason")
            )

        return SupportResponseSchema(**data)
