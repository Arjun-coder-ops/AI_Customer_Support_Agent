import os
import json
import logging
from typing import Dict, Any, List

from src.generation.schemas import SupportResponseSchema
from src.generation.prompts import GROUNDED_REPLY_SYSTEM_PROMPT, GROUNDED_REPLY_USER_PROMPT

logger = logging.getLogger(__name__)

class GroundedReplyGenerator:
    """
    Grounded Reply Generator using LLM (Gemini / OpenAI API) or deterministic Mock Generator.
    """
    def __init__(self, brand_name: str = "AmazonHelp", use_mock: bool = None):
        self.brand_name = brand_name
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if use_mock is not None:
            self.use_mock = use_mock
        else:
            mock_env = os.getenv("MOCK_LLM", "false").lower()
            self.use_mock = mock_env in ("true", "1", "yes") or (not self.gemini_key and not self.openai_key)

        if self.use_mock:
            logger.info("GroundedReplyGenerator running in MOCK mode.")
        else:
            logger.info("GroundedReplyGenerator initialized with active LLM Provider.")

    def generate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_cases: List[Dict[str, Any]],
        should_escalate: bool = False,
        escalation_reason_code: str = None,
    ) -> SupportResponseSchema:
        """
        Generate grounded support reply using LLM or Mock Engine.
        """
        if self.use_mock:
            return self._generate_mock(
                customer_message, predicted_intent, intent_confidence, evidence_cases, should_escalate, escalation_reason_code
            )

        try:
            return self._generate_llm(
                customer_message, predicted_intent, intent_confidence, evidence_cases, should_escalate, escalation_reason_code
            )
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}. Falling back to Mock Generator.")
            return self._generate_mock(
                customer_message, predicted_intent, intent_confidence, evidence_cases, should_escalate, escalation_reason_code
            )

    def _generate_mock(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_cases: List[Dict[str, Any]],
        should_escalate: bool,
        escalation_reason_code: str,
    ) -> SupportResponseSchema:
        evidence_ids = [c["case_id"] for c in evidence_cases if "case_id" in c]

        if should_escalate:
            reply = f"We have escalated your inquiry regarding '{predicted_intent.replace('_', ' ')}' to a customer specialist for manual review. (Reason: {escalation_reason_code})"
            return SupportResponseSchema(
                reply=reply,
                confidence=round(intent_confidence, 2),
                evidence_ids=evidence_ids,
                should_escalate=True,
                escalation_reason=escalation_reason_code or "PRE_DECIDED_ESCALATION",
            )

        if evidence_cases:
            hist_reply = evidence_cases[0].get("historical_response", "")
            reply = f"Hello! Regarding your inquiry: {hist_reply} If you need further assistance, please DM us your order details."
        else:
            reply = "Thank you for reaching out to customer support. Please provide your order ID so we can assist you promptly."

        return SupportResponseSchema(
            reply=reply,
            confidence=round(intent_confidence, 2),
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
        escalation_reason_code: str,
    ) -> SupportResponseSchema:
        # Construct evidence text
        evidence_lines = []
        for idx, c in enumerate(evidence_cases):
            evidence_lines.append(
                f"Case #{c.get('case_id')}: Customer: '{c.get('customer_issue')}' -> Support: '{c.get('historical_response')}' (Sim: {c.get('similarity')})"
            )
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No relevant historical evidence found."

        system_prompt = GROUNDED_REPLY_SYSTEM_PROMPT.format(brand_name=self.brand_name)
        user_prompt = GROUNDED_REPLY_USER_PROMPT.format(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            evidence_text=evidence_text,
            escalation_signal="ESCALATE" if should_escalate else "AUTO_HANDLE",
            escalation_reason_code=escalation_reason_code or "NONE",
        )

        if self.gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            raw_text = response.text
        elif self.openai_key:
            import openai
            client = openai.OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            raw_text = response.choices[0].message.content
        else:
            raise ValueError("No valid LLM API Key found.")

        # Clean JSON markdown if wrapped in ```json
        clean_json_str = raw_text.strip()
        if clean_json_str.startswith("```"):
            lines = clean_json_str.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_json_str = "\n".join(lines).strip()

        data = json.loads(clean_json_str)
        return SupportResponseSchema(**data)
