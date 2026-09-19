import os
import json
import logging
from typing import Dict, Any, List

from src.data.clean import clean_text
from src.intent.classifier import FinalIntentClassifier
from src.retrieval.index import HistoricalCaseRetrievalIndex
from src.escalation.policy import EscalationPolicyEngine
from src.generation.generator import GroundedReplyGenerator
from src.generation.schemas import SupportResponseSchema

logger = logging.getLogger(__name__)

class SupportAgentPipeline:
    """
    End-to-End Autonomous AI Customer Support Agent Pipeline.
    """
    def __init__(self, brand_name: str = "AmazonHelp", use_mock: bool = None):
        self.brand_name = brand_name
        self.classifier = FinalIntentClassifier()
        self.retrieval_index = HistoricalCaseRetrievalIndex()
        self.escalation_engine = EscalationPolicyEngine()
        self.generator = GroundedReplyGenerator(brand_name=brand_name, use_mock=use_mock)
        self.is_trained = False

    def train_and_index(self, train_cases: List[Dict[str, Any]], resolved_cases: List[Dict[str, Any]]):
        """
        Train intent classifier and build retrieval index on training set only (zero leakage).
        """
        X_train = [c["customer_initial_message"] for c in train_cases if "customer_initial_message" in c]
        y_train = []
        for c in train_cases:
            # Map or assign heuristic intent for training
            msg = c.get("customer_initial_message", "").lower()
            intent = "general_inquiry_feedback"
            if any(k in msg for k in ["delay", "where is", "tracking", "late"]):
                intent = "shipping_delay"
            elif any(k in msg for k in ["missing", "incomplete"]):
                intent = "missing_item"
            elif any(k in msg for k in ["cancel", "stop"]):
                intent = "order_cancellation"
            elif any(k in msg for k in ["refund", "return"]):
                intent = "refund_return_request"
            elif any(k in msg for k in ["lock", "login", "password"]):
                intent = "account_access_issue"
            elif any(k in msg for k in ["charge", "paid", "double"]):
                intent = "payment_billing_issue"
            elif any(k in msg for k in ["damage", "defect", "broken"]):
                intent = "product_defect_damage"
            y_train.append(intent)

        self.classifier.fit(X_train, y_train)
        self.retrieval_index.build_index(resolved_cases)
        self.is_trained = True
        logger.info("SupportAgentPipeline training and retrieval indexing complete.")

    def process_message(self, customer_message: str, turn_count: int = 1) -> Dict[str, Any]:
        """
        Process incoming customer message through complete pipeline.
        Returns final structured response dictionary.
        """
        cleaned_msg = clean_text(customer_message)

        # 1. Intent Classification
        if self.is_trained:
            intent_res = self.classifier.predict_single(cleaned_msg)
            intent_name = intent_res["predicted_intent"]
            intent_confidence = intent_res["confidence"]
        else:
            intent_name = "general_inquiry_feedback"
            intent_confidence = 0.50

        # 2. Vector Retrieval
        evidence_cases = self.retrieval_index.search(cleaned_msg, top_k=3)

        # 3. Multi-Signal Escalation Policy
        escalation_eval = self.escalation_engine.evaluate(
            customer_message=cleaned_msg,
            predicted_intent=intent_name,
            intent_confidence=intent_confidence,
            evidence_cases=evidence_cases,
            turn_count=turn_count,
        )

        should_escalate = (escalation_eval["decision"] == "ESCALATE")
        escalation_reason_code = escalation_eval["reason_code"]

        # 4. Grounded Generation
        gen_response: SupportResponseSchema = self.generator.generate(
            customer_message=cleaned_msg,
            predicted_intent=intent_name,
            intent_confidence=intent_confidence,
            evidence_cases=evidence_cases,
            should_escalate=should_escalate,
            escalation_reason_code=escalation_reason_code,
        )

        return {
            "customer_message": customer_message,
            "intent": {
                "name": intent_name,
                "confidence": intent_confidence,
            },
            "retrieval": {
                "retrieved_count": len(evidence_cases),
                "top_similarity": evidence_cases[0]["similarity"] if evidence_cases else 0.0,
                "evidence_cases": evidence_cases,
            },
            "escalation": escalation_eval,
            "generated_reply": gen_response.model_dump(),
        }

if __name__ == "__main__":
    # Simple CLI Demo
    sample_train = [
        {"customer_initial_message": "Where is my delayed package order #123?", "brand": "AmazonHelp"},
        {"customer_initial_message": "Item missing from box", "brand": "AmazonHelp"},
        {"customer_initial_message": "Charged twice on card", "brand": "AmazonHelp"},
    ]
    sample_resolved = [
        {"case_id": "case_101", "customer_message": "Where is my delayed package order #123?", "historical_response": "We updated tracking.", "turn_count": 2},
        {"case_id": "case_201", "customer_message": "Item missing from box", "historical_response": "Replacement order issued.", "turn_count": 2},
    ]

    pipeline = SupportAgentPipeline(brand_name="AmazonHelp", use_mock=True)
    pipeline.train_and_index(sample_train, sample_resolved)
    result = pipeline.process_message("My order #123 is delayed, where is it?")
    print(json.dumps(result, indent=2))
