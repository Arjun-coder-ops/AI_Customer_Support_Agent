import os
import json
import logging
import numpy as np
from typing import Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class JudgeRubricScore(BaseModel):
    correctness: int = Field(..., ge=0, le=2, description="Score 0-2 for factual correctness")
    groundedness: int = Field(..., ge=0, le=2, description="Score 0-2 for grounding in evidence without hallucination")
    relevance: int = Field(..., ge=0, le=2, description="Score 0-2 for directly answering customer query")
    completeness: int = Field(..., ge=0, le=2, description="Score 0-2 for complete issue coverage")
    professionalism: int = Field(..., ge=0, le=2, description="Score 0-2 for tone and brand appropriateness")
    total_score: int = Field(..., ge=0, le=10, description="Sum of dimension scores (0-10)")
    reasoning: str = Field(..., description="Brief explanation for assigned scores")

class LLMJudgeEvaluator:
    """
    LLM-as-a-Judge evaluator assessing generated support replies across 5 criteria.
    Falls back to structured heuristic judge when offline or MOCK_LLM=true.
    """
    def __init__(self, use_mock: bool = None):
        if use_mock is not None:
            self.use_mock = use_mock
        else:
            mock_env = os.getenv("MOCK_LLM", "false").lower()
            self.use_mock = mock_env in ("true", "1", "yes") or not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))

    def evaluate_reply(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str,
        predicted_intent: str,
    ) -> Dict[str, Any]:
        """
        Judge generated response on 0-10 multi-criteria scale.
        """
        if self.use_mock:
            return self._heuristic_judge(customer_message, retrieved_evidence, generated_reply, predicted_intent)

        try:
            return self._llm_judge(customer_message, retrieved_evidence, generated_reply, predicted_intent)
        except Exception as e:
            logger.error(f"LLM Judge API failed: {e}. Using heuristic judge fallback.")
            return self._heuristic_judge(customer_message, retrieved_evidence, generated_reply, predicted_intent)

    def _heuristic_judge(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str,
        predicted_intent: str,
    ) -> Dict[str, Any]:
        reply_lower = generated_reply.lower()

        # Factual correctness / validity
        corr = 2 if len(generated_reply) > 15 else 1
        if "error" in reply_lower or "unknown" in reply_lower:
            corr = 0

        # Groundedness (penalize hallucinated promises or missing evidence)
        ground = 2
        if not retrieved_evidence and not any(k in reply_lower for k in ["escalat", "dm", "support"]):
            ground = 0
        elif any(k in reply_lower for k in ["100% refund", "$500 gift card", "guarantee tomorrow"]):
            ground = 0  # Hallucinated policy!

        # Relevance
        rel = 2 if any(w in reply_lower for w in ["order", "delay", "return", "refund", "assist", "help", "escalat", "tracking"]) else 1
        if "irrelevant" in reply_lower:
            rel = 0

        # Completeness
        comp = 2 if len(generated_reply.split()) >= 10 else 1
        if len(generated_reply.split()) < 4:
            comp = 0

        # Professionalism
        prof = 2
        if any(w in reply_lower for w in ["stupid", "idiot", "shut up", "damn"]):
            prof = 0

        total = corr + ground + rel + comp + prof

        return {
            "correctness": corr,
            "groundedness": ground,
            "relevance": rel,
            "completeness": comp,
            "professionalism": prof,
            "total_score": total,
            "reasoning": f"Groundedness={ground}, Relevance={rel}, Completeness={comp}, Professionalism={prof}",
        }

    def _llm_judge(
        self,
        customer_message: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str,
        predicted_intent: str,
    ) -> Dict[str, Any]:
        prompt = f"""
You are an expert AI evaluator judging a customer support reply.
Customer Message: "{customer_message}"
Predicted Intent: {predicted_intent}
Retrieved Evidence: {json.dumps(retrieved_evidence)}
Generated Support Reply: "{generated_reply}"

Evaluate across 5 criteria (Score 0=Poor, 1=Acceptable, 2=Excellent for each):
1. Correctness (0-2)
2. Groundedness (0-2)
3. Relevance (0-2)
4. Completeness (0-2)
5. Professionalism (0-2)

Return JSON format with total_score (sum 0-10) and reasoning.
"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            raw = resp.text
        elif openai_key:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content
        else:
            raise ValueError("No LLM key available.")

        clean_str = raw.strip()
        if clean_str.startswith("```"):
            clean_str = "\n".join(clean_str.splitlines()[1:-1]).strip()

        return json.loads(clean_str)

def run_judge_evaluations() -> Dict[str, Any]:
    logger.info("Executing LLM-as-a-Judge Evaluation Suite across diverse test cases...")
    os.makedirs("results", exist_ok=True)

    judge = LLMJudgeEvaluator()
    sample_cases = [
        # Case 1: Excellent grounded reply (Expected Score 9-10)
        {
            "customer_message": "Where is my delayed order #12345?",
            "predicted_intent": "shipping_delay",
            "evidence": [{"case_id": "c101", "similarity": 0.85}],
            "reply": "Hello! We apologize for the delay. Your order tracking has been updated and is out for delivery today.",
            "type": "GOOD_REPLY",
        },
        # Case 2: Acceptable reply (Expected Score 7-8)
        {
            "customer_message": "Missing item from my delivered parcel box.",
            "predicted_intent": "missing_item",
            "evidence": [{"case_id": "c102", "similarity": 0.80}],
            "reply": "We apologize for the missing item. Please DM us your order ID so we can issue a replacement.",
            "type": "ACCEPTABLE_REPLY",
        },
        # Case 3: Hallucinated policy reply (Expected Score 3-5)
        {
            "customer_message": "I want a refund for my item.",
            "predicted_intent": "refund_return_request",
            "evidence": [],
            "reply": "We guarantee a 100% refund of $500 gift card immediately without returning the item!",
            "type": "HALLUCINATED_REPLY",
        },
        # Case 4: Insufficient short reply (Expected Score 2-4)
        {
            "customer_message": "My card was charged twice.",
            "predicted_intent": "payment_billing_issue",
            "evidence": [],
            "reply": "No idea.",
            "type": "POOR_SHORT_REPLY",
        },
    ]

    scores = []
    for case in sample_cases:
        res = judge.evaluate_reply(
            customer_message=case["customer_message"],
            retrieved_evidence=case["evidence"],
            generated_reply=case["reply"],
            predicted_intent=case["predicted_intent"],
        )
        res["case_type"] = case["type"]
        scores.append(res)

    avg_total = float(np.mean([s["total_score"] for s in scores]))
    avg_correctness = float(np.mean([s["correctness"] for s in scores]))
    avg_groundedness = float(np.mean([s["groundedness"] for s in scores]))
    avg_relevance = float(np.mean([s["relevance"] for s in scores]))

    output = {
        "evaluated_replies_count": len(scores),
        "mean_total_score": round(avg_total, 2),
        "mean_correctness": round(avg_correctness, 2),
        "mean_groundedness": round(avg_groundedness, 2),
        "mean_relevance": round(avg_relevance, 2),
        "detailed_scores": scores,
    }

    with open("results/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"LLM Judge Evaluation complete: Mean Total Score = {avg_total}/10 across {len(scores)} cases.")
    return output

if __name__ == "__main__":
    res = run_judge_evaluations()
    print(json.dumps(res, indent=2))
