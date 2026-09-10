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
    Falls back to structured deterministic heuristic judge when offline or MOCK_LLM=true.
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
        msg_lower = customer_message.lower()
        reply_lower = generated_reply.lower()

        # Rubric scoring heuristics
        corr = 2 if len(generated_reply) > 10 else 1
        ground = 2 if (retrieved_evidence or "escalat" in reply_lower or "dm" in reply_lower) else 1
        rel = 2 if any(w in reply_lower for w in ["order", "delay", "return", "refund", "assist", "help", "escalat"]) else 1
        comp = 2 if len(generated_reply.split()) >= 8 else 1
        prof = 2 if not any(w in reply_lower for w in ["stupid", "idiot", "damn"]) else 0

        total = corr + ground + rel + comp + prof

        return {
            "correctness": corr,
            "groundedness": ground,
            "relevance": rel,
            "completeness": comp,
            "professionalism": prof,
            "total_score": total,
            "reasoning": "Heuristic evaluation based on grounding evidence, keyword relevance, and length.",
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
    logger.info("Executing LLM-as-a-Judge Evaluation Suite...")
    os.makedirs("results", exist_ok=True)

    judge = LLMJudgeEvaluator()
    sample_cases = [
        {
            "customer_message": "Where is my delayed order #12345?",
            "predicted_intent": "shipping_delay",
            "evidence": [{"case_id": "c101", "similarity": 0.85}],
            "reply": "Hello! We apologize for the delay. Your order tracking has been updated and is out for delivery today.",
        },
        {
            "customer_message": "Missing item from my delivered parcel box.",
            "predicted_intent": "missing_item",
            "evidence": [{"case_id": "c102", "similarity": 0.80}],
            "reply": "We apologize for the missing item. We have issued a free replacement order for you.",
        },
        {
            "customer_message": "How do I return a product?",
            "predicted_intent": "refund_return_request",
            "evidence": [{"case_id": "c103", "similarity": 0.88}],
            "reply": "You can start a return under Your Orders -> Return or Replace Items on our app.",
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

    logger.info(f"LLM Judge Evaluation complete: Mean Total Score = {avg_total}/10")
    return output

if __name__ == "__main__":
    res = run_judge_evaluations()
    print(json.dumps(res, indent=2))
