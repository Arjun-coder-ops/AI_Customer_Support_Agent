import pytest
from src.pipeline import SupportAgentPipeline

def test_full_pipeline_execution():
    pipeline = SupportAgentPipeline(brand_name="AmazonHelp", use_mock=True)
    
    train_cases = [
        {"customer_initial_message": "Where is my delayed package?", "brand": "AmazonHelp"},
        {"customer_initial_message": "Missing item from box", "brand": "AmazonHelp"},
    ]
    resolved_cases = [
        {"case_id": "c1", "customer_message": "Where is my delayed package?", "historical_response": "Tracking updated.", "turn_count": 2},
    ]

    pipeline.train_and_index(train_cases, resolved_cases)
    res = pipeline.process_message("Where is my package?")

    assert "customer_message" in res
    assert "intent" in res
    assert "retrieval" in res
    assert "escalation" in res
    assert "generated_reply" in res
