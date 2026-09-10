import pytest
from src.retrieval.index import HistoricalCaseRetrievalIndex

def test_retrieval_index():
    cases = [
        {"case_id": "c1", "customer_message": "Where is my delayed package?", "historical_response": "We updated tracking."},
        {"case_id": "c2", "customer_message": "Missing item from my box", "historical_response": "Replacement issued."},
    ]

    index = HistoricalCaseRetrievalIndex()
    index.build_index(cases)

    results = index.search("Where is my package?", top_k=1)
    assert len(results) == 1
    assert results[0]["case_id"] == "c1"
    assert results[0]["similarity"] > 0.5
