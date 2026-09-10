import pytest
import pandas as pd
from src.data.load import validate_schema, generate_schema_sample
from src.data.clean import clean_text

def test_schema_validation():
    df = generate_schema_sample()
    is_valid, errors = validate_schema(df)
    assert is_valid is True
    assert len(errors) == 0

def test_schema_validation_missing_col():
    df = pd.DataFrame({"tweet_id": ["1"], "text": ["hello"]})
    is_valid, errors = validate_schema(df)
    assert is_valid is False
    assert any("Missing required columns" in err for err in errors)

def test_clean_text():
    raw = "  Hello @AmazonHelp check out https://amazon.com/order123   \n "
    cleaned = clean_text(raw, remove_handles=True, remove_urls=True)
    assert cleaned == "Hello check out [URL]"
