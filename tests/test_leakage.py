import pytest
from evaluation.leakage_check import check_split_leakage

def test_split_leakage_check():
    res = check_split_leakage()
    assert isinstance(res, dict)
    assert "has_leakage" in res
    assert res["train_val_id_overlap"] == 0
    assert res["train_test_id_overlap"] == 0
