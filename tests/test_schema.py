import pandas as pd
import pytest

from failurelab.data.schema import infer_schema


def test_infer_schema_basic():
    df = pd.DataFrame({
        "age": [25, 33, 45],
        "income": [50000, 60000, 70000],
        "channel": ["online", "offline", "online"],
        "target": ["product_a", "product_b", "product_c"],
    })

    result = infer_schema(df, target_column="target")

    assert result["target_column"] == "target"  
    assert "age" in result["numeric_columns"]
    assert "income" in result["numeric_columns"]
    assert "channel" in result["categorical_columns"]
    assert "target" not in result["feature_columns"]

def test_missing_target_column():
    df = pd.DataFrame({
        "age": [25, 33, 45],
        "income": [50000, 60000, 70000],
        "channel": ["online", "offline", "online"],
        "target": ["product_a", "product_b", "product_c"],
    })

    with pytest.raises(ValueError):
        infer_schema(df, target_column="non_existent_column")

def test_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        infer_schema(df, target_column="target")