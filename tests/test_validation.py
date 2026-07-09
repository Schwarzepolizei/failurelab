import pandas as pd
import pytest

from failurelab.data.validation import validate_target


def test_binary_classification():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "target": ["A", "B", "A", "B"]
    })

    result = validate_target(df, target_column="target")

    assert result["task_type"] == "binary"
    assert result["n_classes"] == 2
    assert result["class_counts"] == {"A": 2, "B": 2}

def test_multiclass_classification():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "target": ["A", "B", "C", "A", "B"]
    })

    result = validate_target(df, target_column="target")

    assert result["task_type"] == "multiclass"
    assert result["n_classes"] == 3
    assert result["class_counts"] == {"A": 2, "B": 2, "C": 1}

def test_one_class():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "target": ["A", "A", "A"]
    })

    with pytest.raises(ValueError):
        validate_target(df, target_column="target")

def test_missing_values_in_target():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "target": ["A", None, "B"]
    })

    with pytest.raises(ValueError):
        validate_target(df, target_column="target")

def test_missing_target_column():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "target": ["A", "B", "C"]
    })

    with pytest.raises(ValueError):
        validate_target(df, target_column="non_existent_column")