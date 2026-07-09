import pandas as pd
import pytest   

from failurelab.data.split import split_dataset


def test_basic_split():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test = split_dataset(df, target_column="target")

    assert len(X_train) + len(X_test) == len(df)
    assert "target" not in X_train.columns
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_missing_target_column():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": ["A", "B", "C"]
    })

    with pytest.raises(ValueError):
        split_dataset(df, target_column="target")

def test_no_stratify_split():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test = split_dataset(df, target_column="target", stratify=False)

    assert len(X_train) + len(X_test) == len(df)
    assert "target" not in X_train.columns