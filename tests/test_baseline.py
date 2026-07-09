import pandas as pd
import pytest

from failurelab.modeling.baseline import train_baseline_model


@pytest.fixture
def training_data():
    X_train = pd.DataFrame({
        "age": [25, 33, 45, 52, 23, 40],
        "income": [50000, 60000, 70000, 80000, 45000, 65000],
        "channel": ["online", "offline", "online", "offline", "online", "offline"],
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    return X_train, y_train

def test_logistic_regression_model_training(training_data):

    X_train, y_train = training_data
    model = train_baseline_model(X_train, y_train, model_type="logistic_regression")
    preds = model.predict(X_train)

    assert model is not None
    assert hasattr(model, "predict")
    assert len(preds) == len(y_train)


def test_random_forest_model_training(training_data):

    X_train, y_train = training_data
    model = train_baseline_model(X_train, y_train, model_type="random_forest")
    preds = model.predict(X_train)

    assert model is not None
    assert hasattr(model, "predict")
    assert len(preds) == len(y_train)


def test_invalid_model_type(training_data):
    
    X_train, y_train = training_data

    with pytest.raises(ValueError):
        train_baseline_model(X_train, y_train, model_type="unsupported_model")