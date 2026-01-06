import pytest
import pandas as pd

from ml.data import process_data
from ml.model import train_model, inference

# Shared fixtures (used across tests)
@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    data = {
        "age": [39, 50],
        "workclass": ["State-gov", "Self-emp-not-inc"],
        "fnlgt": [77516, 83311],
        "education": ["Bachelors", "Bachelors"],
        "education-num": [13, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"],
    }
    return pd.DataFrame(data)


# TODO: implement the first test
def test_process_data(sample_data):
    """
    Test that process_data returns non-empty feature and label arrays.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X is not None
    assert y is not None
    assert X.shape[0] == sample_data.shape[0]
    assert encoder is not None
    assert lb is not None


# TODO: implement the second test
def test_train_model(sample_data):
    """
    Test that train_model returns a trained model.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")


# TODO: implement the third test
def test_inference(sample_data):
    """
    Test that inference returns predictions with the correct length.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert preds is not None
    assert len(preds) == X.shape[0]
