"""Generate predictions with the saved Random Forest model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest.joblib"
)


def load_model():
    """Load the trained Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Random Forest model not found. "
            "Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def validate_input(
    sample: pd.DataFrame,
    expected_features: list[str],
) -> None:
    """Validate input columns and numerical values."""

    missing_features = (
        set(expected_features)
        - set(sample.columns)
    )

    unexpected_features = (
        set(sample.columns)
        - set(expected_features)
    )

    if missing_features:
        raise ValueError(
            "Missing features: "
            f"{sorted(missing_features)}"
        )

    if unexpected_features:
        raise ValueError(
            "Unexpected features: "
            f"{sorted(unexpected_features)}"
        )

    if sample.isnull().any().any():
        raise ValueError(
            "Input contains missing values."
        )

    non_numeric_columns = (
        sample
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric input columns: "
            f"{non_numeric_columns}"
        )


def main() -> None:
    """Generate a prediction for one example observation."""

    model = load_model()

    dataset = load_breast_cancer(
        as_frame=True
    )

    feature_names = list(
        dataset.feature_names
    )

    sample = (
        dataset.data
        .iloc[[0]]
        .copy()
    )

    validate_input(
        sample=sample,
        expected_features=feature_names,
    )

    predicted_class = (
        model.predict(sample)[0]
    )

    class_probabilities = (
        model.predict_proba(sample)[0]
    )

    predicted_label = (
        dataset.target_names[
            predicted_class
        ]
    )

    print(
        "\nRandom Forest Prediction"
    )

    print(
        f"Predicted class: "
        f"{predicted_label}"
    )

    print(
        "Malignant probability: "
        f"{class_probabilities[0]:.4f}"
    )

    print(
        "Benign probability: "
        f"{class_probabilities[1]:.4f}"
    )

    print(
        "Prediction confidence: "
        f"{max(class_probabilities):.4f}"
    )


if __name__ == "__main__":
    main()