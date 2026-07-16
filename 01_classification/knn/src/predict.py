"""Generate predictions using the saved KNN classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "knn_classifier.joblib"


def load_model():
    """Load the trained KNN pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "KNN model not found. Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def validate_input(
    sample: pd.DataFrame,
    expected_features: list[str],
) -> None:
    """Validate feature names and values."""

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
            f"Missing features: {sorted(missing_features)}"
        )

    if unexpected_features:
        raise ValueError(
            f"Unexpected features: {sorted(unexpected_features)}"
        )

    if sample.isnull().any().any():
        raise ValueError(
            "Input contains missing values."
        )

    non_numeric_columns = (
        sample
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            f"Non-numeric columns: {non_numeric_columns}"
        )


def main() -> None:
    """Predict the class of one sample observation."""

    model = load_model()

    dataset = load_breast_cancer(as_frame=True)
    feature_names = list(dataset.feature_names)

    sample = dataset.data.iloc[[0]].copy()

    validate_input(
        sample=sample,
        expected_features=feature_names,
    )

    predicted_class = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    predicted_label = dataset.target_names[
        predicted_class
    ]

    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    scaled_sample = scaler.transform(sample)

    distances, indices = classifier.kneighbors(
        scaled_sample,
        return_distance=True,
    )

    print("\nKNN Prediction")
    print(f"Predicted class: {predicted_label}")
    print(
        f"Malignant probability: "
        f"{probabilities[0]:.4f}"
    )
    print(
        f"Benign probability: "
        f"{probabilities[1]:.4f}"
    )
    print(
        f"Prediction confidence: "
        f"{max(probabilities):.4f}"
    )

    print("\nNearest-neighbor distances:")

    for rank, distance in enumerate(
        distances[0],
        start=1,
    ):
        print(
            f"Neighbor {rank}: "
            f"distance={distance:.4f}, "
            f"training_index={indices[0][rank - 1]}"
        )


if __name__ == "__main__":
    main()