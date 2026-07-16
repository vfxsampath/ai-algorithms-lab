"""Generate new predictions with the saved Linear Regression model."""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "linear_regression.joblib"

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_model():
    """Load the trained model from disk."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def validate_input(sample: pd.DataFrame) -> None:
    """Check that all expected model features are provided."""

    missing_columns = set(FEATURE_NAMES) - set(sample.columns)
    unexpected_columns = set(sample.columns) - set(FEATURE_NAMES)

    if missing_columns:
        raise ValueError(
            f"Missing features: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected features: {sorted(unexpected_columns)}"
        )


def main() -> None:
    """Predict a median house value for one example record."""

    model = load_model()

    sample_property = pd.DataFrame(
        [
            {
                "MedInc": 5.0,
                "HouseAge": 25.0,
                "AveRooms": 6.0,
                "AveBedrms": 1.1,
                "Population": 1200.0,
                "AveOccup": 3.0,
                "Latitude": 34.05,
                "Longitude": -118.25,
            }
        ],
        columns=FEATURE_NAMES,
    )

    validate_input(sample_property)

    prediction = model.predict(sample_property)[0]

    print(
        "Predicted median house value "
        f"(units of $100,000): {prediction:.3f}"
    )
    print(
        "Approximate predicted value: "
        f"${prediction * 100_000:,.0f}"
    )


if __name__ == "__main__":
    main()
