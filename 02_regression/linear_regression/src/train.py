"""Train and save a Linear Regression model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset."""

    dataset = fetch_california_housing(as_frame=True)
    return dataset.data, dataset.target


def build_pipeline() -> Pipeline:
    """Build the preprocessing and regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def save_test_data(
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save the held-out test data for independent evaluation."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_data = x_test.copy()
    test_data["target"] = y_test.to_numpy()

    test_data.to_csv(
        DATA_DIR / "test_data.csv",
        index=False,
    )


def main() -> None:
    """Train the model and save all required artifacts."""

    features, target = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "linear_regression.joblib",
    )

    save_test_data(x_test, y_test)

    print("Linear Regression training completed.")
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(x_test)}")
    print(
        "Model saved to: "
        f"{MODELS_DIR / 'linear_regression.joblib'}"
    )


if __name__ == "__main__":
    main()