"""Demonstrate one-period Prophet forecasting."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "prophet_model.joblib"
)

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "test_data.csv"
)


def load_artifacts():
    """Load the saved Prophet model and test data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Prophet model not found. "
            "Run train.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test data not found. "
            "Run train.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    test_data = pd.read_csv(
        TEST_DATA_PATH,
        parse_dates=["ds"],
    )

    return model, test_data


def main() -> None:
    """Forecast the first held-out future observation."""

    model, test_data = load_artifacts()

    example = (
        test_data
        .iloc[[0]]
        .copy()
    )

    forecast = model.predict(
        example[["ds"]]
    )

    forecast_value = float(
        forecast["yhat"].iloc[0]
    )

    lower_bound = float(
        forecast[
            "yhat_lower"
        ].iloc[0]
    )

    upper_bound = float(
        forecast[
            "yhat_upper"
        ].iloc[0]
    )

    trend_value = float(
        forecast["trend"].iloc[0]
    )

    actual_value = float(
        example["y"].iloc[0]
    )

    forecast_date = (
        example["ds"].iloc[0]
    )

    error = (
        actual_value
        - forecast_value
    )

    absolute_error = abs(error)

    within_interval = (
        lower_bound
        <= actual_value
        <= upper_bound
    )

    print(
        "\nProphet Held-Out Forecast Example"
    )

    print(
        f"Forecast date: "
        f"{forecast_date.date()}"
    )

    print(
        f"Forecast value: "
        f"{forecast_value:.4f}"
    )

    print(
        f"Actual held-out value: "
        f"{actual_value:.4f}"
    )

    print(
        f"Estimated trend: "
        f"{trend_value:.4f}"
    )

    print(
        f"Forecast error: "
        f"{error:.4f}"
    )

    print(
        f"Absolute error: "
        f"{absolute_error:.4f}"
    )

    print(
        "95% uncertainty interval: "
        f"[{lower_bound:.4f}, "
        f"{upper_bound:.4f}]"
    )

    print(
        "Actual value within interval: "
        f"{within_interval}"
    )

    print(
        "\nThis observation was not used "
        "when fitting the Prophet model."
    )


if __name__ == "__main__":
    main()
