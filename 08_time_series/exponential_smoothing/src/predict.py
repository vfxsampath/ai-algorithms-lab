"""Demonstrate one-period exponential-smoothing forecasting."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "exponential_smoothing_model.joblib"
)

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "test_data.csv"
)


def load_artifacts():
    """Load fitted model and held-out future data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Exponential-smoothing model "
            "not found. Run train.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Held-out test data not found. "
            "Run train.py first."
        )

    fitted_model = joblib.load(
        MODEL_PATH
    )

    test_data = pd.read_csv(
        TEST_DATA_PATH,
        parse_dates=["date"],
    )

    return fitted_model, test_data


def main() -> None:
    """Forecast the first held-out future observation."""

    fitted_model, test_data = (
        load_artifacts()
    )

    forecast_values = (
        fitted_model.forecast(1)
    )

    forecast_value = float(
        forecast_values.iloc[0]
        if hasattr(
            forecast_values,
            "iloc",
        )
        else forecast_values[0]
    )

    forecast_date = (
        test_data["date"].iloc[0]
    )

    actual_value = float(
        test_data["value"].iloc[0]
    )

    forecast_error = (
        actual_value
        - forecast_value
    )

    print(
        "\nExponential Smoothing "
        "Held-Out Forecast Example"
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
        f"Forecast error: "
        f"{forecast_error:.4f}"
    )

    print(
        f"Absolute error: "
        f"{abs(forecast_error):.4f}"
    )

    print(
        "\nThis observation was not "
        "used during model fitting."
    )

    print(
        "The selected exponential-smoothing "
        "implementation does not provide "
        "forecast intervals in this workflow."
    )


if __name__ == "__main__":
    main()
