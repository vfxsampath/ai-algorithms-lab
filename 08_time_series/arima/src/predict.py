"""Demonstrate one-step ARIMA forecasting on held-out data."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "arima_model.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"


def load_test_series() -> pd.Series:
    """Load the held-out future series."""

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Held-out test data not found. Run train.py first."
        )

    table = pd.read_csv(
        TEST_DATA_PATH,
        parse_dates=["date"],
    )

    series = table.set_index("date")["value"]

    return series


def load_model():
    """Load the saved fitted ARIMA model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ARIMA model not found. Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def main() -> None:
    """Forecast the first held-out period."""

    model = load_model()
    test_series = load_test_series()

    forecast_result = model.get_forecast(
        steps=1
    )

    predicted_value = float(
        forecast_result.predicted_mean.iloc[0]
    )

    confidence_interval = (
        forecast_result.conf_int(
            alpha=0.05
        )
        .iloc[0]
    )

    actual_value = float(
        test_series.iloc[0]
    )

    forecast_date = test_series.index[0]

    error = actual_value - predicted_value
    absolute_error = abs(error)

    lower_bound = float(
        confidence_interval.iloc[0]
    )

    upper_bound = float(
        confidence_interval.iloc[1]
    )

    within_interval = (
        lower_bound
        <= actual_value
        <= upper_bound
    )

    print("\nARIMA Held-Out Forecast Example")

    print(
        f"Forecast date: "
        f"{forecast_date.date()}"
    )

    print(
        f"Forecast value: "
        f"{predicted_value:.4f}"
    )

    print(
        f"Actual held-out value: "
        f"{actual_value:.4f}"
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
        "95% forecast interval: "
        f"[{lower_bound:.4f}, "
        f"{upper_bound:.4f}]"
    )

    print(
        "Actual value within interval: "
        f"{within_interval}"
    )

    print(
        "\nThis period was not used when "
        "fitting the ARIMA model."
    )


if __name__ == "__main__":
    main()
