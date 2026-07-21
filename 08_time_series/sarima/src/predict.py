"""Demonstrate a one-period SARIMA held-out forecast."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "sarima_model.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"


def load_artifacts():
    """Load the fitted model and held-out future data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "SARIMA model not found. Run train.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test data not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

    test_data = pd.read_csv(
        TEST_DATA_PATH,
        parse_dates=["date"],
    )

    return model, test_data


def main() -> None:
    """Forecast the first held-out future period."""

    model, test_data = load_artifacts()

    result = model.get_forecast(steps=1)

    forecast_value = float(
        result.predicted_mean.iloc[0]
    )

    interval = result.conf_int(
        alpha=0.05
    ).iloc[0]

    lower_bound = float(interval.iloc[0])
    upper_bound = float(interval.iloc[1])

    forecast_date = test_data[
        "date"
    ].iloc[0]

    actual_value = float(
        test_data["value"].iloc[0]
    )

    error = actual_value - forecast_value

    within_interval = (
        lower_bound
        <= actual_value
        <= upper_bound
    )

    print("\nSARIMA Held-Out Forecast Example")

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
        f"{error:.4f}"
    )

    print(
        f"Absolute error: "
        f"{abs(error):.4f}"
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
        "\nThis observation was not used "
        "during model fitting."
    )


if __name__ == "__main__":
    main()
