"""Train and save a Prophet time-series forecasting model."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from prophet import Prophet
from statsmodels.datasets import co2


TEST_PERIODS = 24
FREQUENCY = "MS"

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

FULL_DATA_PATH = DATA_DIR / "full_series.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

MODEL_PATH = MODEL_DIR / "prophet_model.joblib"
TRAINING_SUMMARY_PATH = (
    METRICS_DIR / "training_summary.json"
)


def load_series() -> pd.DataFrame:
    """Load and convert monthly atmospheric CO2 data."""

    dataset = co2.load_pandas().data.copy()

    series = dataset["co2"]
    series.index = pd.to_datetime(series.index)

    monthly_series = (
        series
        .resample(FREQUENCY)
        .mean()
        .interpolate(method="time")
        .dropna()
    )

    prophet_data = (
        monthly_series
        .rename("y")
        .reset_index()
        .rename(columns={"index": "ds"})
    )

    prophet_data["ds"] = pd.to_datetime(
        prophet_data["ds"]
    )

    return prophet_data


def split_series(
    full_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological training and future test data."""

    if len(full_data) <= TEST_PERIODS:
        raise ValueError(
            "The dataset is too short for the configured test period."
        )

    train_data = (
        full_data
        .iloc[:-TEST_PERIODS]
        .copy()
        .reset_index(drop=True)
    )

    test_data = (
        full_data
        .iloc[-TEST_PERIODS:]
        .copy()
        .reset_index(drop=True)
    )

    return train_data, test_data


def build_model() -> Prophet:
    """Create the Prophet forecasting model."""

    return Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
        uncertainty_samples=1000,
    )


def save_datasets(
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save the complete chronological data split."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    full_data.to_csv(
        FULL_DATA_PATH,
        index=False,
    )

    train_data.to_csv(
        TRAIN_DATA_PATH,
        index=False,
    )

    test_data.to_csv(
        TEST_DATA_PATH,
        index=False,
    )


def save_training_summary(
    model: Prophet,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save training configuration and model information."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    changepoint_count = len(
        model.changepoints
    )

    summary = {
        "algorithm": "Prophet",
        "dataset": "Statsmodels atmospheric CO2",
        "frequency": FREQUENCY,
        "training_periods": len(train_data),
        "held_out_test_periods": len(test_data),
        "training_start": str(
            train_data["ds"].min().date()
        ),
        "training_end": str(
            train_data["ds"].max().date()
        ),
        "test_start": str(
            test_data["ds"].min().date()
        ),
        "test_end": str(
            test_data["ds"].max().date()
        ),
        "growth": model.growth,
        "seasonality_mode": model.seasonality_mode,
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "seasonality_prior_scale": (
            model.seasonality_prior_scale
        ),
        "changepoint_prior_scale": (
            model.changepoint_prior_scale
        ),
        "interval_width": model.interval_width,
        "uncertainty_samples": (
            model.uncertainty_samples
        ),
        "changepoint_count": changepoint_count,
        "chronological_split": True,
        "random_shuffling": False,
        "test_data_used_during_fitting": False,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def main() -> None:
    """Run the complete Prophet training workflow."""

    full_data = load_series()

    train_data, test_data = split_series(
        full_data
    )

    model = build_model()

    # Prophet is fitted only on historical training data.
    model.fit(train_data)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    save_datasets(
        full_data=full_data,
        train_data=train_data,
        test_data=test_data,
    )

    save_training_summary(
        model=model,
        train_data=train_data,
        test_data=test_data,
    )

    print("\nProphet training completed.")
    print(
        f"Training periods: {len(train_data)}"
    )
    print(
        f"Held-out future periods: {len(test_data)}"
    )
    print(
        "Training period: "
        f"{train_data['ds'].min().date()} "
        f"to {train_data['ds'].max().date()}"
    )
    print(
        "Held-out period: "
        f"{test_data['ds'].min().date()} "
        f"to {test_data['ds'].max().date()}"
    )
    print(
        f"Detected changepoints: "
        f"{len(model.changepoints)}"
    )
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()