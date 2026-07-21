"""Demonstrate one-step XGBoost time-series forecasting."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_time_series_model.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "model_metadata.json"
)

TRAIN_DATA_PATH = (
    BASE_DIR
    / "data"
    / "train_data.csv"
)

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "test_data.csv"
)


def load_series(path: Path) -> pd.Series:
    """Load monthly time-series data."""

    table = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    series = table.set_index(
        "date"
    )["value"]

    return series


def load_artifacts():
    """Load model, metadata, and train/test data."""

    required_files = [
        MODEL_PATH,
        METADATA_PATH,
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required files are missing: "
            f"{missing_files}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    train_series = load_series(
        TRAIN_DATA_PATH
    )

    test_series = load_series(
        TEST_DATA_PATH
    )

    return (
        model,
        metadata,
        train_series,
        test_series,
    )


def create_feature_row(
    history: pd.Series,
    forecast_date: pd.Timestamp,
    metadata: dict,
) -> pd.DataFrame:
    """Create one prediction row using historical values only."""

    values = history.to_numpy()

    features: dict[str, float | int] = {}

    for lag_value in metadata["lags"]:
        lag = int(lag_value)

        features[f"lag_{lag}"] = float(
            values[-lag]
        )

    for window_value in metadata[
        "rolling_windows"
    ]:
        window = int(window_value)

        recent_values = values[-window:]

        features[
            f"rolling_mean_{window}"
        ] = float(
            np.mean(recent_values)
        )

        features[
            f"rolling_std_{window}"
        ] = float(
            np.std(
                recent_values,
                ddof=1,
            )
        )

        features[
            f"rolling_min_{window}"
        ] = float(
            np.min(recent_values)
        )

        features[
            f"rolling_max_{window}"
        ] = float(
            np.max(recent_values)
        )

    month_number = forecast_date.month

    features["month"] = month_number
    features["quarter"] = (
        forecast_date.quarter
    )

    features["month_sin"] = float(
        np.sin(
            2
            * np.pi
            * month_number
            / 12
        )
    )

    features["month_cos"] = float(
        np.cos(
            2
            * np.pi
            * month_number
            / 12
        )
    )

    features["time_index"] = len(
        history
    )

    feature_row = pd.DataFrame(
        [features]
    )

    return feature_row.loc[
        :,
        metadata["feature_columns"],
    ]


def main() -> None:
    """Forecast the first held-out future observation."""

    (
        model,
        metadata,
        train_series,
        test_series,
    ) = load_artifacts()

    forecast_date = test_series.index[0]

    feature_row = create_feature_row(
        history=train_series,
        forecast_date=forecast_date,
        metadata=metadata,
    )

    forecast_value = float(
        model.predict(
            feature_row
        )[0]
    )

    actual_value = float(
        test_series.iloc[0]
    )

    forecast_error = (
        actual_value
        - forecast_value
    )

    print(
        "\nXGBoost Time-Series "
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
        "\nFeature values used:"
    )

    print(
        feature_row.to_string(
            index=False
        )
    )

    print(
        "\nThis held-out value was not "
        "used during model fitting."
    )


if __name__ == "__main__":
    main()
