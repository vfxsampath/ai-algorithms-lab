"""Evaluate recursive XGBoost forecasts on held-out future periods."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


SEASONAL_PERIOD = 12

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

FIGURES_DIR = (
    BASE_DIR
    / "outputs"
    / "figures"
)

METRICS_DIR = (
    BASE_DIR
    / "outputs"
    / "metrics"
)

PREDICTIONS_DIR = (
    BASE_DIR
    / "outputs"
    / "predictions"
)


def load_series(path: Path) -> pd.Series:
    """Load a monthly dated time series."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}"
        )

    table = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    series = table.set_index(
        "date"
    )["value"]

    series.index = pd.DatetimeIndex(
        series.index,
        freq="MS",
    )

    return series


def load_artifacts():
    """Load model, metadata, and chronological data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "XGBoost model not found. "
            "Run train.py first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata not found. "
            "Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

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


def create_single_feature_row(
    history: pd.Series,
    forecast_date: pd.Timestamp,
    feature_names: list[str],
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Build one future feature row from available history."""

    values = history.to_numpy()

    required_history = max(
        max(lags),
        max(rolling_windows),
    )

    if len(values) < required_history:
        raise ValueError(
            "Insufficient historical values "
            "for feature construction."
        )

    feature_values: dict[str, float | int] = {}

    for lag in lags:
        feature_values[f"lag_{lag}"] = float(
            values[-lag]
        )

    for window in rolling_windows:
        recent_values = values[-window:]

        feature_values[
            f"rolling_mean_{window}"
        ] = float(
            np.mean(recent_values)
        )

        feature_values[
            f"rolling_std_{window}"
        ] = float(
            np.std(
                recent_values,
                ddof=1,
            )
        )

        feature_values[
            f"rolling_min_{window}"
        ] = float(
            np.min(recent_values)
        )

        feature_values[
            f"rolling_max_{window}"
        ] = float(
            np.max(recent_values)
        )

    month_number = forecast_date.month

    feature_values["month"] = month_number
    feature_values["quarter"] = (
        forecast_date.quarter
    )

    feature_values["month_sin"] = float(
        np.sin(
            2
            * np.pi
            * month_number
            / 12
        )
    )

    feature_values["month_cos"] = float(
        np.cos(
            2
            * np.pi
            * month_number
            / 12
        )
    )

    feature_values["time_index"] = len(
        history
    )

    feature_row = pd.DataFrame(
        [feature_values]
    )

    return feature_row.loc[
        :,
        feature_names,
    ]


def recursive_forecast(
    model,
    train_series: pd.Series,
    test_dates: pd.DatetimeIndex,
    metadata: dict,
) -> pd.Series:
    """Forecast multiple periods recursively."""

    history = train_series.copy()

    forecasts = []

    feature_names = metadata[
        "feature_columns"
    ]

    lags = [
        int(value)
        for value in metadata["lags"]
    ]

    rolling_windows = [
        int(value)
        for value in metadata[
            "rolling_windows"
        ]
    ]

    for forecast_date in test_dates:
        feature_row = (
            create_single_feature_row(
                history=history,
                forecast_date=forecast_date,
                feature_names=feature_names,
                lags=lags,
                rolling_windows=rolling_windows,
            )
        )

        prediction = float(
            model.predict(feature_row)[0]
        )

        forecasts.append(prediction)

        history.loc[forecast_date] = prediction

    return pd.Series(
        forecasts,
        index=test_dates,
        name="xgboost_forecast",
    )


def create_naive_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
) -> pd.Series:
    """Repeat the final training observation."""

    return pd.Series(
        float(train_series.iloc[-1]),
        index=test_series.index,
        name="naive_forecast",
    )


def create_seasonal_naive_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
) -> pd.Series:
    """Repeat the most recent yearly cycle."""

    latest_season = (
        train_series
        .iloc[-SEASONAL_PERIOD:]
        .to_numpy()
    )

    forecasts = [
        latest_season[
            index % SEASONAL_PERIOD
        ]
        for index in range(
            len(test_series)
        )
    ]

    return pd.Series(
        forecasts,
        index=test_series.index,
        name="seasonal_naive_forecast",
    )


def calculate_mape(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> float:
    """Calculate mean absolute percentage error."""

    valid_mask = actual != 0

    if not np.any(valid_mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (
                    actual[valid_mask]
                    - forecast[valid_mask]
                )
                / actual[valid_mask]
            )
        )
        * 100
    )


def calculate_smape(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> float:
    """Calculate symmetric MAPE."""

    denominator = (
        np.abs(actual)
        + np.abs(forecast)
    )

    valid_mask = denominator != 0

    if not np.any(valid_mask):
        return float("nan")

    return float(
        200
        * np.mean(
            np.abs(
                actual[valid_mask]
                - forecast[valid_mask]
            )
            / denominator[valid_mask]
        )
    )


def calculate_seasonal_mase(
    train_series: pd.Series,
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate seasonal mean absolute scaled error."""

    training_values = (
        train_series.to_numpy()
    )

    seasonal_errors = np.abs(
        training_values[
            SEASONAL_PERIOD:
        ]
        - training_values[
            :-SEASONAL_PERIOD
        ]
    )

    scale = np.mean(seasonal_errors)

    if scale == 0:
        return float("nan")

    return float(
        mean_absolute_error(
            actual,
            forecast,
        )
        / scale
    )


def basic_metrics(
    actual: pd.Series,
    forecast: pd.Series,
) -> tuple[float, float, float]:
    """Calculate MAE, MSE, and RMSE."""

    mae = mean_absolute_error(
        actual,
        forecast,
    )

    mse = mean_squared_error(
        actual,
        forecast,
    )

    return (
        float(mae),
        float(mse),
        float(np.sqrt(mse)),
    )


def calculate_metrics(
    train_series: pd.Series,
    actual: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive: pd.Series,
) -> dict[str, float]:
    """Calculate XGBoost and baseline forecast metrics."""

    model_mae, model_mse, model_rmse = (
        basic_metrics(
            actual,
            forecast,
        )
    )

    naive_mae, naive_mse, naive_rmse = (
        basic_metrics(
            actual,
            naive_forecast,
        )
    )

    (
        seasonal_mae,
        seasonal_mse,
        seasonal_rmse,
    ) = basic_metrics(
        actual,
        seasonal_naive,
    )

    return {
        "xgboost_mae": model_mae,
        "xgboost_mse": model_mse,
        "xgboost_rmse": model_rmse,
        "xgboost_mape_percent": (
            calculate_mape(
                actual.to_numpy(),
                forecast.to_numpy(),
            )
        ),
        "xgboost_smape_percent": (
            calculate_smape(
                actual.to_numpy(),
                forecast.to_numpy(),
            )
        ),
        "xgboost_seasonal_mase": (
            calculate_seasonal_mase(
                train_series,
                actual,
                forecast,
            )
        ),
        "naive_mae": naive_mae,
        "naive_mse": naive_mse,
        "naive_rmse": naive_rmse,
        "seasonal_naive_mae": seasonal_mae,
        "seasonal_naive_mse": seasonal_mse,
        "seasonal_naive_rmse": seasonal_rmse,
        "mae_improvement_over_naive_percent": float(
            (
                naive_mae - model_mae
            )
            / naive_mae
            * 100
            if naive_mae != 0
            else np.nan
        ),
        "rmse_improvement_over_naive_percent": float(
            (
                naive_rmse - model_rmse
            )
            / naive_rmse
            * 100
            if naive_rmse != 0
            else np.nan
        ),
        "mae_improvement_over_seasonal_naive_percent": float(
            (
                seasonal_mae - model_mae
            )
            / seasonal_mae
            * 100
            if seasonal_mae != 0
            else np.nan
        ),
        "rmse_improvement_over_seasonal_naive_percent": float(
            (
                seasonal_rmse
                - model_rmse
            )
            / seasonal_rmse
            * 100
            if seasonal_rmse != 0
            else np.nan
        ),
    }


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Save forecast metrics."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        METRICS_DIR / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )


def save_forecast_results(
    actual: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive: pd.Series,
) -> None:
    """Save held-out forecasts and errors."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        {
            "date": actual.index,
            "actual": actual.to_numpy(),
            "xgboost_recursive_forecast": (
                forecast.to_numpy()
            ),
            "naive_forecast": (
                naive_forecast.to_numpy()
            ),
            "seasonal_naive_forecast": (
                seasonal_naive.to_numpy()
            ),
        }
    )

    results["forecast_error"] = (
        results["actual"]
        - results[
            "xgboost_recursive_forecast"
        ]
    )

    results["absolute_error"] = (
        results["forecast_error"].abs()
    )

    results["forecast_horizon"] = (
        np.arange(
            1,
            len(results) + 1,
        )
    )

    results.to_csv(
        PREDICTIONS_DIR
        / "test_forecasts.csv",
        index=False,
    )


def save_train_test_plot(
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Plot chronological train-test separation."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        train_series.index,
        train_series,
        label="Training history",
    )

    plt.plot(
        test_series.index,
        test_series,
        label="Held-out future",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")

    plt.title(
        "XGBoost Time-Series "
        "Chronological Split"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "train_test_split.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_forecast_plot(
    train_series: pd.Series,
    test_series: pd.Series,
    forecast: pd.Series,
) -> None:
    """Plot recursive held-out forecasts."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recent_training = (
        train_series.iloc[-72:]
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        recent_training.index,
        recent_training,
        label="Recent training history",
    )

    plt.plot(
        test_series.index,
        test_series,
        marker="o",
        label="Actual held-out values",
    )

    plt.plot(
        forecast.index,
        forecast,
        marker="o",
        linestyle="--",
        label="XGBoost recursive forecast",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")

    plt.title(
        "XGBoost Recursive "
        "Held-Out Forecast"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "held_out_forecast.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_baseline_plot(
    actual: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive: pd.Series,
) -> None:
    """Compare XGBoost with forecasting baselines."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        actual.index,
        actual,
        marker="o",
        label="Actual",
    )

    plt.plot(
        forecast.index,
        forecast,
        marker="o",
        label="XGBoost",
    )

    plt.plot(
        naive_forecast.index,
        naive_forecast,
        linestyle="--",
        label="Naive baseline",
    )

    plt.plot(
        seasonal_naive.index,
        seasonal_naive,
        linestyle=":",
        label="Seasonal-naive baseline",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")

    plt.title(
        "XGBoost vs Forecasting Baselines"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "baseline_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_error_by_horizon_plot(
    actual: pd.Series,
    forecast: pd.Series,
) -> None:
    """Visualize error accumulation across the forecast horizon."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    absolute_errors = np.abs(
        actual.to_numpy()
        - forecast.to_numpy()
    )

    horizons = np.arange(
        1,
        len(actual) + 1,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        horizons,
        absolute_errors,
        marker="o",
    )

    plt.xlabel("Forecast Horizon")
    plt.ylabel("Absolute Forecast Error")

    plt.title(
        "Recursive Forecast Error "
        "by Horizon"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "error_by_horizon.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_feature_importance_plot() -> None:
    """Visualize the model's strongest input features."""

    importance_path = (
        METRICS_DIR
        / "feature_importance.csv"
    )

    if not importance_path.exists():
        raise FileNotFoundError(
            "Feature importance file not found. "
            "Run train.py first."
        )

    importance_data = pd.read_csv(
        importance_path
    ).head(15)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 7))

    ordered = importance_data.sort_values(
        by="importance",
        ascending=True,
    )

    plt.barh(
        ordered["feature"],
        ordered["importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.title(
        "XGBoost Time-Series "
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run complete held-out XGBoost evaluation."""

    (
        model,
        metadata,
        train_series,
        test_series,
    ) = load_artifacts()

    forecast = recursive_forecast(
        model=model,
        train_series=train_series,
        test_dates=test_series.index,
        metadata=metadata,
    )

    naive_forecast = create_naive_forecast(
        train_series,
        test_series,
    )

    seasonal_naive = (
        create_seasonal_naive_forecast(
            train_series,
            test_series,
        )
    )

    metrics = calculate_metrics(
        train_series=train_series,
        actual=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive=seasonal_naive,
    )

    save_metrics(metrics)

    save_forecast_results(
        actual=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive=seasonal_naive,
    )

    save_train_test_plot(
        train_series,
        test_series,
    )

    save_forecast_plot(
        train_series,
        test_series,
        forecast,
    )

    save_baseline_plot(
        actual=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive=seasonal_naive,
    )

    save_error_by_horizon_plot(
        actual=test_series,
        forecast=forecast,
    )

    save_feature_importance_plot()

    print(
        "\nXGBoost Time-Series "
        "Held-Out Evaluation"
    )

    for metric_name, metric_value in (
        metrics.items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    print(
        "\nEvaluation outputs "
        "saved successfully."
    )


if __name__ == "__main__":
    main()
