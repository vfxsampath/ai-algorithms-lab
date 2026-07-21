"""Evaluate exponential-smoothing forecasts on held-out periods."""

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
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
)


SEASONAL_PERIODS = 12

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "exponential_smoothing_model.joblib"
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


def load_series(
    path: Path,
) -> pd.Series:
    """Load a dated monthly time series."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    table = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    series = (
        table
        .set_index("date")["value"]
    )

    series.index = pd.DatetimeIndex(
        series.index,
        freq="MS",
    )

    return series


def load_artifacts():
    """Load fitted model and chronological data splits."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Exponential-smoothing model "
            "not found. Run train.py first."
        )

    fitted_model = joblib.load(
        MODEL_PATH
    )

    train_series = load_series(
        TRAIN_DATA_PATH
    )

    test_series = load_series(
        TEST_DATA_PATH
    )

    return (
        fitted_model,
        train_series,
        test_series,
    )


def create_forecast(
    fitted_model,
    test_series: pd.Series,
) -> pd.Series:
    """Forecast the full held-out future horizon."""

    forecast_values = fitted_model.forecast(
        len(test_series)
    )

    return pd.Series(
        np.asarray(forecast_values),
        index=test_series.index,
        name="forecast",
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
    """Repeat the latest complete seasonal cycle."""

    if len(train_series) < SEASONAL_PERIODS:
        raise ValueError(
            "Insufficient training data for "
            "seasonal-naive forecasting."
        )

    latest_season = (
        train_series
        .iloc[-SEASONAL_PERIODS:]
        .to_numpy()
    )

    forecast_values = [
        latest_season[
            index % SEASONAL_PERIODS
        ]
        for index in range(
            len(test_series)
        )
    ]

    return pd.Series(
        forecast_values,
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
    """Calculate symmetric mean absolute percentage error."""

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
    """Calculate MASE using a seasonal-naive scaling error."""

    training_values = (
        train_series.to_numpy()
    )

    seasonal_errors = np.abs(
        training_values[
            SEASONAL_PERIODS:
        ]
        - training_values[
            :-SEASONAL_PERIODS
        ]
    )

    scaling_value = np.mean(
        seasonal_errors
    )

    if scaling_value == 0:
        return float("nan")

    forecast_mae = mean_absolute_error(
        actual,
        forecast,
    )

    return float(
        forecast_mae
        / scaling_value
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

    rmse = np.sqrt(mse)

    return (
        float(mae),
        float(mse),
        float(rmse),
    )


def calculate_metrics(
    train_series: pd.Series,
    test_series: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive_forecast: pd.Series,
) -> dict[str, float]:
    """Calculate forecast and baseline performance."""

    model_mae, model_mse, model_rmse = (
        basic_metrics(
            test_series,
            forecast,
        )
    )

    naive_mae, naive_mse, naive_rmse = (
        basic_metrics(
            test_series,
            naive_forecast,
        )
    )

    (
        seasonal_naive_mae,
        seasonal_naive_mse,
        seasonal_naive_rmse,
    ) = basic_metrics(
        test_series,
        seasonal_naive_forecast,
    )

    return {
        "model_mae": model_mae,
        "model_mse": model_mse,
        "model_rmse": model_rmse,
        "model_mape_percent": (
            calculate_mape(
                test_series.to_numpy(),
                forecast.to_numpy(),
            )
        ),
        "model_smape_percent": (
            calculate_smape(
                test_series.to_numpy(),
                forecast.to_numpy(),
            )
        ),
        "model_seasonal_mase": (
            calculate_seasonal_mase(
                train_series=train_series,
                actual=test_series,
                forecast=forecast,
            )
        ),
        "naive_mae": naive_mae,
        "naive_mse": naive_mse,
        "naive_rmse": naive_rmse,
        "seasonal_naive_mae": (
            seasonal_naive_mae
        ),
        "seasonal_naive_mse": (
            seasonal_naive_mse
        ),
        "seasonal_naive_rmse": (
            seasonal_naive_rmse
        ),
        "mae_improvement_over_naive_percent": float(
            (
                naive_mae
                - model_mae
            )
            / naive_mae
            * 100
            if naive_mae != 0
            else np.nan
        ),
        "rmse_improvement_over_naive_percent": float(
            (
                naive_rmse
                - model_rmse
            )
            / naive_rmse
            * 100
            if naive_rmse != 0
            else np.nan
        ),
        "mae_improvement_over_seasonal_naive_percent": float(
            (
                seasonal_naive_mae
                - model_mae
            )
            / seasonal_naive_mae
            * 100
            if seasonal_naive_mae != 0
            else np.nan
        ),
        "rmse_improvement_over_seasonal_naive_percent": float(
            (
                seasonal_naive_rmse
                - model_rmse
            )
            / seasonal_naive_rmse
            * 100
            if seasonal_naive_rmse != 0
            else np.nan
        ),
    }


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Save forecast evaluation metrics."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        METRICS_DIR
        / "metrics.json"
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
    test_series: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive_forecast: pd.Series,
) -> None:
    """Save future forecasts and forecast errors."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        {
            "date": (
                test_series.index
            ),
            "actual": (
                test_series.to_numpy()
            ),
            "exponential_smoothing_forecast": (
                forecast.to_numpy()
            ),
            "naive_forecast": (
                naive_forecast.to_numpy()
            ),
            "seasonal_naive_forecast": (
                seasonal_naive_forecast
                .to_numpy()
            ),
        }
    )

    results["forecast_error"] = (
        results["actual"]
        - results[
            "exponential_smoothing_forecast"
        ]
    )

    results["absolute_error"] = (
        results["forecast_error"]
        .abs()
    )

    results[
        "seasonal_naive_absolute_error"
    ] = (
        results["actual"]
        - results[
            "seasonal_naive_forecast"
        ]
    ).abs()

    results.to_csv(
        PREDICTIONS_DIR
        / "test_forecasts.csv",
        index=False,
    )


def save_residual_diagnostics(
    fitted_model,
) -> None:
    """Save residual statistics and autocorrelation tests."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals = pd.Series(
        fitted_model.resid
    ).dropna()

    summary = {
        "count": int(
            residuals.count()
        ),
        "mean": float(
            residuals.mean()
        ),
        "standard_deviation": float(
            residuals.std()
        ),
        "minimum": float(
            residuals.min()
        ),
        "median": float(
            residuals.median()
        ),
        "maximum": float(
            residuals.max()
        ),
    }

    with (
        METRICS_DIR
        / "residual_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    lags = [
        lag
        for lag in [
            6,
            12,
            18,
            24,
        ]
        if lag < len(residuals)
    ]

    if lags:
        ljung_box_results = (
            acorr_ljungbox(
                residuals,
                lags=lags,
                return_df=True,
            )
        )

        ljung_box_results.index.name = (
            "lag"
        )

        ljung_box_results.to_csv(
            METRICS_DIR
            / "ljung_box_test.csv"
        )


def save_component_data(
    fitted_model,
    train_series: pd.Series,
) -> None:
    """Save level, trend, seasonal, fitted, and residual values."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_count = len(train_series)

    level = np.asarray(
        fitted_model.level
    )

    trend = np.asarray(
        fitted_model.trend
    )

    season = np.asarray(
        fitted_model.season
    )

    if level.size != row_count:
        level = np.full(
            row_count,
            np.nan,
        )

    if trend.size != row_count:
        trend = np.full(
            row_count,
            np.nan,
        )

    if season.size != row_count:
        season = np.full(
            row_count,
            np.nan,
        )

    component_table = pd.DataFrame(
        {
            "date": train_series.index,
            "actual": train_series.to_numpy(),
            "fitted_value": np.asarray(
                fitted_model.fittedvalues
            ),
            "level": level,
            "trend": trend,
            "season": season,
            "residual": np.asarray(
                fitted_model.resid
            ),
        }
    )

    component_table.to_csv(
        METRICS_DIR
        / "fitted_components.csv",
        index=False,
    )


def save_train_test_plot(
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Plot chronological training and future periods."""

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
        "Exponential Smoothing "
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
    """Plot the held-out future forecast."""

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
        label=(
            "Exponential smoothing forecast"
        ),
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")

    plt.title(
        "Exponential Smoothing "
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


def save_baseline_comparison_plot(
    test_series: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive_forecast: pd.Series,
) -> None:
    """Compare the model with two baseline forecasts."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        test_series.index,
        test_series,
        marker="o",
        label="Actual",
    )

    plt.plot(
        forecast.index,
        forecast,
        marker="o",
        label="Selected smoothing model",
    )

    plt.plot(
        naive_forecast.index,
        naive_forecast,
        linestyle="--",
        label="Naive baseline",
    )

    plt.plot(
        seasonal_naive_forecast.index,
        seasonal_naive_forecast,
        linestyle=":",
        label="Seasonal-naive baseline",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")

    plt.title(
        "Exponential Smoothing "
        "vs Baselines"
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


def save_fitted_values_plot(
    fitted_model,
    train_series: pd.Series,
) -> None:
    """Compare training observations with fitted values."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recent_training = (
        train_series.iloc[-120:]
    )

    fitted_values = pd.Series(
        np.asarray(
            fitted_model.fittedvalues
        ),
        index=train_series.index,
    ).iloc[-120:]

    plt.figure(figsize=(11, 6))

    plt.plot(
        recent_training.index,
        recent_training,
        label="Observed training values",
    )

    plt.plot(
        fitted_values.index,
        fitted_values,
        linestyle="--",
        label="Fitted values",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")

    plt.title(
        "Observed and Fitted "
        "Training Values"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "fitted_values.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_component_plots(
    fitted_model,
    train_series: pd.Series,
) -> None:
    """Plot learned level, trend, and seasonal states."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    level = np.asarray(
        fitted_model.level
    )

    trend = np.asarray(
        fitted_model.trend
    )

    season = np.asarray(
        fitted_model.season
    )

    if level.size == len(train_series):
        plt.figure(figsize=(11, 5))

        plt.plot(
            train_series.index,
            level,
        )

        plt.xlabel("Date")
        plt.ylabel("Estimated level")

        plt.title(
            "Exponential Smoothing "
            "Level Component"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "level_component.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    if (
        trend.size == len(train_series)
        and not np.all(
            np.isnan(trend)
        )
    ):
        plt.figure(figsize=(11, 5))

        plt.plot(
            train_series.index,
            trend,
        )

        plt.xlabel("Date")
        plt.ylabel("Estimated trend")

        plt.title(
            "Exponential Smoothing "
            "Trend Component"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "trend_component.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    if (
        season.size == len(train_series)
        and not np.all(
            np.isnan(season)
        )
    ):
        recent_season = (
            pd.Series(
                season,
                index=train_series.index,
            )
            .iloc[-60:]
        )

        plt.figure(figsize=(11, 5))

        plt.plot(
            recent_season.index,
            recent_season,
            marker="o",
        )

        plt.xlabel("Date")
        plt.ylabel("Estimated seasonal effect")

        plt.title(
            "Exponential Smoothing "
            "Seasonal Component"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "seasonal_component.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


def save_residual_plots(
    fitted_model,
    train_series: pd.Series,
) -> None:
    """Plot residuals over time and their distribution."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals = pd.Series(
        np.asarray(
            fitted_model.resid
        ),
        index=train_series.index,
    ).dropna()

    plt.figure(figsize=(11, 5))

    plt.plot(
        residuals.index,
        residuals,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Date")
    plt.ylabel("Residual")

    plt.title(
        "Exponential Smoothing "
        "Residuals Over Time"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "residuals_over_time.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=30,
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.title(
        "Exponential Smoothing "
        "Residual Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "residual_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run complete held-out forecasting evaluation."""

    (
        fitted_model,
        train_series,
        test_series,
    ) = load_artifacts()

    forecast = create_forecast(
        fitted_model=fitted_model,
        test_series=test_series,
    )

    naive_forecast = create_naive_forecast(
        train_series=train_series,
        test_series=test_series,
    )

    seasonal_naive_forecast = (
        create_seasonal_naive_forecast(
            train_series=train_series,
            test_series=test_series,
        )
    )

    metrics = calculate_metrics(
        train_series=train_series,
        test_series=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive_forecast=(
            seasonal_naive_forecast
        ),
    )

    save_metrics(metrics)

    save_forecast_results(
        test_series=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive_forecast=(
            seasonal_naive_forecast
        ),
    )

    save_residual_diagnostics(
        fitted_model
    )

    save_component_data(
        fitted_model=fitted_model,
        train_series=train_series,
    )

    save_train_test_plot(
        train_series=train_series,
        test_series=test_series,
    )

    save_forecast_plot(
        train_series=train_series,
        test_series=test_series,
        forecast=forecast,
    )

    save_baseline_comparison_plot(
        test_series=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive_forecast=(
            seasonal_naive_forecast
        ),
    )

    save_fitted_values_plot(
        fitted_model=fitted_model,
        train_series=train_series,
    )

    save_component_plots(
        fitted_model=fitted_model,
        train_series=train_series,
    )

    save_residual_plots(
        fitted_model=fitted_model,
        train_series=train_series,
    )

    print(
        "\nExponential Smoothing "
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
