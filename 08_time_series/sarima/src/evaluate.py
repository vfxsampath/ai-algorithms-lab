"""Evaluate SARIMA forecasts on held-out future periods."""

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
from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf,
)
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
)


SEASONAL_PERIOD = 12

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "sarima_model.joblib"
TRAIN_DATA_PATH = BASE_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = (
    BASE_DIR / "outputs" / "predictions"
)


def load_series(path: Path) -> pd.Series:
    """Load a dated monthly series."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    table = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    series = table.set_index("date")["value"]
    series.index = pd.DatetimeIndex(
        series.index,
        freq="MS",
    )

    return series


def load_artifacts():
    """Load SARIMA model and chronological data splits."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "SARIMA model not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    train_series = load_series(TRAIN_DATA_PATH)
    test_series = load_series(TEST_DATA_PATH)

    return model, train_series, test_series


def create_forecast(
    model,
    test_series: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Forecast the complete held-out horizon."""

    result = model.get_forecast(
        steps=len(test_series)
    )

    forecast = pd.Series(
        result.predicted_mean.to_numpy(),
        index=test_series.index,
        name="sarima_forecast",
    )

    interval = result.conf_int(
        alpha=0.05
    ).copy()

    interval.index = test_series.index
    interval.columns = [
        "lower_95",
        "upper_95",
    ]

    return forecast, interval


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
    """Repeat observations from the previous seasonal cycle."""

    if len(train_series) < SEASONAL_PERIOD:
        raise ValueError(
            "Not enough training data for seasonal naïve forecasting."
        )

    last_season = train_series.iloc[
        -SEASONAL_PERIOD:
    ].to_numpy()

    forecast_values = [
        last_season[
            index % SEASONAL_PERIOD
        ]
        for index in range(len(test_series))
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

    valid = actual != 0

    if not np.any(valid):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (
                    actual[valid]
                    - forecast[valid]
                )
                / actual[valid]
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

    valid = denominator != 0

    if not np.any(valid):
        return float("nan")

    return float(
        200
        * np.mean(
            np.abs(
                actual[valid]
                - forecast[valid]
            )
            / denominator[valid]
        )
    )


def calculate_mase(
    train_series: pd.Series,
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate seasonally scaled MASE."""

    seasonal_errors = np.abs(
        train_series.to_numpy()[
            SEASONAL_PERIOD:
        ]
        - train_series.to_numpy()[
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


def model_metrics(
    actual: pd.Series,
    forecast: pd.Series,
) -> tuple[float, float, float]:
    """Return MAE, MSE, and RMSE."""

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
    sarima_forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive: pd.Series,
    interval: pd.DataFrame,
) -> dict[str, float]:
    """Calculate SARIMA and baseline metrics."""

    sarima_mae, sarima_mse, sarima_rmse = (
        model_metrics(
            actual,
            sarima_forecast,
        )
    )

    naive_mae, naive_mse, naive_rmse = (
        model_metrics(
            actual,
            naive_forecast,
        )
    )

    seasonal_mae, seasonal_mse, seasonal_rmse = (
        model_metrics(
            actual,
            seasonal_naive,
        )
    )

    interval_coverage = np.mean(
        (
            actual.to_numpy()
            >= interval["lower_95"].to_numpy()
        )
        & (
            actual.to_numpy()
            <= interval["upper_95"].to_numpy()
        )
    )

    return {
        "sarima_mae": sarima_mae,
        "sarima_mse": sarima_mse,
        "sarima_rmse": sarima_rmse,
        "sarima_mape_percent": calculate_mape(
            actual.to_numpy(),
            sarima_forecast.to_numpy(),
        ),
        "sarima_smape_percent": calculate_smape(
            actual.to_numpy(),
            sarima_forecast.to_numpy(),
        ),
        "sarima_seasonal_mase": calculate_mase(
            train_series,
            actual,
            sarima_forecast,
        ),
        "naive_mae": naive_mae,
        "naive_mse": naive_mse,
        "naive_rmse": naive_rmse,
        "seasonal_naive_mae": seasonal_mae,
        "seasonal_naive_mse": seasonal_mse,
        "seasonal_naive_rmse": seasonal_rmse,
        "mae_improvement_over_seasonal_naive_percent": float(
            (
                seasonal_mae - sarima_mae
            )
            / seasonal_mae
            * 100
            if seasonal_mae != 0
            else np.nan
        ),
        "rmse_improvement_over_seasonal_naive_percent": float(
            (
                seasonal_rmse - sarima_rmse
            )
            / seasonal_rmse
            * 100
            if seasonal_rmse != 0
            else np.nan
        ),
        "forecast_interval_95_coverage": float(
            interval_coverage
        ),
    }


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Save evaluation metrics."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with (
        METRICS_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_forecasts(
    actual: pd.Series,
    sarima_forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive: pd.Series,
    interval: pd.DataFrame,
) -> None:
    """Save forecasts and errors."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        {
            "date": actual.index,
            "actual": actual.to_numpy(),
            "sarima_forecast": (
                sarima_forecast.to_numpy()
            ),
            "naive_forecast": (
                naive_forecast.to_numpy()
            ),
            "seasonal_naive_forecast": (
                seasonal_naive.to_numpy()
            ),
            "lower_95": (
                interval["lower_95"].to_numpy()
            ),
            "upper_95": (
                interval["upper_95"].to_numpy()
            ),
        }
    )

    results["sarima_error"] = (
        results["actual"]
        - results["sarima_forecast"]
    )

    results["absolute_error"] = (
        results["sarima_error"].abs()
    )

    results["within_interval"] = (
        (
            results["actual"]
            >= results["lower_95"]
        )
        & (
            results["actual"]
            <= results["upper_95"]
        )
    )

    results.to_csv(
        PREDICTIONS_DIR / "test_forecasts.csv",
        index=False,
    )


def save_residual_diagnostics(model) -> None:
    """Save residual summary and Ljung-Box tests."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    residuals = pd.Series(
        model.resid
    ).dropna()

    summary = {
        "count": int(residuals.count()),
        "mean": float(residuals.mean()),
        "standard_deviation": float(
            residuals.std()
        ),
        "minimum": float(residuals.min()),
        "median": float(residuals.median()),
        "maximum": float(residuals.max()),
    }

    with (
        METRICS_DIR / "residual_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    lags = [
        lag
        for lag in [12, 24, 36]
        if lag < len(residuals)
    ]

    if lags:
        test_results = acorr_ljungbox(
            residuals,
            lags=lags,
            return_df=True,
        )

        test_results.index.name = "lag"

        test_results.to_csv(
            METRICS_DIR / "ljung_box_test.csv"
        )


def save_train_test_plot(
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Plot chronological split."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    plt.title("SARIMA Chronological Train-Test Split")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "train_test_split.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_forecast_plot(
    train_series: pd.Series,
    test_series: pd.Series,
    forecast: pd.Series,
    interval: pd.DataFrame,
) -> None:
    """Plot held-out forecast and interval."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    recent_train = train_series.iloc[-72:]

    plt.figure(figsize=(11, 6))

    plt.plot(
        recent_train.index,
        recent_train,
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
        linestyle="--",
        label="SARIMA forecast",
    )

    plt.fill_between(
        interval.index,
        interval["lower_95"],
        interval["upper_95"],
        alpha=0.2,
        label="95% forecast interval",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title("SARIMA Held-Out Forecast")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "held_out_forecast.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_baseline_comparison(
    actual: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    seasonal_naive: pd.Series,
) -> None:
    """Compare SARIMA with both baselines."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
        label="SARIMA",
    )

    plt.plot(
        naive_forecast.index,
        naive_forecast,
        linestyle="--",
        label="Naïve baseline",
    )

    plt.plot(
        seasonal_naive.index,
        seasonal_naive,
        linestyle=":",
        label="Seasonal-naïve baseline",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title("SARIMA vs Forecasting Baselines")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "baseline_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_seasonal_difference_plot(
    train_series: pd.Series,
) -> None:
    """Plot first and seasonal differencing."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    first_difference = (
        train_series
        .diff()
        .dropna()
    )

    seasonal_difference = (
        train_series
        .diff(SEASONAL_PERIOD)
        .dropna()
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        first_difference.index,
        first_difference,
        label="First difference",
    )

    plt.plot(
        seasonal_difference.index,
        seasonal_difference,
        alpha=0.75,
        label="12-month seasonal difference",
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Date")
    plt.ylabel("Differenced value")
    plt.title("Non-Seasonal and Seasonal Differencing")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "seasonal_differencing.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_acf_pacf(
    train_series: pd.Series,
) -> None:
    """Save ACF and PACF after seasonal differencing."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    transformed = (
        train_series
        .diff()
        .diff(SEASONAL_PERIOD)
        .dropna()
    )

    maximum_lags = min(
        48,
        len(transformed) // 2 - 1,
    )

    plot_acf(
        transformed,
        lags=maximum_lags,
    )

    plt.title(
        "ACF After First and Seasonal Differencing"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "acf.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    plot_pacf(
        transformed,
        lags=maximum_lags,
        method="ywm",
    )

    plt.title(
        "PACF After First and Seasonal Differencing"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "pacf.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_residual_plots(model) -> None:
    """Save residual time and distribution charts."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = pd.Series(
        model.resid
    ).dropna()

    plt.figure(figsize=(11, 5))

    plt.plot(
        residuals.index,
        residuals,
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.title("SARIMA Residuals Over Time")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "residuals_over_time.png",
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
    plt.title("SARIMA Residual Distribution")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "residual_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run complete held-out SARIMA evaluation."""

    model, train_series, test_series = (
        load_artifacts()
    )

    forecast, interval = create_forecast(
        model=model,
        test_series=test_series,
    )

    naive_forecast = create_naive_forecast(
        train_series=train_series,
        test_series=test_series,
    )

    seasonal_naive = (
        create_seasonal_naive_forecast(
            train_series=train_series,
            test_series=test_series,
        )
    )

    metrics = calculate_metrics(
        train_series=train_series,
        actual=test_series,
        sarima_forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive=seasonal_naive,
        interval=interval,
    )

    save_metrics(metrics)

    save_forecasts(
        actual=test_series,
        sarima_forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive=seasonal_naive,
        interval=interval,
    )

    save_residual_diagnostics(model)

    save_train_test_plot(
        train_series,
        test_series,
    )

    save_forecast_plot(
        train_series=train_series,
        test_series=test_series,
        forecast=forecast,
        interval=interval,
    )

    save_baseline_comparison(
        actual=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        seasonal_naive=seasonal_naive,
    )

    save_seasonal_difference_plot(
        train_series
    )

    save_acf_pacf(train_series)
    save_residual_plots(model)

    print("\nSARIMA Held-Out Evaluation")

    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    print(
        "\nEvaluation outputs saved successfully."
    )


if __name__ == "__main__":
    main()
