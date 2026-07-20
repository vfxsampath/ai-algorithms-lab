"""Evaluate the saved ARIMA model on held-out future periods."""

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


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "arima_model.joblib"
TRAIN_DATA_PATH = BASE_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"


def load_series(path: Path) -> pd.Series:
    """Load a dated time series from CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}"
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
    """Load model and chronological train/test series."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ARIMA model not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    train_series = load_series(TRAIN_DATA_PATH)
    test_series = load_series(TEST_DATA_PATH)

    return model, train_series, test_series


def calculate_mape(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate mean absolute percentage error."""

    actual_values = actual.to_numpy()
    forecast_values = forecast.to_numpy()

    non_zero_mask = actual_values != 0

    if not np.any(non_zero_mask):
        return float("nan")

    percentage_errors = np.abs(
        (
            actual_values[non_zero_mask]
            - forecast_values[non_zero_mask]
        )
        / actual_values[non_zero_mask]
    )

    return float(
        np.mean(percentage_errors) * 100
    )


def calculate_smape(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate symmetric MAPE."""

    actual_values = actual.to_numpy()
    forecast_values = forecast.to_numpy()

    denominator = (
        np.abs(actual_values)
        + np.abs(forecast_values)
    )

    valid_mask = denominator != 0

    if not np.any(valid_mask):
        return float("nan")

    result = (
        200
        * np.mean(
            np.abs(
                actual_values[valid_mask]
                - forecast_values[valid_mask]
            )
            / denominator[valid_mask]
        )
    )

    return float(result)


def calculate_mase(
    train_series: pd.Series,
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate mean absolute scaled error."""

    naive_training_errors = np.abs(
        np.diff(train_series.to_numpy())
    )

    scaling_value = np.mean(
        naive_training_errors
    )

    if scaling_value == 0:
        return float("nan")

    model_mae = mean_absolute_error(
        actual,
        forecast,
    )

    return float(model_mae / scaling_value)


def calculate_metrics(
    train_series: pd.Series,
    actual: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
) -> dict[str, float]:
    """Calculate ARIMA and baseline forecast metrics."""

    mse = mean_squared_error(
        actual,
        forecast,
    )

    naive_mse = mean_squared_error(
        actual,
        naive_forecast,
    )

    arima_mae = mean_absolute_error(
        actual,
        forecast,
    )

    naive_mae = mean_absolute_error(
        actual,
        naive_forecast,
    )

    return {
        "arima_mae": float(arima_mae),
        "arima_mse": float(mse),
        "arima_rmse": float(np.sqrt(mse)),
        "arima_mape_percent": calculate_mape(
            actual,
            forecast,
        ),
        "arima_smape_percent": calculate_smape(
            actual,
            forecast,
        ),
        "arima_mase": calculate_mase(
            train_series,
            actual,
            forecast,
        ),
        "naive_mae": float(naive_mae),
        "naive_mse": float(naive_mse),
        "naive_rmse": float(
            np.sqrt(naive_mse)
        ),
        "mae_improvement_over_naive_percent": float(
            (
                naive_mae - arima_mae
            )
            / naive_mae
            * 100
            if naive_mae != 0
            else np.nan
        ),
        "rmse_improvement_over_naive_percent": float(
            (
                np.sqrt(naive_mse)
                - np.sqrt(mse)
            )
            / np.sqrt(naive_mse)
            * 100
            if naive_mse != 0
            else np.nan
        ),
    }


def create_forecast(
    model,
    test_series: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate point forecasts and confidence intervals."""

    forecast_result = model.get_forecast(
        steps=len(test_series)
    )

    forecast_mean = pd.Series(
        forecast_result.predicted_mean.to_numpy(),
        index=test_series.index,
        name="forecast",
    )

    confidence_interval = (
        forecast_result.conf_int(alpha=0.05)
        .copy()
    )

    confidence_interval.index = test_series.index

    confidence_interval.columns = [
        "lower_95",
        "upper_95",
    ]

    return forecast_mean, confidence_interval


def create_naive_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
) -> pd.Series:
    """Use the last training value as a naïve forecast."""

    last_training_value = float(
        train_series.iloc[-1]
    )

    return pd.Series(
        last_training_value,
        index=test_series.index,
        name="naive_forecast",
    )


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Save forecast metrics."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with (
        METRICS_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_forecast_table(
    actual: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
    confidence_interval: pd.DataFrame,
) -> None:
    """Save predictions and forecast errors."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        {
            "date": actual.index,
            "actual": actual.to_numpy(),
            "arima_forecast": forecast.to_numpy(),
            "naive_forecast": (
                naive_forecast.to_numpy()
            ),
            "lower_95": (
                confidence_interval[
                    "lower_95"
                ].to_numpy()
            ),
            "upper_95": (
                confidence_interval[
                    "upper_95"
                ].to_numpy()
            ),
        }
    )

    results["arima_error"] = (
        results["actual"]
        - results["arima_forecast"]
    )

    results["absolute_error"] = (
        results["arima_error"].abs()
    )

    results["within_95_interval"] = (
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
    """Save residual statistics and Ljung-Box results."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    residuals = pd.Series(
        model.resid
    ).dropna()

    residual_summary = {
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
        json.dump(
            residual_summary,
            file,
            indent=4,
        )

    lags = [
        lag
        for lag in [6, 12, 18, 24]
        if lag < len(residuals)
    ]

    if lags:
        ljung_box = acorr_ljungbox(
            residuals,
            lags=lags,
            return_df=True,
        )

        ljung_box.index.name = "lag"

        ljung_box.to_csv(
            METRICS_DIR / "ljung_box_test.csv"
        )


def save_series_plot(
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Plot training and held-out time periods."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))

    plt.plot(
        train_series.index,
        train_series,
        label="Training period",
    )

    plt.plot(
        test_series.index,
        test_series,
        label="Held-out future period",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title("Chronological ARIMA Train-Test Split")
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
    confidence_interval: pd.DataFrame,
) -> None:
    """Plot held-out forecast and interval."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    recent_training = train_series.iloc[-60:]

    plt.figure(figsize=(11, 6))

    plt.plot(
        recent_training.index,
        recent_training,
        label="Recent training data",
    )

    plt.plot(
        test_series.index,
        test_series,
        label="Actual held-out values",
    )

    plt.plot(
        forecast.index,
        forecast,
        linestyle="--",
        label="ARIMA forecast",
    )

    plt.fill_between(
        confidence_interval.index,
        confidence_interval["lower_95"],
        confidence_interval["upper_95"],
        alpha=0.2,
        label="95% forecast interval",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title("ARIMA Held-Out Forecast")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "held_out_forecast.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_actual_vs_forecast_plot(
    actual: pd.Series,
    forecast: pd.Series,
    naive_forecast: pd.Series,
) -> None:
    """Compare ARIMA and naïve forecasts."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

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
        label="ARIMA",
    )

    plt.plot(
        naive_forecast.index,
        naive_forecast,
        linestyle="--",
        label="Naïve baseline",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title("ARIMA vs Naïve Forecast")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "arima_vs_naive_forecast.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_acf_pacf_plots(
    train_series: pd.Series,
) -> None:
    """Save ACF and PACF plots of differenced training data."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    differenced = (
        train_series
        .diff()
        .dropna()
    )

    maximum_lags = min(
        36,
        len(differenced) // 2 - 1,
    )

    plot_acf(
        differenced,
        lags=maximum_lags,
    )

    plt.title(
        "ACF of First-Differenced Training Series"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "acf.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    plot_pacf(
        differenced,
        lags=maximum_lags,
        method="ywm",
    )

    plt.title(
        "PACF of First-Differenced Training Series"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "pacf.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_residual_plots(model) -> None:
    """Save residual time-series and distribution plots."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = pd.Series(
        model.resid
    ).dropna()

    plt.figure(figsize=(11, 5))

    plt.plot(residuals.index, residuals)

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("ARIMA Residuals Over Time")
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
    plt.title("ARIMA Residual Distribution")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "residual_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run complete held-out ARIMA evaluation."""

    model, train_series, test_series = (
        load_artifacts()
    )

    forecast, confidence_interval = (
        create_forecast(
            model=model,
            test_series=test_series,
        )
    )

    naive_forecast = create_naive_forecast(
        train_series=train_series,
        test_series=test_series,
    )

    metrics = calculate_metrics(
        train_series=train_series,
        actual=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
    )

    interval_coverage = float(
        np.mean(
            (
                test_series
                >= confidence_interval["lower_95"]
            )
            & (
                test_series
                <= confidence_interval["upper_95"]
            )
        )
    )

    metrics[
        "forecast_interval_95_coverage"
    ] = interval_coverage

    save_metrics(metrics)

    save_forecast_table(
        actual=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
        confidence_interval=confidence_interval,
    )

    save_residual_diagnostics(model)

    save_series_plot(
        train_series=train_series,
        test_series=test_series,
    )

    save_forecast_plot(
        train_series=train_series,
        test_series=test_series,
        forecast=forecast,
        confidence_interval=confidence_interval,
    )

    save_actual_vs_forecast_plot(
        actual=test_series,
        forecast=forecast,
        naive_forecast=naive_forecast,
    )

    save_acf_pacf_plots(train_series)

    save_residual_plots(model)

    print("\nARIMA Held-Out Evaluation")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(
        "\nForecast outputs saved successfully."
    )


if __name__ == "__main__":
    main()
