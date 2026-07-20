"""Evaluate Prophet forecasts on held-out future periods."""

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


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "prophet_model.joblib"
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


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a Prophet-formatted time-series CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    data = pd.read_csv(
        path,
        parse_dates=["ds"],
    )

    return data


def load_artifacts():
    """Load the model and chronological train/test data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Prophet model not found. "
            "Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    train_data = load_dataframe(
        TRAIN_DATA_PATH
    )
    test_data = load_dataframe(
        TEST_DATA_PATH
    )

    return model, train_data, test_data


def calculate_mape(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> float:
    """Calculate mean absolute percentage error."""

    non_zero_mask = actual != 0

    if not np.any(non_zero_mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (
                    actual[non_zero_mask]
                    - forecast[non_zero_mask]
                )
                / actual[non_zero_mask]
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


def calculate_mase(
    training_values: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
) -> float:
    """Calculate mean absolute scaled error."""

    naive_training_errors = np.abs(
        np.diff(training_values)
    )

    scaling_value = np.mean(
        naive_training_errors
    )

    if scaling_value == 0:
        return float("nan")

    forecast_mae = mean_absolute_error(
        actual,
        forecast,
    )

    return float(
        forecast_mae / scaling_value
    )


def create_forecast(
    model,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Generate forecasts for exact held-out dates."""

    future_dates = test_data[["ds"]].copy()

    forecast = model.predict(
        future_dates
    )

    required_columns = [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "trend",
    ]

    available_columns = [
        column
        for column in required_columns
        if column in forecast.columns
    ]

    return forecast[
        available_columns
    ].copy()


def create_naive_forecast(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> np.ndarray:
    """Forecast every future period using the last training value."""

    last_value = float(
        train_data["y"].iloc[-1]
    )

    return np.full(
        shape=len(test_data),
        fill_value=last_value,
        dtype=float,
    )


def calculate_metrics(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    forecast: pd.DataFrame,
    naive_forecast: np.ndarray,
) -> dict[str, float]:
    """Calculate Prophet and naïve-baseline metrics."""

    actual = test_data["y"].to_numpy()
    predicted = forecast["yhat"].to_numpy()

    prophet_mse = mean_squared_error(
        actual,
        predicted,
    )

    naive_mse = mean_squared_error(
        actual,
        naive_forecast,
    )

    prophet_mae = mean_absolute_error(
        actual,
        predicted,
    )

    naive_mae = mean_absolute_error(
        actual,
        naive_forecast,
    )

    interval_coverage = np.mean(
        (
            actual
            >= forecast[
                "yhat_lower"
            ].to_numpy()
        )
        & (
            actual
            <= forecast[
                "yhat_upper"
            ].to_numpy()
        )
    )

    return {
        "prophet_mae": float(
            prophet_mae
        ),
        "prophet_mse": float(
            prophet_mse
        ),
        "prophet_rmse": float(
            np.sqrt(prophet_mse)
        ),
        "prophet_mape_percent": (
            calculate_mape(
                actual,
                predicted,
            )
        ),
        "prophet_smape_percent": (
            calculate_smape(
                actual,
                predicted,
            )
        ),
        "prophet_mase": calculate_mase(
            train_data["y"].to_numpy(),
            actual,
            predicted,
        ),
        "naive_mae": float(
            naive_mae
        ),
        "naive_mse": float(
            naive_mse
        ),
        "naive_rmse": float(
            np.sqrt(naive_mse)
        ),
        "mae_improvement_over_naive_percent": float(
            (
                naive_mae
                - prophet_mae
            )
            / naive_mae
            * 100
            if naive_mae != 0
            else np.nan
        ),
        "rmse_improvement_over_naive_percent": float(
            (
                np.sqrt(naive_mse)
                - np.sqrt(prophet_mse)
            )
            / np.sqrt(naive_mse)
            * 100
            if naive_mse != 0
            else np.nan
        ),
        "forecast_interval_coverage": float(
            interval_coverage
        ),
    }


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Save Prophet evaluation metrics."""

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
    test_data: pd.DataFrame,
    forecast: pd.DataFrame,
    naive_forecast: np.ndarray,
) -> None:
    """Save held-out forecasts and forecast errors."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        {
            "date": test_data[
                "ds"
            ].to_numpy(),
            "actual": test_data[
                "y"
            ].to_numpy(),
            "prophet_forecast": forecast[
                "yhat"
            ].to_numpy(),
            "lower_95": forecast[
                "yhat_lower"
            ].to_numpy(),
            "upper_95": forecast[
                "yhat_upper"
            ].to_numpy(),
            "trend": forecast[
                "trend"
            ].to_numpy(),
            "naive_forecast": (
                naive_forecast
            ),
        }
    )

    results["forecast_error"] = (
        results["actual"]
        - results["prophet_forecast"]
    )

    results["absolute_error"] = (
        results["forecast_error"]
        .abs()
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
        PREDICTIONS_DIR
        / "test_forecasts.csv",
        index=False,
    )


def save_train_test_plot(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Visualize the chronological split."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        train_data["ds"],
        train_data["y"],
        label="Training history",
    )

    plt.plot(
        test_data["ds"],
        test_data["y"],
        label="Held-out future",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title(
        "Prophet Chronological Train-Test Split"
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
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    forecast: pd.DataFrame,
) -> None:
    """Plot recent history and held-out forecast."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recent_training = (
        train_data
        .tail(60)
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        recent_training["ds"],
        recent_training["y"],
        label="Recent training history",
    )

    plt.plot(
        test_data["ds"],
        test_data["y"],
        marker="o",
        label="Actual held-out values",
    )

    plt.plot(
        forecast["ds"],
        forecast["yhat"],
        linestyle="--",
        label="Prophet forecast",
    )

    plt.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2,
        label="95% interval",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title(
        "Prophet Held-Out Forecast"
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
    test_data: pd.DataFrame,
    forecast: pd.DataFrame,
    naive_forecast: np.ndarray,
) -> None:
    """Compare Prophet with the naïve baseline."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        test_data["ds"],
        test_data["y"],
        marker="o",
        label="Actual",
    )

    plt.plot(
        forecast["ds"],
        forecast["yhat"],
        marker="o",
        label="Prophet",
    )

    plt.plot(
        test_data["ds"],
        naive_forecast,
        linestyle="--",
        label="Naïve baseline",
    )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title(
        "Prophet vs Naïve Forecast"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "prophet_vs_naive.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_residual_outputs(
    test_data: pd.DataFrame,
    forecast: pd.DataFrame,
) -> None:
    """Save and visualize held-out forecast residuals."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals = (
        test_data["y"].to_numpy()
        - forecast["yhat"].to_numpy()
    )

    summary = {
        "count": int(
            len(residuals)
        ),
        "mean": float(
            np.mean(residuals)
        ),
        "standard_deviation": float(
            np.std(
                residuals,
                ddof=1,
            )
        ),
        "minimum": float(
            np.min(residuals)
        ),
        "median": float(
            np.median(residuals)
        ),
        "maximum": float(
            np.max(residuals)
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

    plt.figure(figsize=(10, 5))

    plt.plot(
        test_data["ds"],
        residuals,
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Date")
    plt.ylabel("Forecast Residual")
    plt.title(
        "Prophet Held-Out Residuals"
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
        bins=12,
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(
        "Prophet Residual Distribution"
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "residual_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_component_plots(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save Prophet trend and seasonality components."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_dates = pd.concat(
        [
            train_data[["ds"]],
            test_data[["ds"]],
        ],
        ignore_index=True,
    )

    complete_forecast = model.predict(
        complete_dates
    )

    component_figure = (
        model.plot_components(
            complete_forecast
        )
    )

    component_figure.savefig(
        FIGURES_DIR
        / "forecast_components.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        component_figure
    )


def save_changepoint_plot(
    model,
    train_data: pd.DataFrame,
) -> None:
    """Visualize trend and learned changepoint locations."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fitted = model.predict(
        train_data[["ds"]]
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        train_data["ds"],
        train_data["y"],
        label="Training observations",
    )

    plt.plot(
        fitted["ds"],
        fitted["trend"],
        label="Estimated trend",
    )

    for changepoint in model.changepoints:
        plt.axvline(
            changepoint,
            alpha=0.15,
        )

    plt.xlabel("Date")
    plt.ylabel("Monthly CO2")
    plt.title(
        "Prophet Trend and Potential Changepoints"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "trend_changepoints.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run the complete held-out Prophet evaluation."""

    model, train_data, test_data = (
        load_artifacts()
    )

    forecast = create_forecast(
        model=model,
        test_data=test_data,
    )

    naive_forecast = create_naive_forecast(
        train_data=train_data,
        test_data=test_data,
    )

    metrics = calculate_metrics(
        train_data=train_data,
        test_data=test_data,
        forecast=forecast,
        naive_forecast=naive_forecast,
    )

    save_metrics(metrics)

    save_forecast_results(
        test_data=test_data,
        forecast=forecast,
        naive_forecast=naive_forecast,
    )

    save_train_test_plot(
        train_data=train_data,
        test_data=test_data,
    )

    save_forecast_plot(
        train_data=train_data,
        test_data=test_data,
        forecast=forecast,
    )

    save_baseline_comparison_plot(
        test_data=test_data,
        forecast=forecast,
        naive_forecast=naive_forecast,
    )

    save_residual_outputs(
        test_data=test_data,
        forecast=forecast,
    )

    save_component_plots(
        model=model,
        train_data=train_data,
        test_data=test_data,
    )

    save_changepoint_plot(
        model=model,
        train_data=train_data,
    )

    print("\nProphet Held-Out Evaluation")

    for metric_name, metric_value in (
        metrics.items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    print(
        "\nEvaluation outputs saved successfully."
    )


if __name__ == "__main__":
    main()
