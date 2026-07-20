"""Select, fit, and save an ARIMA forecasting model."""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from statsmodels.datasets import co2
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


TEST_PERIODS = 24
FORECAST_FREQUENCY = "MS"

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

FULL_DATA_PATH = DATA_DIR / "full_series.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

MODEL_PATH = MODEL_DIR / "arima_model.joblib"
MODEL_SELECTION_PATH = METRICS_DIR / "candidate_orders.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"
STATIONARITY_PATH = METRICS_DIR / "stationarity_tests.json"


CANDIDATE_ORDERS = [
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
]


def load_series() -> pd.Series:
    """Load and prepare the monthly atmospheric CO2 series."""

    dataset = co2.load_pandas().data.copy()

    series = dataset["co2"]

    series.index = pd.to_datetime(series.index)

    monthly_series = (
        series
        .resample(FORECAST_FREQUENCY)
        .mean()
        .interpolate(method="time")
        .dropna()
    )

    monthly_series.name = "value"

    return monthly_series


def split_series(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Split the series chronologically into past and future periods."""

    if len(series) <= TEST_PERIODS:
        raise ValueError(
            "The series is too short for the configured test period."
        )

    train_series = series.iloc[:-TEST_PERIODS].copy()
    test_series = series.iloc[-TEST_PERIODS:].copy()

    return train_series, test_series


def save_series_data(
    full_series: pd.Series,
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Save full, training, and held-out series."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    full_series.rename("value").to_csv(
        FULL_DATA_PATH,
        index_label="date",
    )

    train_series.rename("value").to_csv(
        TRAIN_DATA_PATH,
        index_label="date",
    )

    test_series.rename("value").to_csv(
        TEST_DATA_PATH,
        index_label="date",
    )


def run_adf_test(
    series: pd.Series,
) -> dict[str, float | int | bool]:
    """Run the Augmented Dickey-Fuller stationarity test."""

    result = adfuller(
        series.dropna(),
        autolag="AIC",
    )

    return {
        "test_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lags": int(result[2]),
        "observation_count": int(result[3]),
        "critical_value_1_percent": float(
            result[4]["1%"]
        ),
        "critical_value_5_percent": float(
            result[4]["5%"]
        ),
        "critical_value_10_percent": float(
            result[4]["10%"]
        ),
        "stationary_at_5_percent": bool(
            result[1] < 0.05
        ),
    }


def save_stationarity_tests(
    train_series: pd.Series,
) -> None:
    """Save ADF results for original and differenced series."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    original_result = run_adf_test(train_series)

    differenced_series = (
        train_series
        .diff()
        .dropna()
    )

    differenced_result = run_adf_test(
        differenced_series
    )

    results = {
        "original_training_series": original_result,
        "first_differenced_training_series": (
            differenced_result
        ),
    }

    with STATIONARITY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)


def evaluate_candidate_orders(
    train_series: pd.Series,
) -> pd.DataFrame:
    """Fit candidate ARIMA orders and compare AIC and BIC."""

    rows: list[dict[str, object]] = []

    for order in CANDIDATE_ORDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fitted_model = ARIMA(
                    train_series,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()

            rows.append(
                {
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "order": str(order),
                    "aic": float(fitted_model.aic),
                    "bic": float(fitted_model.bic),
                    "hqic": float(fitted_model.hqic),
                    "log_likelihood": float(
                        fitted_model.llf
                    ),
                    "converged": bool(
                        fitted_model.mle_retvals.get(
                            "converged",
                            True,
                        )
                    ),
                    "status": "success",
                    "error": "",
                }
            )

        except (ValueError, np.linalg.LinAlgError) as error:
            rows.append(
                {
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "order": str(order),
                    "aic": np.nan,
                    "bic": np.nan,
                    "hqic": np.nan,
                    "log_likelihood": np.nan,
                    "converged": False,
                    "status": "failed",
                    "error": str(error),
                }
            )

    results = pd.DataFrame(rows)

    successful = results[
        results["status"] == "success"
    ]

    if successful.empty:
        raise RuntimeError(
            "All candidate ARIMA orders failed."
        )

    results = results.sort_values(
        by=["status", "aic"],
        ascending=[False, True],
        na_position="last",
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        MODEL_SELECTION_PATH,
        index=False,
    )

    return results


def select_best_order(
    candidate_results: pd.DataFrame,
) -> tuple[int, int, int]:
    """Select the successful candidate with the lowest AIC."""

    successful = candidate_results[
        candidate_results["status"] == "success"
    ].copy()

    best_row = successful.loc[
        successful["aic"].idxmin()
    ]

    return (
        int(best_row["p"]),
        int(best_row["d"]),
        int(best_row["q"]),
    )


def fit_final_model(
    train_series: pd.Series,
    order: tuple[int, int, int],
):
    """Fit the final selected ARIMA model."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fitted_model = ARIMA(
            train_series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()

    return fitted_model


def save_training_summary(
    train_series: pd.Series,
    test_series: pd.Series,
    model,
    selected_order: tuple[int, int, int],
) -> None:
    """Save model settings and training information."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "algorithm": "ARIMA",
        "dataset": "Statsmodels atmospheric CO2",
        "frequency": FORECAST_FREQUENCY,
        "training_periods": len(train_series),
        "held_out_test_periods": len(test_series),
        "training_start": str(
            train_series.index.min().date()
        ),
        "training_end": str(
            train_series.index.max().date()
        ),
        "test_start": str(
            test_series.index.min().date()
        ),
        "test_end": str(
            test_series.index.max().date()
        ),
        "selected_order": list(selected_order),
        "p": selected_order[0],
        "d": selected_order[1],
        "q": selected_order[2],
        "aic": float(model.aic),
        "bic": float(model.bic),
        "hqic": float(model.hqic),
        "log_likelihood": float(model.llf),
        "candidate_order_count": len(
            CANDIDATE_ORDERS
        ),
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
    """Run the complete ARIMA training workflow."""

    full_series = load_series()

    train_series, test_series = split_series(
        full_series
    )

    save_series_data(
        full_series=full_series,
        train_series=train_series,
        test_series=test_series,
    )

    save_stationarity_tests(train_series)

    candidate_results = evaluate_candidate_orders(
        train_series
    )

    selected_order = select_best_order(
        candidate_results
    )

    fitted_model = fit_final_model(
        train_series=train_series,
        order=selected_order,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        fitted_model,
        MODEL_PATH,
    )

    save_training_summary(
        train_series=train_series,
        test_series=test_series,
        model=fitted_model,
        selected_order=selected_order,
    )

    print("\nARIMA training completed.")
    print(
        f"Training periods: {len(train_series)}"
    )
    print(
        f"Held-out future periods: {len(test_series)}"
    )
    print(
        f"Selected order: {selected_order}"
    )
    print(f"AIC: {fitted_model.aic:.4f}")
    print(f"BIC: {fitted_model.bic:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()