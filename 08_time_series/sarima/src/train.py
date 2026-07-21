"""Select, fit, and save a SARIMA forecasting model."""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from statsmodels.datasets import co2
from statsmodels.tsa.statespace.sarimax import SARIMAX


TEST_PERIODS = 24
FREQUENCY = "MS"
SEASONAL_PERIOD = 12

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

FULL_DATA_PATH = DATA_DIR / "full_series.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

MODEL_PATH = MODEL_DIR / "sarima_model.joblib"
MODEL_SELECTION_PATH = (
    METRICS_DIR / "candidate_models.csv"
)
TRAINING_SUMMARY_PATH = (
    METRICS_DIR / "training_summary.json"
)


CANDIDATE_CONFIGURATIONS = [
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 0), (0, 1, 1, 12)),
    ((1, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 0, 12)),
    ((1, 1, 1), (1, 1, 1, 12)),
    ((2, 1, 0), (0, 1, 1, 12)),
    ((0, 1, 2), (0, 1, 1, 12)),
    ((2, 1, 1), (1, 1, 0, 12)),
]


def load_series() -> pd.Series:
    """Load and prepare monthly atmospheric CO2 data."""

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

    monthly_series.name = "value"

    return monthly_series


def split_series(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Create chronological training and future test periods."""

    if len(series) <= TEST_PERIODS:
        raise ValueError(
            "Series is too short for the configured test horizon."
        )

    train_series = series.iloc[:-TEST_PERIODS].copy()
    test_series = series.iloc[-TEST_PERIODS:].copy()

    return train_series, test_series


def save_series_data(
    full_series: pd.Series,
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Save the full and split time-series datasets."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    full_series.to_csv(
        FULL_DATA_PATH,
        index_label="date",
    )

    train_series.to_csv(
        TRAIN_DATA_PATH,
        index_label="date",
    )

    test_series.to_csv(
        TEST_DATA_PATH,
        index_label="date",
    )


def fit_candidate(
    train_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
):
    """Fit one candidate seasonal model."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        return model.fit(
            disp=False,
            maxiter=300,
        )


def evaluate_candidates(
    train_series: pd.Series,
) -> pd.DataFrame:
    """Fit candidate SARIMA configurations and save criteria."""

    rows: list[dict[str, object]] = []

    for order, seasonal_order in CANDIDATE_CONFIGURATIONS:
        try:
            fitted_model = fit_candidate(
                train_series=train_series,
                order=order,
                seasonal_order=seasonal_order,
            )

            rows.append(
                {
                    "order": str(order),
                    "seasonal_order": str(
                        seasonal_order
                    ),
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "P": seasonal_order[0],
                    "D": seasonal_order[1],
                    "Q": seasonal_order[2],
                    "seasonal_period": seasonal_order[3],
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

        except (
            ValueError,
            np.linalg.LinAlgError,
            RuntimeError,
        ) as error:
            rows.append(
                {
                    "order": str(order),
                    "seasonal_order": str(
                        seasonal_order
                    ),
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "P": seasonal_order[0],
                    "D": seasonal_order[1],
                    "Q": seasonal_order[2],
                    "seasonal_period": seasonal_order[3],
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
            "All candidate SARIMA configurations failed."
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


def select_best_configuration(
    candidate_results: pd.DataFrame,
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int, int],
]:
    """Select the successful configuration with lowest AIC."""

    successful = candidate_results[
        candidate_results["status"] == "success"
    ]

    best_row = successful.loc[
        successful["aic"].idxmin()
    ]

    order = (
        int(best_row["p"]),
        int(best_row["d"]),
        int(best_row["q"]),
    )

    seasonal_order = (
        int(best_row["P"]),
        int(best_row["D"]),
        int(best_row["Q"]),
        int(best_row["seasonal_period"]),
    )

    return order, seasonal_order


def save_training_summary(
    train_series: pd.Series,
    test_series: pd.Series,
    fitted_model,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> None:
    """Save training and model-selection information."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "algorithm": "SARIMA",
        "implementation": "Statsmodels SARIMAX",
        "dataset": "Statsmodels atmospheric CO2",
        "frequency": FREQUENCY,
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
        "selected_order": list(order),
        "selected_seasonal_order": list(
            seasonal_order
        ),
        "seasonal_period": SEASONAL_PERIOD,
        "aic": float(fitted_model.aic),
        "bic": float(fitted_model.bic),
        "hqic": float(fitted_model.hqic),
        "log_likelihood": float(
            fitted_model.llf
        ),
        "candidate_model_count": len(
            CANDIDATE_CONFIGURATIONS
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
    """Run the complete SARIMA training workflow."""

    full_series = load_series()

    train_series, test_series = split_series(
        full_series
    )

    save_series_data(
        full_series=full_series,
        train_series=train_series,
        test_series=test_series,
    )

    candidate_results = evaluate_candidates(
        train_series
    )

    order, seasonal_order = (
        select_best_configuration(
            candidate_results
        )
    )

    fitted_model = fit_candidate(
        train_series=train_series,
        order=order,
        seasonal_order=seasonal_order,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        fitted_model,
        MODEL_PATH,
    )

    save_training_summary(
        train_series=train_series,
        test_series=test_series,
        fitted_model=fitted_model,
        order=order,
        seasonal_order=seasonal_order,
    )

    print("\nSARIMA training completed.")
    print(f"Training periods: {len(train_series)}")
    print(
        f"Held-out periods: {len(test_series)}"
    )
    print(f"Selected order: {order}")
    print(
        "Selected seasonal order: "
        f"{seasonal_order}"
    )
    print(f"AIC: {fitted_model.aic:.4f}")
    print(f"BIC: {fitted_model.bic:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
