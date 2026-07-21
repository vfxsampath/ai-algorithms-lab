"""Select, fit, and save an exponential-smoothing forecasting model."""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from statsmodels.datasets import co2
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    Holt,
    SimpleExpSmoothing,
)


TEST_PERIODS = 24
FREQUENCY = "MS"
SEASONAL_PERIODS = 12

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

FULL_DATA_PATH = DATA_DIR / "full_series.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

MODEL_PATH = (
    MODEL_DIR
    / "exponential_smoothing_model.joblib"
)

MODEL_SELECTION_PATH = (
    METRICS_DIR
    / "candidate_models.csv"
)

TRAINING_SUMMARY_PATH = (
    METRICS_DIR
    / "training_summary.json"
)

SMOOTHING_PARAMETERS_PATH = (
    METRICS_DIR
    / "smoothing_parameters.json"
)


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
    """Split data chronologically into training and future test periods."""

    if len(series) <= TEST_PERIODS:
        raise ValueError(
            "Series is too short for the configured test horizon."
        )

    train_series = (
        series
        .iloc[:-TEST_PERIODS]
        .copy()
    )

    test_series = (
        series
        .iloc[-TEST_PERIODS:]
        .copy()
    )

    return train_series, test_series


def save_series_data(
    full_series: pd.Series,
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Save full, training, and held-out future series."""

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def fit_simple_exponential_smoothing(
    train_series: pd.Series,
):
    """Fit level-only simple exponential smoothing."""

    model = SimpleExpSmoothing(
        train_series,
        initialization_method="estimated",
    )

    return model.fit(
        optimized=True,
    )


def fit_holt_linear(
    train_series: pd.Series,
):
    """Fit Holt's linear-trend model."""

    model = Holt(
        train_series,
        damped_trend=False,
        initialization_method="estimated",
    )

    return model.fit(
        optimized=True,
    )


def fit_holt_damped(
    train_series: pd.Series,
):
    """Fit Holt's damped linear-trend model."""

    model = Holt(
        train_series,
        damped_trend=True,
        initialization_method="estimated",
    )

    return model.fit(
        optimized=True,
    )


def fit_holt_winters_additive(
    train_series: pd.Series,
):
    """Fit additive-trend, additive-seasonality Holt-Winters."""

    model = ExponentialSmoothing(
        train_series,
        trend="add",
        damped_trend=False,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated",
    )

    return model.fit(
        optimized=True,
        remove_bias=False,
    )


def fit_holt_winters_multiplicative(
    train_series: pd.Series,
):
    """Fit additive-trend, multiplicative-seasonality Holt-Winters."""

    if (train_series <= 0).any():
        raise ValueError(
            "Multiplicative seasonality requires strictly positive values."
        )

    model = ExponentialSmoothing(
        train_series,
        trend="add",
        damped_trend=False,
        seasonal="mul",
        seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated",
    )

    return model.fit(
        optimized=True,
        remove_bias=False,
    )


def candidate_fitters() -> dict[str, object]:
    """Return candidate model names and fitting functions."""

    return {
        "simple_exponential_smoothing": (
            fit_simple_exponential_smoothing
        ),
        "holt_linear_trend": (
            fit_holt_linear
        ),
        "holt_damped_trend": (
            fit_holt_damped
        ),
        "holt_winters_additive": (
            fit_holt_winters_additive
        ),
        "holt_winters_multiplicative": (
            fit_holt_winters_multiplicative
        ),
    }


def safe_float(value: object) -> float:
    """Convert model output to float or NaN."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_candidates(
    train_series: pd.Series,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    """Fit candidate models and compare information criteria."""

    rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    for model_name, fitting_function in (
        candidate_fitters().items()
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fitted_model = fitting_function(
                    train_series
                )

            fitted_models[model_name] = (
                fitted_model
            )

            rows.append(
                {
                    "model_name": model_name,
                    "aic": safe_float(
                        fitted_model.aic
                    ),
                    "aicc": safe_float(
                        fitted_model.aicc
                    ),
                    "bic": safe_float(
                        fitted_model.bic
                    ),
                    "sse": safe_float(
                        fitted_model.sse
                    ),
                    "optimized": bool(
                        np.any(
                            fitted_model.optimized
                        )
                    ),
                    "status": "success",
                    "error": "",
                }
            )

        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            rows.append(
                {
                    "model_name": model_name,
                    "aic": np.nan,
                    "aicc": np.nan,
                    "bic": np.nan,
                    "sse": np.nan,
                    "optimized": False,
                    "status": "failed",
                    "error": str(error),
                }
            )

    candidate_results = pd.DataFrame(rows)

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_results.to_csv(
        MODEL_SELECTION_PATH,
        index=False,
    )

    successful = candidate_results[
        candidate_results["status"]
        == "success"
    ]

    if successful.empty:
        failure_details = "; ".join(
            f"{row.model_name}: {row.error}"
            for row in candidate_results.itertuples()
        )

        raise RuntimeError(
            "All exponential-smoothing candidates failed. "
            f"Details: {failure_details}. "
            f"Diagnostics saved to: {MODEL_SELECTION_PATH}"
        )

    candidate_results = (
        candidate_results
        .sort_values(
            by=[
                "status",
                "aicc",
                "aic",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            na_position="last",
        )
    )

    candidate_results.to_csv(
        MODEL_SELECTION_PATH,
        index=False,
    )

    return (
        candidate_results,
        fitted_models,
    )


def select_best_model_name(
    candidate_results: pd.DataFrame,
) -> str:
    """Select the successful model with the lowest AICc."""

    successful = candidate_results[
        candidate_results["status"]
        == "success"
    ].copy()

    valid_aicc = successful.dropna(
        subset=["aicc"]
    )

    if not valid_aicc.empty:
        best_index = valid_aicc[
            "aicc"
        ].idxmin()
    else:
        best_index = successful[
            "aic"
        ].idxmin()

    return str(
        successful.loc[
            best_index,
            "model_name",
        ]
    )


def serialize_parameter(
    value: object,
) -> object:
    """Convert fitted parameters to JSON-safe values."""

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    ):
        return value

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def save_smoothing_parameters(
    fitted_model,
) -> None:
    """Save the fitted smoothing parameters."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters = {
        str(name): serialize_parameter(value)
        for name, value in (
            fitted_model.params.items()
        )
    }

    with SMOOTHING_PARAMETERS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            parameters,
            file,
            indent=4,
        )


def save_training_summary(
    train_series: pd.Series,
    test_series: pd.Series,
    selected_model_name: str,
    fitted_model,
    candidate_results: pd.DataFrame,
) -> None:
    """Save selected-model and training information."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_configuration = {
        "trend": (
            fitted_model.model.trend
        ),
        "damped_trend": bool(
            fitted_model.model.damped_trend
        ),
        "seasonal": (
            fitted_model.model.seasonal
        ),
        "seasonal_periods": (
            fitted_model.model.seasonal_periods
        ),
    }

    summary = {
        "algorithm": (
            "Exponential Smoothing"
        ),
        "selected_model": (
            selected_model_name
        ),
        "dataset": (
            "Statsmodels atmospheric CO2"
        ),
        "frequency": FREQUENCY,
        "training_periods": len(
            train_series
        ),
        "held_out_test_periods": len(
            test_series
        ),
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
        "seasonal_periods": (
            SEASONAL_PERIODS
        ),
        "model_configuration": (
            model_configuration
        ),
        "aic": safe_float(
            fitted_model.aic
        ),
        "aicc": safe_float(
            fitted_model.aicc
        ),
        "bic": safe_float(
            fitted_model.bic
        ),
        "sse": safe_float(
            fitted_model.sse
        ),
        "candidate_model_count": int(
            len(candidate_results)
        ),
        "successful_candidate_count": int(
            (
                candidate_results["status"]
                == "success"
            ).sum()
        ),
        "chronological_split": True,
        "random_shuffling": False,
        "test_data_used_during_fitting": False,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )


def main() -> None:
    """Run the complete model-selection and training workflow."""

    full_series = load_series()

    train_series, test_series = (
        split_series(full_series)
    )

    save_series_data(
        full_series=full_series,
        train_series=train_series,
        test_series=test_series,
    )

    (
        candidate_results,
        fitted_models,
    ) = evaluate_candidates(
        train_series
    )

    selected_model_name = (
        select_best_model_name(
            candidate_results
        )
    )

    fitted_model = fitted_models[
        selected_model_name
    ]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        fitted_model,
        MODEL_PATH,
    )

    save_smoothing_parameters(
        fitted_model
    )

    save_training_summary(
        train_series=train_series,
        test_series=test_series,
        selected_model_name=(
            selected_model_name
        ),
        fitted_model=fitted_model,
        candidate_results=candidate_results,
    )

    print(
        "\nExponential Smoothing "
        "training completed."
    )

    print(
        f"Training periods: "
        f"{len(train_series)}"
    )

    print(
        f"Held-out periods: "
        f"{len(test_series)}"
    )

    print(
        f"Selected model: "
        f"{selected_model_name}"
    )

    print(
        f"AIC: "
        f"{safe_float(fitted_model.aic):.4f}"
    )

    print(
        f"AICc: "
        f"{safe_float(fitted_model.aicc):.4f}"
    )

    print(
        f"BIC: "
        f"{safe_float(fitted_model.bic):.4f}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
