"""Train and save an XGBoost time-series forecasting model."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.datasets import co2
from xgboost import XGBRegressor


RANDOM_STATE = 42
TEST_PERIODS = 24
VALIDATION_PERIODS = 24
FREQUENCY = "MS"

LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

FULL_DATA_PATH = DATA_DIR / "full_series.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

MODEL_PATH = MODEL_DIR / "xgboost_time_series_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

CANDIDATE_RESULTS_PATH = (
    METRICS_DIR / "candidate_models.csv"
)

FEATURE_IMPORTANCE_PATH = (
    METRICS_DIR / "feature_importance.csv"
)

TRAINING_SUMMARY_PATH = (
    METRICS_DIR / "training_summary.json"
)


CANDIDATE_CONFIGURATIONS = [
    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
    },
    {
        "n_estimators": 600,
        "learning_rate": 0.02,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    },
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
    """Create chronological training and future test sets."""

    if len(series) <= TEST_PERIODS:
        raise ValueError(
            "The series is too short for the configured test horizon."
        )

    train_series = series.iloc[:-TEST_PERIODS].copy()
    test_series = series.iloc[-TEST_PERIODS:].copy()

    return train_series, test_series


def save_series_data(
    full_series: pd.Series,
    train_series: pd.Series,
    test_series: pd.Series,
) -> None:
    """Save the full chronological series and its split."""

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


def create_supervised_features(
    series: pd.Series,
) -> pd.DataFrame:
    """
    Convert historical values into supervised features.

    Every lag and rolling feature uses earlier observations only.
    """

    frame = pd.DataFrame(
        {
            "date": series.index,
            "target": series.to_numpy(),
        }
    )

    frame = frame.set_index("date")

    for lag in LAGS:
        frame[f"lag_{lag}"] = (
            frame["target"].shift(lag)
        )

    shifted_target = frame["target"].shift(1)

    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = (
            shifted_target
            .rolling(window=window)
            .mean()
        )

        frame[f"rolling_std_{window}"] = (
            shifted_target
            .rolling(window=window)
            .std()
        )

        frame[f"rolling_min_{window}"] = (
            shifted_target
            .rolling(window=window)
            .min()
        )

        frame[f"rolling_max_{window}"] = (
            shifted_target
            .rolling(window=window)
            .max()
        )

    frame["month"] = frame.index.month
    frame["quarter"] = frame.index.quarter

    frame["month_sin"] = np.sin(
        2
        * np.pi
        * frame.index.month
        / 12
    )

    frame["month_cos"] = np.cos(
        2
        * np.pi
        * frame.index.month
        / 12
    )

    frame["time_index"] = np.arange(
        len(frame)
    )

    frame = frame.dropna().copy()

    return frame


def feature_columns(
    supervised_data: pd.DataFrame,
) -> list[str]:
    """Return model-input feature names."""

    return [
        column
        for column in supervised_data.columns
        if column != "target"
    ]


def build_model(
    configuration: dict[str, float | int],
) -> XGBRegressor:
    """Build an XGBoost regression model."""

    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **configuration,
    )


def chronological_validation_split(
    supervised_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Reserve the latest training rows for validation."""

    if len(supervised_data) <= VALIDATION_PERIODS:
        raise ValueError(
            "Insufficient feature rows for chronological validation."
        )

    model_training_data = (
        supervised_data
        .iloc[:-VALIDATION_PERIODS]
        .copy()
    )

    validation_data = (
        supervised_data
        .iloc[-VALIDATION_PERIODS:]
        .copy()
    )

    return (
        model_training_data,
        validation_data,
    )


def evaluate_candidates(
    supervised_data: pd.DataFrame,
) -> pd.DataFrame:
    """Select hyperparameters using chronological validation."""

    model_training_data, validation_data = (
        chronological_validation_split(
            supervised_data
        )
    )

    features = feature_columns(
        supervised_data
    )

    x_train = model_training_data[features]
    y_train = model_training_data["target"]

    x_validation = validation_data[features]
    y_validation = validation_data["target"]

    candidate_rows = []

    for candidate_number, configuration in enumerate(
        CANDIDATE_CONFIGURATIONS,
        start=1,
    ):
        model = build_model(configuration)

        model.fit(
            x_train,
            y_train,
            eval_set=[
                (
                    x_validation,
                    y_validation,
                )
            ],
            verbose=False,
        )

        validation_predictions = model.predict(
            x_validation
        )

        validation_mae = mean_absolute_error(
            y_validation,
            validation_predictions,
        )

        validation_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        y_validation.to_numpy()
                        - validation_predictions
                    )
                    ** 2
                )
            )
        )

        row = {
            "candidate_number": candidate_number,
            **configuration,
            "validation_mae": float(
                validation_mae
            ),
            "validation_rmse": validation_rmse,
            "training_rows": len(
                model_training_data
            ),
            "validation_rows": len(
                validation_data
            ),
        }

        candidate_rows.append(row)

    candidate_results = pd.DataFrame(
        candidate_rows
    ).sort_values(
        by=[
            "validation_mae",
            "validation_rmse",
        ],
        ascending=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_results.to_csv(
        CANDIDATE_RESULTS_PATH,
        index=False,
    )

    return candidate_results


def select_best_configuration(
    candidate_results: pd.DataFrame,
) -> dict[str, float | int]:
    """Return parameters from the best validation candidate."""

    best_row = candidate_results.iloc[0]

    parameter_names = list(
        CANDIDATE_CONFIGURATIONS[0].keys()
    )

    configuration: dict[str, float | int] = {}

    integer_parameters = {
        "n_estimators",
        "max_depth",
        "min_child_weight",
    }

    for parameter_name in parameter_names:
        value = best_row[parameter_name]

        if parameter_name in integer_parameters:
            configuration[parameter_name] = int(value)
        else:
            configuration[parameter_name] = float(value)

    return configuration


def save_feature_importance(
    model: XGBRegressor,
    features: list[str],
) -> None:
    """Save built-in feature importance values."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_table = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    importance_table.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )


def save_metadata(
    features: list[str],
    selected_configuration: dict[str, float | int],
    train_series: pd.Series,
) -> None:
    """Save metadata required for evaluation and inference."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "feature_columns": features,
        "lags": LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "frequency": FREQUENCY,
        "training_start": str(
            train_series.index.min().date()
        ),
        "training_end": str(
            train_series.index.max().date()
        ),
        "training_period_count": len(
            train_series
        ),
        "selected_configuration": (
            selected_configuration
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )


def save_training_summary(
    train_series: pd.Series,
    test_series: pd.Series,
    supervised_data: pd.DataFrame,
    candidate_results: pd.DataFrame,
    selected_configuration: dict[str, float | int],
) -> None:
    """Save complete training-workflow information."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_candidate = (
        candidate_results.iloc[0]
    )

    summary = {
        "algorithm": (
            "XGBoost Time-Series Forecasting"
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
        "supervised_training_rows": len(
            supervised_data
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
        "lags": LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "feature_count": len(
            feature_columns(
                supervised_data
            )
        ),
        "candidate_model_count": len(
            candidate_results
        ),
        "validation_periods": (
            VALIDATION_PERIODS
        ),
        "best_validation_mae": float(
            best_candidate[
                "validation_mae"
            ]
        ),
        "best_validation_rmse": float(
            best_candidate[
                "validation_rmse"
            ]
        ),
        "selected_configuration": (
            selected_configuration
        ),
        "chronological_split": True,
        "chronological_validation": True,
        "random_shuffling": False,
        "test_data_used_during_training": False,
        "test_data_used_during_tuning": False,
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
    """Run the complete XGBoost training workflow."""

    full_series = load_series()

    train_series, test_series = split_series(
        full_series
    )

    save_series_data(
        full_series=full_series,
        train_series=train_series,
        test_series=test_series,
    )

    supervised_data = (
        create_supervised_features(
            train_series
        )
    )

    candidate_results = evaluate_candidates(
        supervised_data
    )

    selected_configuration = (
        select_best_configuration(
            candidate_results
        )
    )

    features = feature_columns(
        supervised_data
    )

    final_model = build_model(
        selected_configuration
    )

    final_model.fit(
        supervised_data[features],
        supervised_data["target"],
        verbose=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    save_feature_importance(
        model=final_model,
        features=features,
    )

    save_metadata(
        features=features,
        selected_configuration=(
            selected_configuration
        ),
        train_series=train_series,
    )

    save_training_summary(
        train_series=train_series,
        test_series=test_series,
        supervised_data=supervised_data,
        candidate_results=candidate_results,
        selected_configuration=(
            selected_configuration
        ),
    )

    print(
        "\nXGBoost time-series "
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
        f"Supervised feature rows: "
        f"{len(supervised_data)}"
    )

    print(
        f"Feature count: "
        f"{len(features)}"
    )

    print(
        "\nSelected configuration:"
    )

    for parameter_name, value in (
        selected_configuration.items()
    ):
        print(
            f"{parameter_name}: {value}"
        )

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
