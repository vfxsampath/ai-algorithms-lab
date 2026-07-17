"""Train and save an Isolation Forest anomaly-detection pipeline."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
CONTAMINATION = 0.05
SYNTHETIC_ANOMALY_COUNT = 12

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
EVALUATION_DATA_PATH = DATA_DIR / "evaluation_data.csv"
MODEL_PATH = MODEL_DIR / "isolation_forest_pipeline.joblib"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"


def load_dataset() -> pd.DataFrame:
    """Load numerical Wine dataset features."""

    dataset = load_wine(as_frame=True)
    return dataset.data


def split_dataset(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create training and held-out normal datasets."""

    train_data, test_normal_data = train_test_split(
        features,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return (
        train_data.reset_index(drop=True),
        test_normal_data.reset_index(drop=True),
    )


def create_synthetic_anomalies(
    train_data: pd.DataFrame,
    test_normal_data: pd.DataFrame,
    anomaly_count: int,
) -> pd.DataFrame:
    """
    Create controlled synthetic anomalies for evaluation only.

    Anomalies are generated from sampled held-out observations by
    strongly perturbing a random subset of standardized feature ranges.
    """

    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    sampled_rows = test_normal_data.sample(
        n=anomaly_count,
        replace=True,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    anomalies = sampled_rows.copy()

    training_means = train_data.mean()
    training_standard_deviations = (
        train_data.std(ddof=0)
        .replace(0, 1.0)
    )

    feature_names = list(train_data.columns)
    features_per_anomaly = max(
        3,
        len(feature_names) // 2,
    )

    for row_index in range(len(anomalies)):
        selected_features = random_generator.choice(
            feature_names,
            size=features_per_anomaly,
            replace=False,
        )

        for feature_name in selected_features:
            direction = random_generator.choice(
                [-1.0, 1.0]
            )

            magnitude = random_generator.uniform(
                4.0,
                7.0,
            )

            anomalies.loc[
                row_index,
                feature_name,
            ] = (
                training_means[feature_name]
                + direction
                * magnitude
                * training_standard_deviations[
                    feature_name
                ]
            )

    return anomalies


def build_pipeline() -> Pipeline:
    """Create scaling and Isolation Forest pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=300,
                    contamination=CONTAMINATION,
                    max_samples="auto",
                    max_features=1.0,
                    bootstrap=False,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_datasets(
    train_data: pd.DataFrame,
    test_normal_data: pd.DataFrame,
    synthetic_anomalies: pd.DataFrame,
) -> None:
    """Save training data and labelled evaluation data."""

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data.to_csv(
        TRAIN_DATA_PATH,
        index=False,
    )

    normal_evaluation = (
        test_normal_data.copy()
    )

    normal_evaluation[
        "actual_is_anomaly"
    ] = 0

    normal_evaluation[
        "observation_source"
    ] = "held_out_normal"

    anomaly_evaluation = (
        synthetic_anomalies.copy()
    )

    anomaly_evaluation[
        "actual_is_anomaly"
    ] = 1

    anomaly_evaluation[
        "observation_source"
    ] = "synthetic_anomaly"

    evaluation_data = pd.concat(
        [
            normal_evaluation,
            anomaly_evaluation,
        ],
        ignore_index=True,
    )

    evaluation_data = evaluation_data.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    evaluation_data.to_csv(
        EVALUATION_DATA_PATH,
        index=False,
    )


def save_training_summary(
    model: Pipeline,
    train_data: pd.DataFrame,
    test_normal_data: pd.DataFrame,
    synthetic_anomalies: pd.DataFrame,
) -> None:
    """Save model settings and data-preparation details."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = model.named_steps["detector"]

    train_predictions = detector.predict(
        model.named_steps["scaler"].transform(
            train_data
        )
    )

    train_flagged_count = int(
        np.sum(train_predictions == -1)
    )

    summary = {
        "algorithm": "Isolation Forest",
        "dataset": "Scikit-learn Wine dataset",
        "training_records": len(train_data),
        "held_out_normal_records": len(
            test_normal_data
        ),
        "synthetic_anomaly_records": len(
            synthetic_anomalies
        ),
        "evaluation_records": (
            len(test_normal_data)
            + len(synthetic_anomalies)
        ),
        "feature_count": train_data.shape[1],
        "n_estimators": detector.n_estimators,
        "contamination": detector.contamination,
        "max_samples": detector.max_samples,
        "max_features": detector.max_features,
        "bootstrap": detector.bootstrap,
        "feature_scaling": "StandardScaler",
        "training_records_flagged_as_outliers": (
            train_flagged_count
        ),
        "training_outlier_percentage": float(
            train_flagged_count
            / len(train_data)
            * 100
        ),
        "random_state": RANDOM_STATE,
        "model_fitted_on_training_data_only": True,
        "held_out_data_used_during_fitting": False,
        "synthetic_anomalies_used_during_fitting": False,
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
    """Run the complete Isolation Forest training workflow."""

    features = load_dataset()

    train_data, test_normal_data = split_dataset(
        features
    )

    synthetic_anomalies = (
        create_synthetic_anomalies(
            train_data=train_data,
            test_normal_data=test_normal_data,
            anomaly_count=SYNTHETIC_ANOMALY_COUNT,
        )
    )

    model = build_pipeline()

    # Only the training dataset is used for fitting.
    model.fit(train_data)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    save_datasets(
        train_data=train_data,
        test_normal_data=test_normal_data,
        synthetic_anomalies=synthetic_anomalies,
    )

    save_training_summary(
        model=model,
        train_data=train_data,
        test_normal_data=test_normal_data,
        synthetic_anomalies=synthetic_anomalies,
    )

    print(
        "\nIsolation Forest training completed."
    )

    print(
        f"Training records: {len(train_data)}"
    )

    print(
        "Held-out normal records: "
        f"{len(test_normal_data)}"
    )

    print(
        "Synthetic evaluation anomalies: "
        f"{len(synthetic_anomalies)}"
    )

    print(
        "Total evaluation records: "
        f"{len(test_normal_data) + len(synthetic_anomalies)}"
    )

    print(
        f"Feature count: {train_data.shape[1]}"
    )

    print(
        f"Contamination: {CONTAMINATION}"
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
