"""Train and save a K-Nearest Neighbors classification model."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

MODEL_PATH = MODEL_DIR / "knn_classifier.joblib"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def build_pipeline() -> Pipeline:
    """Build the scaling and KNN classification pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                    algorithm="auto",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_test_data(
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save held-out test data for independent evaluation."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_data = x_test.copy()
    test_data["target"] = y_test.to_numpy()

    test_data.to_csv(
        TEST_DATA_PATH,
        index=False,
    )


def save_training_summary(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> None:
    """Save training information and model configuration."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    classifier = model.named_steps["classifier"]

    summary = {
        "algorithm": "KNeighborsClassifier",
        "training_records": len(x_train),
        "testing_records": len(x_test),
        "feature_count": x_train.shape[1],
        "n_neighbors": classifier.n_neighbors,
        "weights": classifier.weights,
        "metric": classifier.metric,
        "p": classifier.p,
        "algorithm": classifier.algorithm,
        "feature_scaling": "StandardScaler",
        "random_state_for_split": RANDOM_STATE,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def main() -> None:
    """Run the complete KNN training workflow."""

    features, target = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    save_test_data(
        x_test=x_test,
        y_test=y_test,
    )

    save_training_summary(
        model=model,
        x_train=x_train,
        x_test=x_test,
    )

    print("\nKNN training completed.")
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(x_test)}")
    print("Number of neighbors: 7")
    print("Weights: distance")
    print("Distance metric: Euclidean")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()