"""Train and save a calibrated Support Vector Machine classifier."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

MODEL_PATH = MODEL_DIR / "svm_classifier.joblib"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer(as_frame=True)

    features = dataset.data
    target = dataset.target

    return features, target


def build_pipeline() -> Pipeline:
    """Build a scaling and calibrated SVM classification pipeline."""

    base_svm = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=5,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", calibrated_svm),
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
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> None:
    """Save model settings and training information."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "algorithm": "Support Vector Machine",
        "estimator": "SVC with CalibratedClassifierCV",
        "kernel": "rbf",
        "C": 2.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability_calibration": "sigmoid",
        "calibration_folds": 5,
        "feature_scaling": "StandardScaler",
        "training_records": len(x_train),
        "testing_records": len(x_test),
        "feature_count": x_train.shape[1],
        "random_state_for_split": RANDOM_STATE,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def main() -> None:
    """Run the complete SVM training workflow."""

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
        x_train=x_train,
        x_test=x_test,
    )

    print("\nSVM training completed.")
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(x_test)}")
    print(f"Number of features: {x_train.shape[1]}")
    print("Kernel: RBF")
    print("C: 2.0")
    print("Gamma: scale")
    print("Probability calibration: sigmoid with 5 folds")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
