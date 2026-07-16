"""Train and save a Random Forest classification model."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer(as_frame=True)

    features = dataset.data
    target = dataset.target

    return features, target


def build_model() -> RandomForestClassifier:
    """Create the Random Forest classifier."""

    return RandomForestClassifier(
        n_estimators=300,
        criterion="gini",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def save_test_data(
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save held-out data for independent evaluation."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_data = x_test.copy()
    test_data["target"] = y_test.to_numpy()

    test_data.to_csv(
        TEST_DATA_PATH,
        index=False,
    )


def save_training_summary(
    model: RandomForestClassifier,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> None:
    """Save model configuration and training information."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "algorithm": "RandomForestClassifier",
        "training_records": len(x_train),
        "testing_records": len(x_test),
        "feature_count": x_train.shape[1],
        "n_estimators": model.n_estimators,
        "criterion": model.criterion,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "max_features": model.max_features,
        "bootstrap": model.bootstrap,
        "class_weight": model.class_weight,
        "oob_score": float(model.oob_score_),
        "random_state": RANDOM_STATE,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def main() -> None:
    """Run the complete training workflow."""

    features, target = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = build_model()
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

    print("\nRandom Forest training completed.")
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(x_test)}")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Number of features: {x_train.shape[1]}")
    print(f"Out-of-bag score: {model.oob_score_:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()