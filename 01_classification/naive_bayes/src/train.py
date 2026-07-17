"""Train and save a Gaussian Naive Bayes classification model."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

MODEL_PATH = MODEL_DIR / "gaussian_naive_bayes.joblib"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"
CLASS_PARAMETERS_PATH = METRICS_DIR / "class_parameters.csv"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer(as_frame=True)

    features = dataset.data
    target = dataset.target

    return features, target


def build_model() -> GaussianNB:
    """Create the Gaussian Naive Bayes classifier."""

    return GaussianNB(
        var_smoothing=1e-9,
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
    model: GaussianNB,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> None:
    """Save training details and model configuration."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "algorithm": "Gaussian Naive Bayes",
        "training_records": len(x_train),
        "testing_records": len(x_test),
        "feature_count": x_train.shape[1],
        "classes": model.classes_.tolist(),
        "class_counts": model.class_count_.tolist(),
        "class_prior": model.class_prior_.tolist(),
        "var_smoothing": model.var_smoothing,
        "random_state_for_split": RANDOM_STATE,
        "feature_scaling": "Not applied",
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def save_class_parameters(
    model: GaussianNB,
    feature_names: list[str],
) -> None:
    """Save class-specific means and variances."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for class_position, class_value in enumerate(model.classes_):
        for feature_position, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "class_value": int(class_value),
                    "feature": feature_name,
                    "mean": float(
                        model.theta_[
                            class_position,
                            feature_position,
                        ]
                    ),
                    "variance": float(
                        model.var_[
                            class_position,
                            feature_position,
                        ]
                    ),
                }
            )

    parameter_table = pd.DataFrame(rows)

    parameter_table.to_csv(
        CLASS_PARAMETERS_PATH,
        index=False,
    )


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

    save_class_parameters(
        model=model,
        feature_names=list(features.columns),
    )

    print("\nGaussian Naive Bayes training completed.")
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(x_test)}")
    print(f"Feature count: {x_train.shape[1]}")
    print(f"Class priors: {model.class_prior_}")
    print(f"Variance smoothing: {model.var_smoothing}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()