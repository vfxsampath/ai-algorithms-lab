"""Train and save a Decision Tree classification model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def build_model() -> DecisionTreeClassifier:
    """Create the Decision Tree classifier."""

    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
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
        DATA_DIR / "test_data.csv",
        index=False,
    )


def main() -> None:
    """Train and save the Decision Tree model."""

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
        MODEL_DIR / "decision_tree.joblib",
    )

    save_test_data(x_test, y_test)

    print("Decision Tree training completed.")
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(x_test)}")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")


if __name__ == "__main__":
    main()