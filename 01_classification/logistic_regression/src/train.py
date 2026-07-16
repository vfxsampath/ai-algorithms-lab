"""Train and evaluate a Logistic Regression classification model."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer Wisconsin dataset."""

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    return features, target


def build_pipeline() -> Pipeline:
    """Create a preprocessing and Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate the trained classification model."""

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    return metrics


def save_visualizations(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save confusion-matrix and ROC-curve visualizations."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title("Logistic Regression ROC Curve")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_metrics(metrics: dict[str, float]) -> None:
    """Save evaluation metrics as JSON."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = METRICS_DIR / "metrics.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def main() -> None:
    """Run the complete training and evaluation workflow."""

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

    metrics = evaluate_model(model, x_test, y_test)
    save_visualizations(model, x_test, y_test)
    save_metrics(metrics)

    print("\nSaved Metrics")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()