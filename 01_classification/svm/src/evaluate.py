"""Evaluate the saved calibrated Support Vector Machine classifier."""

from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "svm_classifier.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"


def load_artifacts():
    """Load the trained pipeline and held-out test data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "SVM model not found. Run train.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test data not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    x_test = test_data.drop(columns="target")
    y_test = test_data["target"]

    return model, x_test, y_test


def calculate_metrics(
    y_test: pd.Series,
    predictions: np.ndarray,
    benign_probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate overall and class-specific classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_benign": float(
            precision_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_benign": float(
            recall_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_benign": float(
            f1_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_malignant": float(
            precision_score(
                y_test,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_malignant": float(
            recall_score(
                y_test,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_malignant": float(
            f1_score(
                y_test,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                benign_probabilities,
            )
        ),
    }


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Save evaluation metrics as JSON."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with (
        METRICS_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_classification_report(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """Save detailed classification metrics."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_test,
        predictions,
        target_names=["malignant", "benign"],
        digits=4,
    )

    with (
        METRICS_DIR / "classification_report.txt"
    ).open("w", encoding="utf-8") as file:
        file.write(report)


def save_confusion_matrix_values(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """Save raw confusion-matrix values."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    matrix_table = pd.DataFrame(
        matrix,
        index=[
            "actual_malignant",
            "actual_benign",
        ],
        columns=[
            "predicted_malignant",
            "predicted_benign",
        ],
    )

    matrix_table.to_csv(
        METRICS_DIR / "confusion_matrix_values.csv"
    )


def save_predictions(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Save predictions and class probabilities."""

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    results = x_test.copy()

    results["actual_class"] = y_test.to_numpy()
    results["predicted_class"] = predictions
    results["malignant_probability"] = probabilities[:, 0]
    results["benign_probability"] = probabilities[:, 1]

    results["prediction_confidence"] = np.max(
        probabilities,
        axis=1,
    )

    results["correct_prediction"] = (
        results["actual_class"]
        == results["predicted_class"]
    )

    results.to_csv(
        PREDICTIONS_DIR / "test_predictions.csv",
        index=False,
    )


def save_figures(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Create and save evaluation charts."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=["malignant", "benign"],
    )

    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    RocCurveDisplay.from_estimator(
        model,
        x_test,
        y_test,
        pos_label=1,
    )

    plt.title("SVM ROC Curve")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    PrecisionRecallDisplay.from_estimator(
        model,
        x_test,
        y_test,
        pos_label=1,
    )

    plt.title("SVM Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    CalibrationDisplay.from_estimator(
        model,
        x_test,
        y_test,
        n_bins=10,
        strategy="uniform",
        pos_label=1,
    )

    plt.title("SVM Probability Calibration")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "calibration_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run the complete SVM evaluation workflow."""

    model, x_test, y_test = load_artifacts()

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    benign_probabilities = probabilities[:, 1]

    metrics = calculate_metrics(
        y_test=y_test,
        predictions=predictions,
        benign_probabilities=benign_probabilities,
    )

    save_metrics(metrics)

    save_classification_report(
        y_test=y_test,
        predictions=predictions,
    )

    save_confusion_matrix_values(
        y_test=y_test,
        predictions=predictions,
    )

    save_predictions(
        x_test=x_test,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
    )

    save_figures(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    print("\nSVM Evaluation")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["malignant", "benign"],
            digits=4,
        )
    )

    print("Evaluation metrics:")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nEvaluation outputs saved successfully.")


if __name__ == "__main__":
    main()
