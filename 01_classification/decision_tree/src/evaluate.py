"""Evaluate the saved Decision Tree classification model."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import plot_tree


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "decision_tree.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"


def load_artifacts():
    """Load model and test data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run train.py first.")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError("Test data not found. Run train.py first.")

    model = joblib.load(MODEL_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    x_test = test_data.drop(columns="target")
    y_test = test_data["target"]

    return model, x_test, y_test


def calculate_metrics(
    y_test: pd.Series,
    predictions,
    probabilities,
) -> dict[str, float]:
    """Calculate classification metrics."""

    return {
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
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }


def save_metrics(metrics: dict[str, float]) -> None:
    """Save metrics as JSON."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with (
        METRICS_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_predictions(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
    probabilities,
) -> None:
    """Save prediction results."""

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    results = x_test.copy()
    results["actual"] = y_test.to_numpy()
    results["predicted"] = predictions
    results["positive_class_probability"] = probabilities

    results.to_csv(
        PREDICTIONS_DIR / "test_predictions.csv",
        index=False,
    )


def save_figures(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save evaluation figures and tree visualization."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
    )
    plt.title("Decision Tree Confusion Matrix")
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
    )
    plt.title("Decision Tree ROC Curve")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    feature_importance = pd.Series(
        model.feature_importances_,
        index=x_test.columns,
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    feature_importance.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Decision Tree Feature Importances")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=list(x_test.columns),
        class_names=["malignant", "benign"],
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "decision_tree_structure.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run model evaluation."""

    model, x_test, y_test = load_artifacts()

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    save_metrics(metrics)
    save_predictions(
        x_test,
        y_test,
        predictions,
        probabilities,
    )
    save_figures(model, x_test, y_test)

    print("\nDecision Tree Evaluation")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
