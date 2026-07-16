"""Evaluate the saved Random Forest classification model."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
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

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest.joblib"
)

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "test_data.csv"
)

FIGURES_DIR = (
    BASE_DIR
    / "outputs"
    / "figures"
)

METRICS_DIR = (
    BASE_DIR
    / "outputs"
    / "metrics"
)

PREDICTIONS_DIR = (
    BASE_DIR
    / "outputs"
    / "predictions"
)


def load_artifacts():
    """Load the trained model and held-out test data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Random Forest model not found. "
            "Run train.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test data not found. "
            "Run train.py first."
        )

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
    """Calculate overall and class-specific metrics."""

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predictions,
            )
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
                probabilities,
            )
        ),
    }


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Save metrics in JSON format."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        METRICS_DIR / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )


def save_classification_report(
    y_test: pd.Series,
    predictions,
) -> None:
    """Save the detailed classification report."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_text = classification_report(
        y_test,
        predictions,
        target_names=[
            "malignant",
            "benign",
        ],
        digits=4,
    )

    with (
        METRICS_DIR
        / "classification_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)


def save_confusion_matrix_values(
    y_test: pd.Series,
    predictions,
) -> None:
    """Save raw confusion-matrix values."""

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
        METRICS_DIR
        / "confusion_matrix_values.csv"
    )


def save_predictions(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
    probabilities,
) -> None:
    """Save test predictions and probabilities."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = x_test.copy()

    results["actual_class"] = (
        y_test.to_numpy()
    )

    results["predicted_class"] = (
        predictions
    )

    results[
        "malignant_probability"
    ] = 1 - probabilities

    results[
        "benign_probability"
    ] = probabilities

    results["correct_prediction"] = (
        results["actual_class"]
        == results["predicted_class"]
    )

    results.to_csv(
        PREDICTIONS_DIR
        / "test_predictions.csv",
        index=False,
    )


def save_feature_importance(
    model,
    feature_names,
) -> None:
    """Save feature-importance values and plot."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    importance_table.to_csv(
        METRICS_DIR
        / "feature_importance.csv",
        index=False,
    )

    top_features = (
        importance_table
        .head(15)
        .sort_values(
            by="importance",
            ascending=True,
        )
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        "Top Random Forest Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_figures(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Create and save evaluation figures."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=[
            "malignant",
            "benign",
        ],
    )

    plt.title(
        "Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    RocCurveDisplay.from_estimator(
        model,
        x_test,
        y_test,
    )

    plt.title(
        "Random Forest ROC Curve"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    PrecisionRecallDisplay.from_estimator(
        model,
        x_test,
        y_test,
    )

    plt.title(
        "Random Forest Precision–Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run the complete evaluation workflow."""

    model, x_test, y_test = (
        load_artifacts()
    )

    predictions = model.predict(
        x_test
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
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

    save_feature_importance(
        model=model,
        feature_names=list(
            x_test.columns
        ),
    )

    save_figures(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    print(
        "\nRandom Forest Evaluation"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "malignant",
                "benign",
            ],
            digits=4,
        )
    )

    print("Saved metrics:")

    for metric_name, metric_value in (
        metrics.items()
    ):
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    print(
        "\nEvaluation outputs saved successfully."
    )


if __name__ == "__main__":
    main()