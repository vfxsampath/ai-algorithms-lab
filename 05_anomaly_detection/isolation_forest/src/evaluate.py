"""Evaluate the saved Isolation Forest on held-out data."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
    average_precision_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "isolation_forest_pipeline.joblib"
)

TRAIN_DATA_PATH = (
    BASE_DIR
    / "data"
    / "train_data.csv"
)

EVALUATION_DATA_PATH = (
    BASE_DIR
    / "data"
    / "evaluation_data.csv"
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
    """Load model, training data, and evaluation data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Isolation Forest model not found. "
            "Run train.py first."
        )

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            "Training data not found. "
            "Run train.py first."
        )

    if not EVALUATION_DATA_PATH.exists():
        raise FileNotFoundError(
            "Evaluation data not found. "
            "Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    train_data = pd.read_csv(
        TRAIN_DATA_PATH
    )

    evaluation_data = pd.read_csv(
        EVALUATION_DATA_PATH
    )

    return model, train_data, evaluation_data


def prepare_evaluation_data(
    evaluation_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate features, labels, and source metadata."""

    metadata_columns = [
        "actual_is_anomaly",
        "observation_source",
    ]

    features = evaluation_data.drop(
        columns=metadata_columns
    )

    actual_labels = evaluation_data[
        "actual_is_anomaly"
    ].astype(int)

    observation_sources = evaluation_data[
        "observation_source"
    ]

    return (
        features,
        actual_labels,
        observation_sources,
    )


def generate_outputs(
    model,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions and anomaly scores.

    IsolationForest.predict returns:
    1  = inlier
    -1 = outlier

    This project converts the result to:
    0 = normal
    1 = anomaly
    """

    raw_predictions = model.predict(
        features
    )

    predicted_is_anomaly = np.where(
        raw_predictions == -1,
        1,
        0,
    )

    decision_scores = model.decision_function(
        features
    )

    # Larger values should mean more anomalous
    # for ROC and precision-recall analysis.
    anomaly_scores = -decision_scores

    return (
        raw_predictions,
        predicted_is_anomaly,
        anomaly_scores,
    )


def calculate_metrics(
    actual_labels: pd.Series,
    predicted_labels: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict[str, float | int]:
    """Calculate anomaly-detection performance metrics."""

    return {
        "evaluation_records": int(
            len(actual_labels)
        ),
        "actual_anomaly_count": int(
            actual_labels.sum()
        ),
        "predicted_anomaly_count": int(
            predicted_labels.sum()
        ),
        "accuracy": float(
            accuracy_score(
                actual_labels,
                predicted_labels,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual_labels,
                predicted_labels,
            )
        ),
        "anomaly_precision": float(
            precision_score(
                actual_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            )
        ),
        "anomaly_recall": float(
            recall_score(
                actual_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            )
        ),
        "anomaly_f1": float(
            f1_score(
                actual_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            )
        ),
        "normal_precision": float(
            precision_score(
                actual_labels,
                predicted_labels,
                pos_label=0,
                zero_division=0,
            )
        ),
        "normal_recall": float(
            recall_score(
                actual_labels,
                predicted_labels,
                pos_label=0,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                actual_labels,
                anomaly_scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                actual_labels,
                anomaly_scores,
            )
        ),
    }


def save_metrics(
    metrics: dict[str, float | int],
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
    actual_labels: pd.Series,
    predicted_labels: np.ndarray,
) -> None:
    """Save class-specific evaluation report."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = classification_report(
        actual_labels,
        predicted_labels,
        target_names=[
            "normal",
            "anomaly",
        ],
        digits=4,
        zero_division=0,
    )

    with (
        METRICS_DIR
        / "classification_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)


def save_confusion_matrix_values(
    actual_labels: pd.Series,
    predicted_labels: np.ndarray,
) -> None:
    """Save confusion-matrix values as CSV."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
    )

    matrix_table = pd.DataFrame(
        matrix,
        index=[
            "actual_normal",
            "actual_anomaly",
        ],
        columns=[
            "predicted_normal",
            "predicted_anomaly",
        ],
    )

    matrix_table.to_csv(
        METRICS_DIR
        / "confusion_matrix_values.csv"
    )


def save_score_summary(
    actual_labels: pd.Series,
    anomaly_scores: np.ndarray,
) -> None:
    """Save anomaly-score summaries by actual class."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_table = pd.DataFrame(
        {
            "actual_is_anomaly": (
                actual_labels.to_numpy()
            ),
            "anomaly_score": anomaly_scores,
        }
    )

    summary = (
        score_table
        .groupby("actual_is_anomaly")[
            "anomaly_score"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
    )

    summary.index = [
        "normal",
        "anomaly",
    ]

    summary.to_csv(
        METRICS_DIR
        / "anomaly_score_summary.csv"
    )


def save_predictions(
    features: pd.DataFrame,
    actual_labels: pd.Series,
    observation_sources: pd.Series,
    raw_predictions: np.ndarray,
    predicted_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    model,
) -> None:
    """Save all evaluation predictions and scores."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision_scores = model.decision_function(
        features
    )

    score_samples = model.score_samples(
        features
    )

    results = features.copy()

    results.insert(
        0,
        "observation_source",
        observation_sources.to_numpy(),
    )

    results.insert(
        1,
        "actual_is_anomaly",
        actual_labels.to_numpy(),
    )

    results.insert(
        2,
        "predicted_is_anomaly",
        predicted_labels,
    )

    results.insert(
        3,
        "raw_isolation_forest_prediction",
        raw_predictions,
    )

    results.insert(
        4,
        "decision_function",
        decision_scores,
    )

    results.insert(
        5,
        "anomaly_score",
        anomaly_scores,
    )

    results.insert(
        6,
        "score_samples",
        score_samples,
    )

    results.insert(
        7,
        "correct_prediction",
        (
            actual_labels.to_numpy()
            == predicted_labels
        ),
    )

    results = results.sort_values(
        by="anomaly_score",
        ascending=False,
    )

    results.to_csv(
        PREDICTIONS_DIR
        / "evaluation_predictions.csv",
        index=False,
    )


def save_confusion_matrix_plot(
    actual_labels: pd.Series,
    predicted_labels: np.ndarray,
) -> None:
    """Save anomaly-detection confusion matrix."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ConfusionMatrixDisplay.from_predictions(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        display_labels=[
            "normal",
            "anomaly",
        ],
    )

    plt.title(
        "Isolation Forest Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_roc_curve(
    actual_labels: pd.Series,
    anomaly_scores: np.ndarray,
) -> None:
    """Save ROC curve from anomaly scores."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RocCurveDisplay.from_predictions(
        actual_labels,
        anomaly_scores,
        pos_label=1,
    )

    plt.title(
        "Isolation Forest ROC Curve"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_precision_recall_curve(
    actual_labels: pd.Series,
    anomaly_scores: np.ndarray,
) -> None:
    """Save anomaly precision-recall curve."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PrecisionRecallDisplay.from_predictions(
        actual_labels,
        anomaly_scores,
        pos_label=1,
    )

    plt.title(
        "Isolation Forest "
        "Precision–Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "precision_recall_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_anomaly_score_distribution(
    actual_labels: pd.Series,
    anomaly_scores: np.ndarray,
) -> None:
    """Compare anomaly-score distributions."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normal_scores = anomaly_scores[
        actual_labels.to_numpy() == 0
    ]

    anomaly_values = anomaly_scores[
        actual_labels.to_numpy() == 1
    ]

    plt.figure(figsize=(9, 6))

    plt.hist(
        normal_scores,
        bins=20,
        alpha=0.65,
        label="Held-out normal",
    )

    plt.hist(
        anomaly_values,
        bins=20,
        alpha=0.65,
        label="Synthetic anomaly",
    )

    plt.axvline(
        0.0,
        linestyle="--",
        label="Decision boundary",
    )

    plt.xlabel(
        "Anomaly Score "
        "(larger = more anomalous)"
    )

    plt.ylabel("Frequency")

    plt.title(
        "Isolation Forest "
        "Anomaly-Score Distribution"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "anomaly_score_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_pca_anomaly_plot(
    model,
    train_data: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    actual_labels: pd.Series,
    predicted_labels: np.ndarray,
) -> None:
    """Visualize train and evaluation data in two PCA dimensions."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler = model.named_steps["scaler"]

    scaled_train = scaler.transform(
        train_data
    )

    scaled_evaluation = scaler.transform(
        evaluation_features
    )

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    pca.fit(scaled_train)

    reduced_train = pca.transform(
        scaled_train
    )

    reduced_evaluation = pca.transform(
        scaled_evaluation
    )

    plt.figure(figsize=(10, 7))

    plt.scatter(
        reduced_train[:, 0],
        reduced_train[:, 1],
        alpha=0.35,
        label="Training observations",
    )

    normal_mask = (
        actual_labels.to_numpy() == 0
    )

    anomaly_mask = (
        actual_labels.to_numpy() == 1
    )

    plt.scatter(
        reduced_evaluation[
            normal_mask,
            0,
        ],
        reduced_evaluation[
            normal_mask,
            1,
        ],
        marker="o",
        label="Held-out normal",
    )

    plt.scatter(
        reduced_evaluation[
            anomaly_mask,
            0,
        ],
        reduced_evaluation[
            anomaly_mask,
            1,
        ],
        marker="x",
        s=90,
        label="Synthetic anomaly",
    )

    predicted_anomaly_mask = (
        predicted_labels == 1
    )

    plt.scatter(
        reduced_evaluation[
            predicted_anomaly_mask,
            0,
        ],
        reduced_evaluation[
            predicted_anomaly_mask,
            1,
        ],
        facecolors="none",
        s=180,
        label="Predicted anomaly",
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(
        "Isolation Forest Evaluation "
        "in PCA Space"
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "pca_anomaly_visualization.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run complete held-out anomaly evaluation."""

    model, train_data, evaluation_data = (
        load_artifacts()
    )

    (
        evaluation_features,
        actual_labels,
        observation_sources,
    ) = prepare_evaluation_data(
        evaluation_data
    )

    (
        raw_predictions,
        predicted_labels,
        anomaly_scores,
    ) = generate_outputs(
        model=model,
        features=evaluation_features,
    )

    metrics = calculate_metrics(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        anomaly_scores=anomaly_scores,
    )

    save_metrics(metrics)

    save_classification_report(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )

    save_confusion_matrix_values(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )

    save_score_summary(
        actual_labels=actual_labels,
        anomaly_scores=anomaly_scores,
    )

    save_predictions(
        features=evaluation_features,
        actual_labels=actual_labels,
        observation_sources=observation_sources,
        raw_predictions=raw_predictions,
        predicted_labels=predicted_labels,
        anomaly_scores=anomaly_scores,
        model=model,
    )

    save_confusion_matrix_plot(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )

    save_roc_curve(
        actual_labels=actual_labels,
        anomaly_scores=anomaly_scores,
    )

    save_precision_recall_curve(
        actual_labels=actual_labels,
        anomaly_scores=anomaly_scores,
    )

    save_anomaly_score_distribution(
        actual_labels=actual_labels,
        anomaly_scores=anomaly_scores,
    )

    save_pca_anomaly_plot(
        model=model,
        train_data=train_data,
        evaluation_features=evaluation_features,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )

    print(
        "\nIsolation Forest Evaluation"
    )

    print(
        classification_report(
            actual_labels,
            predicted_labels,
            target_names=[
                "normal",
                "anomaly",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print("Evaluation metrics:")

    for metric_name, metric_value in (
        metrics.items()
    ):
        if isinstance(metric_value, float):
            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )
        else:
            print(
                f"{metric_name}: "
                f"{metric_value}"
            )

    print(
        "\nEvaluation outputs saved successfully."
    )


if __name__ == "__main__":
    main()
