"""Evaluate the saved Linear Regression model."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "linear_regression.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"


def load_artifacts():
    """Load the trained model and held-out test data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
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


def calculate_adjusted_r_squared(
    r_squared: float,
    sample_count: int,
    feature_count: int,
) -> float:
    """Calculate adjusted R-squared."""

    denominator = sample_count - feature_count - 1

    if denominator <= 0:
        raise ValueError(
            "Insufficient observations for adjusted R-squared."
        )

    return 1 - (
        (1 - r_squared)
        * (sample_count - 1)
        / denominator
    )


def calculate_metrics(
    y_test: pd.Series,
    predictions: np.ndarray,
    feature_count: int,
) -> dict[str, float]:
    """Calculate regression evaluation metrics."""

    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)

    return {
        "mean_absolute_error": float(
            mean_absolute_error(y_test, predictions)
        ),
        "mean_squared_error": float(mse),
        "root_mean_squared_error": float(np.sqrt(mse)),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(
            calculate_adjusted_r_squared(
                r_squared,
                len(y_test),
                feature_count,
            )
        ),
    }


def save_predictions(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """Save actual values, predictions, and residuals."""

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    results = x_test.copy()
    results["actual"] = y_test.to_numpy()
    results["predicted"] = predictions
    results["residual"] = (
        results["actual"] - results["predicted"]
    )

    results.to_csv(
        PREDICTIONS_DIR / "test_predictions.csv",
        index=False,
    )


def save_metrics(metrics: dict[str, float]) -> None:
    """Save evaluation metrics."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with (
        METRICS_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_figures(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """Create and save regression evaluation figures."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = y_test.to_numpy() - predictions

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5)

    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("Actual vs Predicted Values")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "actual_vs_predicted.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "residual_plot.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "residual_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run the independent model evaluation workflow."""

    model, x_test, y_test = load_artifacts()
    predictions = model.predict(x_test)

    metrics = calculate_metrics(
        y_test=y_test,
        predictions=predictions,
        feature_count=x_test.shape[1],
    )

    save_metrics(metrics)
    save_predictions(x_test, y_test, predictions)
    save_figures(y_test, predictions)

    print("\nLinear Regression Evaluation")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nEvaluation outputs saved successfully.")


if __name__ == "__main__":
    main()