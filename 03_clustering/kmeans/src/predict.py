"""Assign a new observation to a saved K-Means cluster."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "kmeans_pipeline.joblib"
)

CENTROIDS_PATH = (
    BASE_DIR
    / "outputs"
    / "metrics"
    / "cluster_centroids.csv"
)


def load_model():
    """Load the saved K-Means pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "K-Means model not found. "
            "Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def validate_input(
    sample: pd.DataFrame,
    expected_features: list[str],
) -> None:
    """Validate new observation feature names and values."""

    missing_features = (
        set(expected_features)
        - set(sample.columns)
    )

    unexpected_features = (
        set(sample.columns)
        - set(expected_features)
    )

    if missing_features:
        raise ValueError(
            "Missing features: "
            f"{sorted(missing_features)}"
        )

    if unexpected_features:
        raise ValueError(
            "Unexpected features: "
            f"{sorted(unexpected_features)}"
        )

    if sample.isnull().any().any():
        raise ValueError(
            "Input contains missing values."
        )

    non_numeric_columns = (
        sample
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric columns: "
            f"{non_numeric_columns}"
        )


def main() -> None:
    """Assign one example observation to a cluster."""

    model = load_model()

    dataset = load_wine(as_frame=True)

    feature_names = list(
        dataset.feature_names
    )

    sample = (
        dataset.data
        .iloc[[0]]
        .copy()
    )

    validate_input(
        sample=sample,
        expected_features=feature_names,
    )

    predicted_cluster = int(
        model.predict(sample)[0]
    )

    scaler = model.named_steps["scaler"]
    clusterer = model.named_steps["clusterer"]

    scaled_sample = scaler.transform(sample)

    distances = np.linalg.norm(
        clusterer.cluster_centers_
        - scaled_sample,
        axis=1,
    )

    nearest_distance = float(
        distances[predicted_cluster]
    )

    print("\nK-Means Cluster Assignment")
    print(
        f"Assigned cluster: "
        f"{predicted_cluster}"
    )
    print(
        "Distance to assigned centroid: "
        f"{nearest_distance:.4f}"
    )

    print("\nDistances to all centroids:")

    for cluster_number, distance in enumerate(
        distances
    ):
        print(
            f"Cluster {cluster_number}: "
            f"{distance:.4f}"
        )

    if CENTROIDS_PATH.exists():
        centroids = pd.read_csv(
            CENTROIDS_PATH
        )

        assigned_centroid = centroids[
            centroids["cluster"]
            == predicted_cluster
        ]

        print(
            "\nAssigned cluster centroid "
            "in original feature units:"
        )

        print(
            assigned_centroid.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
