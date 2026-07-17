"""Approximate cluster assignment for a saved DBSCAN model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "dbscan_pipeline.joblib"
)


def load_model():
    """Load the saved DBSCAN pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "DBSCAN model not found. "
            "Run train.py first."
        )

    return joblib.load(
        MODEL_PATH
    )


def validate_input(
    sample: pd.DataFrame,
    expected_features: list[str],
) -> None:
    """Validate new observation features and values."""

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
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric columns: "
            f"{non_numeric_columns}"
        )


def assign_cluster(
    model,
    sample: pd.DataFrame,
) -> dict[str, int | float | str]:
    """Assign a sample to its nearest core cluster or noise."""

    scaler = model.named_steps["scaler"]
    clusterer = model.named_steps["clusterer"]

    if len(clusterer.components_) == 0:
        raise RuntimeError(
            "The trained DBSCAN model has no core samples."
        )

    scaled_sample = scaler.transform(
        sample
    )

    core_samples = clusterer.components_

    core_labels = clusterer.labels_[
        clusterer.core_sample_indices_
    ]

    neighbor_model = NearestNeighbors(
        n_neighbors=1,
        metric=clusterer.metric,
    )

    neighbor_model.fit(
        core_samples
    )

    distances, indices = (
        neighbor_model.kneighbors(
            scaled_sample
        )
    )

    nearest_distance = float(
        distances[0, 0]
    )

    nearest_core_position = int(
        indices[0, 0]
    )

    nearest_cluster = int(
        core_labels[
            nearest_core_position
        ]
    )

    if nearest_distance <= clusterer.eps:
        assigned_cluster = nearest_cluster
        assignment_type = (
            "Approximate cluster assignment"
        )
    else:
        assigned_cluster = -1
        assignment_type = (
            "Noise / outside learned dense regions"
        )

    return {
        "assigned_cluster": assigned_cluster,
        "nearest_core_cluster": nearest_cluster,
        "distance_to_nearest_core_sample": (
            nearest_distance
        ),
        "eps_threshold": float(
            clusterer.eps
        ),
        "assignment_type": assignment_type,
    }


def main() -> None:
    """Assign one example observation approximately."""

    model = load_model()

    dataset = load_wine(
        as_frame=True
    )

    expected_features = list(
        dataset.feature_names
    )

    sample = (
        dataset.data
        .iloc[[0]]
        .copy()
    )

    validate_input(
        sample=sample,
        expected_features=expected_features,
    )

    result = assign_cluster(
        model=model,
        sample=sample,
    )

    print(
        "\nDBSCAN Approximate Assignment"
    )

    print(
        "Assigned cluster: "
        f"{result['assigned_cluster']}"
    )

    print(
        "Nearest core cluster: "
        f"{result['nearest_core_cluster']}"
    )

    print(
        "Distance to nearest core sample: "
        f"{result['distance_to_nearest_core_sample']:.4f}"
    )

    print(
        "EPS threshold: "
        f"{result['eps_threshold']:.4f}"
    )

    print(
        "Assignment type: "
        f"{result['assignment_type']}"
    )

    print(
        "\nImportant: this is an approximate "
        "post-training assignment. Standard DBSCAN "
        "does not provide native prediction for "
        "unseen observations."
    )


if __name__ == "__main__":
    main()
