"""Train and save a DBSCAN clustering pipeline."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 2.3
MIN_SAMPLES = 5

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

MODEL_PATH = MODEL_DIR / "dbscan_pipeline.joblib"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
CLUSTERED_DATA_PATH = DATA_DIR / "clustered_data.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"


def load_dataset() -> pd.DataFrame:
    """Load Wine dataset features without target labels."""

    dataset = load_wine(as_frame=True)
    return dataset.data


def build_pipeline() -> Pipeline:
    """Build the scaling and DBSCAN clustering pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clusterer",
                DBSCAN(
                    eps=EPS,
                    min_samples=MIN_SAMPLES,
                    metric="euclidean",
                    algorithm="auto",
                    leaf_size=30,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_training_data(features: pd.DataFrame) -> None:
    """Save the original numerical feature data."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    features.to_csv(
        TRAINING_DATA_PATH,
        index=False,
    )


def save_clustered_data(
    features: pd.DataFrame,
    cluster_labels: np.ndarray,
    core_sample_mask: np.ndarray,
) -> None:
    """Save features, cluster labels, and point types."""

    clustered_data = features.copy()
    clustered_data["cluster"] = cluster_labels
    clustered_data["is_noise"] = cluster_labels == -1
    clustered_data["is_core_sample"] = core_sample_mask

    point_types = np.full(
        len(clustered_data),
        "border",
        dtype=object,
    )

    point_types[cluster_labels == -1] = "noise"
    point_types[core_sample_mask] = "core"

    clustered_data["point_type"] = point_types

    clustered_data.to_csv(
        CLUSTERED_DATA_PATH,
        index=False,
    )


def save_training_summary(
    features: pd.DataFrame,
    clusterer: DBSCAN,
    cluster_labels: np.ndarray,
) -> None:
    """Save model settings and clustering summary."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    unique_clusters = set(cluster_labels)
    cluster_count = len(
        unique_clusters - {-1}
    )

    noise_count = int(
        np.sum(cluster_labels == -1)
    )

    core_sample_count = len(
        clusterer.core_sample_indices_
    )

    summary = {
        "algorithm": "DBSCAN",
        "dataset": "Scikit-learn Wine dataset",
        "training_records": len(features),
        "feature_count": features.shape[1],
        "eps": clusterer.eps,
        "min_samples": clusterer.min_samples,
        "metric": clusterer.metric,
        "algorithm_setting": clusterer.algorithm,
        "leaf_size": clusterer.leaf_size,
        "feature_scaling": "StandardScaler",
        "cluster_count_excluding_noise": cluster_count,
        "noise_count": noise_count,
        "noise_percentage": float(
            noise_count / len(features) * 100
        ),
        "core_sample_count": core_sample_count,
        "border_sample_count": int(
            len(features)
            - noise_count
            - core_sample_count
        ),
        "target_labels_used_for_training": False,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )


def main() -> None:
    """Run the complete DBSCAN training workflow."""

    features = load_dataset()
    model = build_pipeline()

    cluster_labels = model.fit_predict(features)

    clusterer = model.named_steps["clusterer"]

    core_sample_mask = np.zeros(
        len(features),
        dtype=bool,
    )

    core_sample_mask[
        clusterer.core_sample_indices_
    ] = True

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    save_training_data(features)

    save_clustered_data(
        features=features,
        cluster_labels=cluster_labels,
        core_sample_mask=core_sample_mask,
    )

    save_training_summary(
        features=features,
        clusterer=clusterer,
        cluster_labels=cluster_labels,
    )

    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
    )

    cluster_count = len(
        set(cluster_labels) - {-1}
    )

    noise_count = int(
        np.sum(cluster_labels == -1)
    )

    print("\nDBSCAN training completed.")
    print(f"Training records: {len(features)}")
    print(f"Feature count: {features.shape[1]}")
    print(f"EPS: {EPS}")
    print(f"Minimum samples: {MIN_SAMPLES}")
    print(
        "Clusters excluding noise: "
        f"{cluster_count}"
    )
    print(f"Noise observations: {noise_count}")
    print(
        "Core observations: "
        f"{len(clusterer.core_sample_indices_)}"
    )

    print("\nCluster assignments:")

    for cluster_label, count in cluster_counts.items():
        label_name = (
            "Noise"
            if cluster_label == -1
            else f"Cluster {cluster_label}"
        )

        print(
            f"{label_name}: "
            f"{count} observations"
        )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
