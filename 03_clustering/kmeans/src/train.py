"""Train and save a K-Means clustering pipeline."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_CLUSTERS = 3

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

MODEL_PATH = MODEL_DIR / "kmeans_pipeline.joblib"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
CLUSTERED_DATA_PATH = DATA_DIR / "clustered_data.csv"
TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"


def load_dataset() -> pd.DataFrame:
    """Load Wine dataset features without using target labels."""

    dataset = load_wine(as_frame=True)
    return dataset.data


def build_pipeline() -> Pipeline:
    """Build scaling and K-Means clustering pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clusterer",
                KMeans(
                    n_clusters=N_CLUSTERS,
                    init="k-means++",
                    n_init=20,
                    max_iter=300,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_training_data(features: pd.DataFrame) -> None:
    """Save the original feature data."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    features.to_csv(
        TRAINING_DATA_PATH,
        index=False,
    )


def save_clustered_data(
    features: pd.DataFrame,
    cluster_labels,
) -> None:
    """Save features with assigned cluster numbers."""

    clustered_data = features.copy()
    clustered_data["cluster"] = cluster_labels

    clustered_data.to_csv(
        CLUSTERED_DATA_PATH,
        index=False,
    )


def save_training_summary(
    model: Pipeline,
    features: pd.DataFrame,
) -> None:
    """Save K-Means configuration and training information."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    clusterer = model.named_steps["clusterer"]

    summary = {
        "algorithm": "K-Means",
        "dataset": "Scikit-learn Wine dataset",
        "training_records": len(features),
        "feature_count": features.shape[1],
        "n_clusters": clusterer.n_clusters,
        "initialization": clusterer.init,
        "n_init": clusterer.n_init,
        "max_iterations": clusterer.max_iter,
        "actual_iterations": int(clusterer.n_iter_),
        "tolerance": clusterer.tol,
        "inertia": float(clusterer.inertia_),
        "feature_scaling": "StandardScaler",
        "random_state": RANDOM_STATE,
        "target_labels_used_for_training": False,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def main() -> None:
    """Run the complete K-Means training workflow."""

    features = load_dataset()

    model = build_pipeline()

    cluster_labels = model.fit_predict(features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    save_training_data(features)

    save_clustered_data(
        features=features,
        cluster_labels=cluster_labels,
    )

    save_training_summary(
        model=model,
        features=features,
    )

    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
    )

    print("\nK-Means training completed.")
    print(f"Training records: {len(features)}")
    print(f"Feature count: {features.shape[1]}")
    print(f"Number of clusters: {N_CLUSTERS}")

    print("\nCluster sizes:")

    for cluster_number, count in cluster_counts.items():
        print(
            f"Cluster {cluster_number}: "
            f"{count} observations"
        )

    print(
        "\nInertia: "
        f"{model.named_steps['clusterer'].inertia_:.4f}"
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()