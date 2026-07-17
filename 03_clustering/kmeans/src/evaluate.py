"""Evaluate and visualize the saved K-Means clustering model."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "kmeans_pipeline.joblib"
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"


def load_artifacts():
    """Load trained pipeline and original training data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "K-Means model not found. Run train.py first."
        )

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            "Training data not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    features = pd.read_csv(TRAINING_DATA_PATH)

    return model, features


def calculate_metrics(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
    inertia: float,
) -> dict[str, float]:
    """Calculate internal clustering-evaluation metrics."""

    return {
        "inertia": float(inertia),
        "silhouette_score": float(
            silhouette_score(
                scaled_features,
                cluster_labels,
            )
        ),
        "davies_bouldin_score": float(
            davies_bouldin_score(
                scaled_features,
                cluster_labels,
            )
        ),
        "calinski_harabasz_score": float(
            calinski_harabasz_score(
                scaled_features,
                cluster_labels,
            )
        ),
        "cluster_count": int(
            len(np.unique(cluster_labels))
        ),
    }


def save_metrics(metrics: dict[str, float]) -> None:
    """Save clustering metrics as JSON."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with (
        METRICS_DIR / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_cluster_profiles(
    features: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> None:
    """Save cluster sizes and feature-level cluster profiles."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    clustered_data = features.copy()
    clustered_data["cluster"] = cluster_labels

    cluster_sizes = (
        clustered_data["cluster"]
        .value_counts()
        .sort_index()
        .rename("observation_count")
        .reset_index()
    )

    cluster_sizes.to_csv(
        METRICS_DIR / "cluster_sizes.csv",
        index=False,
    )

    cluster_profiles = (
        clustered_data
        .groupby("cluster")
        .mean()
        .round(4)
    )

    cluster_profiles.to_csv(
        METRICS_DIR / "cluster_profiles.csv"
    )


def save_centroids(
    model,
    feature_names: list[str],
) -> None:
    """Convert scaled centroids to original feature units."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = model.named_steps["scaler"]
    clusterer = model.named_steps["clusterer"]

    original_centroids = scaler.inverse_transform(
        clusterer.cluster_centers_
    )

    centroid_table = pd.DataFrame(
        original_centroids,
        columns=feature_names,
    )

    centroid_table.insert(
        0,
        "cluster",
        range(clusterer.n_clusters),
    )

    centroid_table.to_csv(
        METRICS_DIR / "cluster_centroids.csv",
        index=False,
    )


def evaluate_candidate_cluster_counts(
    scaled_features: np.ndarray,
) -> pd.DataFrame:
    """Evaluate candidate values of k from 2 through 10."""

    evaluation_rows = []

    for cluster_count in range(2, 11):
        candidate_model = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=RANDOM_STATE,
        )

        labels = candidate_model.fit_predict(
            scaled_features
        )

        evaluation_rows.append(
            {
                "k": cluster_count,
                "inertia": candidate_model.inertia_,
                "silhouette_score": silhouette_score(
                    scaled_features,
                    labels,
                ),
            }
        )

    evaluation_table = pd.DataFrame(evaluation_rows)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_table.to_csv(
        METRICS_DIR / "candidate_k_scores.csv",
        index=False,
    )

    return evaluation_table


def save_elbow_plot(
    evaluation_table: pd.DataFrame,
) -> None:
    """Save elbow-method visualization."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(
        evaluation_table["k"],
        evaluation_table["inertia"],
        marker="o",
    )

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Method")
    plt.xticks(evaluation_table["k"])
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "elbow_plot.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_silhouette_comparison(
    evaluation_table: pd.DataFrame,
) -> None:
    """Save silhouette scores across candidate cluster counts."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(
        evaluation_table["k"],
        evaluation_table["silhouette_score"],
        marker="o",
    )

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Mean Silhouette Score")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xticks(evaluation_table["k"])
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "silhouette_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_pca_cluster_plot(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
) -> None:
    """Visualize clusters in two PCA dimensions."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    reduced_features = pca.fit_transform(
        scaled_features
    )

    reduced_centers = pca.transform(
        cluster_centers
    )

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=cluster_labels,
        alpha=0.75,
    )

    plt.scatter(
        reduced_centers[:, 0],
        reduced_centers[:, 1],
        marker="X",
        s=250,
        label="Centroids",
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means Clusters in PCA Space")
    plt.legend()
    plt.colorbar(
        scatter,
        label="Cluster",
    )
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "pca_cluster_plot.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_silhouette_distribution(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
) -> None:
    """Plot silhouette values for each observation."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sample_scores = silhouette_samples(
        scaled_features,
        cluster_labels,
    )

    silhouette_table = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "silhouette_score": sample_scores,
        }
    ).sort_values(
        by=["cluster", "silhouette_score"]
    )

    silhouette_table.to_csv(
        PREDICTIONS_DIR / "sample_silhouette_scores.csv",
        index=False,
    )

    plt.figure(figsize=(9, 7))

    y_lower = 10

    for cluster_number in sorted(
        np.unique(cluster_labels)
    ):
        values = np.sort(
            sample_scores[
                cluster_labels == cluster_number
            ]
        )

        cluster_size = len(values)
        y_upper = y_lower + cluster_size

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            alpha=0.7,
        )

        plt.text(
            -0.05,
            y_lower + 0.5 * cluster_size,
            str(cluster_number),
        )

        y_lower = y_upper + 10

    mean_score = silhouette_score(
        scaled_features,
        cluster_labels,
    )

    plt.axvline(
        mean_score,
        linestyle="--",
        label=f"Mean = {mean_score:.3f}",
    )

    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Clustered Observations")
    plt.title("K-Means Silhouette Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "silhouette_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run the complete K-Means evaluation workflow."""

    model, features = load_artifacts()

    scaler = model.named_steps["scaler"]
    clusterer = model.named_steps["clusterer"]

    scaled_features = scaler.transform(features)
    cluster_labels = clusterer.predict(scaled_features)

    metrics = calculate_metrics(
        scaled_features=scaled_features,
        cluster_labels=cluster_labels,
        inertia=clusterer.inertia_,
    )

    save_metrics(metrics)

    save_cluster_profiles(
        features=features,
        cluster_labels=cluster_labels,
    )

    save_centroids(
        model=model,
        feature_names=list(features.columns),
    )

    evaluation_table = evaluate_candidate_cluster_counts(
        scaled_features
    )

    save_elbow_plot(evaluation_table)

    save_silhouette_comparison(
        evaluation_table
    )

    save_pca_cluster_plot(
        scaled_features=scaled_features,
        cluster_labels=cluster_labels,
        cluster_centers=clusterer.cluster_centers_,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_silhouette_distribution(
        scaled_features=scaled_features,
        cluster_labels=cluster_labels,
    )

    print("\nK-Means Evaluation")

    for metric_name, metric_value in metrics.items():
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

    best_row = evaluation_table.loc[
        evaluation_table[
            "silhouette_score"
        ].idxmax()
    ]

    print(
        "\nBest candidate by silhouette score: "
        f"k={int(best_row['k'])}"
    )

    print(
        "Best silhouette score: "
        f"{best_row['silhouette_score']:.4f}"
    )

    print(
        "\nEvaluation outputs saved successfully."
    )


if __name__ == "__main__":
    main()