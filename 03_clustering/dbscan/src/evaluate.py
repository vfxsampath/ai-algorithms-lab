"""Evaluate and visualize the saved DBSCAN clustering model."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "dbscan_pipeline.joblib"
)

TRAINING_DATA_PATH = (
    BASE_DIR
    / "data"
    / "training_data.csv"
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
    """Load the saved DBSCAN pipeline and training data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "DBSCAN model not found. "
            "Run train.py first."
        )

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            "Training data not found. "
            "Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    features = pd.read_csv(
        TRAINING_DATA_PATH
    )

    return model, features


def get_non_noise_data(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observations and labels excluding noise."""

    non_noise_mask = cluster_labels != -1

    return (
        scaled_features[non_noise_mask],
        cluster_labels[non_noise_mask],
    )


def calculate_metrics(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
    core_sample_count: int,
) -> dict[str, float | int | None]:
    """Calculate DBSCAN clustering metrics."""

    cluster_count = len(
        set(cluster_labels) - {-1}
    )

    noise_count = int(
        np.sum(cluster_labels == -1)
    )

    border_count = int(
        len(cluster_labels)
        - noise_count
        - core_sample_count
    )

    metrics: dict[str, float | int | None] = {
        "cluster_count_excluding_noise": cluster_count,
        "noise_count": noise_count,
        "noise_percentage": float(
            noise_count / len(cluster_labels) * 100
        ),
        "core_sample_count": core_sample_count,
        "border_sample_count": border_count,
        "silhouette_score_excluding_noise": None,
        "davies_bouldin_score_excluding_noise": None,
        "calinski_harabasz_score_excluding_noise": None,
    }

    non_noise_features, non_noise_labels = (
        get_non_noise_data(
            scaled_features,
            cluster_labels,
        )
    )

    unique_non_noise_clusters = np.unique(
        non_noise_labels
    )

    if (
        len(unique_non_noise_clusters) >= 2
        and len(non_noise_features)
        > len(unique_non_noise_clusters)
    ):
        metrics[
            "silhouette_score_excluding_noise"
        ] = float(
            silhouette_score(
                non_noise_features,
                non_noise_labels,
            )
        )

        metrics[
            "davies_bouldin_score_excluding_noise"
        ] = float(
            davies_bouldin_score(
                non_noise_features,
                non_noise_labels,
            )
        )

        metrics[
            "calinski_harabasz_score_excluding_noise"
        ] = float(
            calinski_harabasz_score(
                non_noise_features,
                non_noise_labels,
            )
        )

    return metrics


def save_metrics(
    metrics: dict[str, float | int | None],
) -> None:
    """Save clustering metrics as JSON."""

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


def save_cluster_sizes(
    cluster_labels: np.ndarray,
) -> None:
    """Save observation counts for clusters and noise."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_sizes = (
        pd.Series(
            cluster_labels,
            name="cluster",
        )
        .value_counts()
        .sort_index()
        .rename("observation_count")
        .reset_index()
    )

    cluster_sizes["cluster_type"] = (
        cluster_sizes["cluster"]
        .apply(
            lambda value: (
                "noise"
                if value == -1
                else "cluster"
            )
        )
    )

    cluster_sizes.to_csv(
        METRICS_DIR / "cluster_sizes.csv",
        index=False,
    )


def save_cluster_profiles(
    features: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> None:
    """Save mean feature profiles by cluster."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clustered_data = features.copy()
    clustered_data["cluster"] = cluster_labels

    profiles = (
        clustered_data
        .groupby("cluster")
        .mean()
        .round(4)
    )

    profiles.to_csv(
        METRICS_DIR / "cluster_profiles.csv"
    )


def save_core_sample_data(
    features: pd.DataFrame,
    cluster_labels: np.ndarray,
    core_sample_indices: np.ndarray,
) -> None:
    """Save original feature values for core samples."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    core_data = features.iloc[
        core_sample_indices
    ].copy()

    core_data.insert(
        0,
        "original_index",
        core_sample_indices,
    )

    core_data["cluster"] = cluster_labels[
        core_sample_indices
    ]

    core_data.to_csv(
        METRICS_DIR / "core_samples.csv",
        index=False,
    )


def calculate_k_distances(
    scaled_features: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    """Calculate sorted k-nearest-neighbor distances."""

    neighbor_model = NearestNeighbors(
        n_neighbors=min_samples,
        metric="euclidean",
    )

    neighbor_model.fit(
        scaled_features
    )

    distances, _ = (
        neighbor_model.kneighbors(
            scaled_features
        )
    )

    k_distances = np.sort(
        distances[:, -1]
    )

    return k_distances


def save_k_distance_plot(
    scaled_features: np.ndarray,
    min_samples: int,
    selected_eps: float,
) -> None:
    """Create the k-distance chart used to assess eps."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    k_distances = calculate_k_distances(
        scaled_features=scaled_features,
        min_samples=min_samples,
    )

    k_distance_table = pd.DataFrame(
        {
            "sorted_observation": np.arange(
                1,
                len(k_distances) + 1,
            ),
            "k_distance": k_distances,
        }
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    k_distance_table.to_csv(
        METRICS_DIR / "k_distances.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        np.arange(1, len(k_distances) + 1),
        k_distances,
    )

    plt.axhline(
        selected_eps,
        linestyle="--",
        label=f"Selected eps = {selected_eps}",
    )

    plt.xlabel("Observations Sorted by Distance")
    plt.ylabel(
        f"Distance to {min_samples}th Nearest Neighbor"
    )
    plt.title("DBSCAN K-Distance Plot")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "k_distance_plot.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def evaluate_parameter_combinations(
    scaled_features: np.ndarray,
) -> pd.DataFrame:
    """Compare multiple eps and min_samples combinations."""

    evaluation_rows = []

    eps_values = [
        1.5,
        1.8,
        2.0,
        2.2,
        2.3,
        2.5,
        2.8,
        3.0,
    ]

    min_samples_values = [
        4,
        5,
        6,
        8,
        10,
    ]

    for eps_value in eps_values:
        for minimum_samples in min_samples_values:
            candidate = DBSCAN(
                eps=eps_value,
                min_samples=minimum_samples,
                metric="euclidean",
                n_jobs=-1,
            )

            labels = candidate.fit_predict(
                scaled_features
            )

            cluster_count = len(
                set(labels) - {-1}
            )

            noise_count = int(
                np.sum(labels == -1)
            )

            row = {
                "eps": eps_value,
                "min_samples": minimum_samples,
                "cluster_count": cluster_count,
                "noise_count": noise_count,
                "noise_percentage": (
                    noise_count
                    / len(labels)
                    * 100
                ),
                "silhouette_score_excluding_noise": None,
            }

            non_noise_mask = labels != -1
            non_noise_labels = labels[
                non_noise_mask
            ]
            non_noise_features = scaled_features[
                non_noise_mask
            ]

            if (
                len(np.unique(non_noise_labels)) >= 2
                and len(non_noise_features)
                > len(np.unique(non_noise_labels))
            ):
                row[
                    "silhouette_score_excluding_noise"
                ] = silhouette_score(
                    non_noise_features,
                    non_noise_labels,
                )

            evaluation_rows.append(row)

    parameter_table = pd.DataFrame(
        evaluation_rows
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_table.to_csv(
        METRICS_DIR
        / "parameter_comparison.csv",
        index=False,
    )

    return parameter_table


def save_parameter_heatmap(
    parameter_table: pd.DataFrame,
) -> None:
    """Plot cluster count across eps and min_samples values."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot_table = parameter_table.pivot(
        index="min_samples",
        columns="eps",
        values="cluster_count",
    )

    plt.figure(figsize=(10, 6))

    image = plt.imshow(
        pivot_table.to_numpy(),
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Number of Clusters",
    )

    plt.xticks(
        range(len(pivot_table.columns)),
        pivot_table.columns,
    )

    plt.yticks(
        range(len(pivot_table.index)),
        pivot_table.index,
    )

    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.title(
        "DBSCAN Cluster Count by Parameters"
    )

    for row_index in range(
        len(pivot_table.index)
    ):
        for column_index in range(
            len(pivot_table.columns)
        ):
            value = pivot_table.iloc[
                row_index,
                column_index,
            ]

            plt.text(
                column_index,
                row_index,
                str(int(value)),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "parameter_cluster_count.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_pca_cluster_plot(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
    core_sample_indices: np.ndarray,
) -> None:
    """Visualize clusters and noise in two PCA dimensions."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pca = PCA(n_components=2)

    reduced_features = pca.fit_transform(
        scaled_features
    )

    core_mask = np.zeros(
        len(cluster_labels),
        dtype=bool,
    )

    core_mask[
        core_sample_indices
    ] = True

    unique_labels = sorted(
        np.unique(cluster_labels)
    )

    plt.figure(figsize=(10, 7))

    for label in unique_labels:
        label_mask = cluster_labels == label

        if label == -1:
            plt.scatter(
                reduced_features[
                    label_mask,
                    0,
                ],
                reduced_features[
                    label_mask,
                    1,
                ],
                marker="x",
                alpha=0.8,
                label="Noise",
            )

            continue

        cluster_core_mask = (
            label_mask & core_mask
        )

        cluster_border_mask = (
            label_mask & ~core_mask
        )

        plt.scatter(
            reduced_features[
                cluster_core_mask,
                0,
            ],
            reduced_features[
                cluster_core_mask,
                1,
            ],
            s=70,
            alpha=0.8,
            label=f"Cluster {label} core",
        )

        plt.scatter(
            reduced_features[
                cluster_border_mask,
                0,
            ],
            reduced_features[
                cluster_border_mask,
                1,
            ],
            s=30,
            alpha=0.5,
            label=f"Cluster {label} border",
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(
        "DBSCAN Clusters, Core Samples, and Noise"
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "pca_cluster_plot.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_silhouette_distribution(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
) -> None:
    """Save silhouette scores for non-noise observations."""

    non_noise_features, non_noise_labels = (
        get_non_noise_data(
            scaled_features,
            cluster_labels,
        )
    )

    unique_clusters = np.unique(
        non_noise_labels
    )

    if (
        len(unique_clusters) < 2
        or len(non_noise_features)
        <= len(unique_clusters)
    ):
        print(
            "Silhouette distribution skipped: "
            "fewer than two non-noise clusters."
        )
        return

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores = silhouette_samples(
        non_noise_features,
        non_noise_labels,
    )

    score_table = pd.DataFrame(
        {
            "cluster": non_noise_labels,
            "silhouette_score": scores,
        }
    )

    score_table.to_csv(
        PREDICTIONS_DIR
        / "sample_silhouette_scores.csv",
        index=False,
    )

    plt.figure(figsize=(9, 7))

    y_lower = 10

    for cluster_number in unique_clusters:
        cluster_scores = np.sort(
            scores[
                non_noise_labels
                == cluster_number
            ]
        )

        cluster_size = len(
            cluster_scores
        )

        y_upper = (
            y_lower
            + cluster_size
        )

        plt.fill_betweenx(
            np.arange(
                y_lower,
                y_upper,
            ),
            0,
            cluster_scores,
            alpha=0.7,
        )

        plt.text(
            -0.05,
            y_lower
            + 0.5
            * cluster_size,
            str(cluster_number),
        )

        y_lower = y_upper + 10

    mean_score = silhouette_score(
        non_noise_features,
        non_noise_labels,
    )

    plt.axvline(
        mean_score,
        linestyle="--",
        label=f"Mean = {mean_score:.3f}",
    )

    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Non-Noise Observations")
    plt.title(
        "DBSCAN Silhouette Distribution"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "silhouette_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run the complete DBSCAN evaluation workflow."""

    model, features = load_artifacts()

    scaler = model.named_steps["scaler"]
    clusterer = model.named_steps["clusterer"]

    scaled_features = scaler.transform(
        features
    )

    cluster_labels = clusterer.labels_

    metrics = calculate_metrics(
        scaled_features=scaled_features,
        cluster_labels=cluster_labels,
        core_sample_count=len(
            clusterer.core_sample_indices_
        ),
    )

    save_metrics(metrics)

    save_cluster_sizes(
        cluster_labels
    )

    save_cluster_profiles(
        features=features,
        cluster_labels=cluster_labels,
    )

    save_core_sample_data(
        features=features,
        cluster_labels=cluster_labels,
        core_sample_indices=(
            clusterer.core_sample_indices_
        ),
    )

    save_k_distance_plot(
        scaled_features=scaled_features,
        min_samples=clusterer.min_samples,
        selected_eps=clusterer.eps,
    )

    parameter_table = (
        evaluate_parameter_combinations(
            scaled_features
        )
    )

    save_parameter_heatmap(
        parameter_table
    )

    save_pca_cluster_plot(
        scaled_features=scaled_features,
        cluster_labels=cluster_labels,
        core_sample_indices=(
            clusterer.core_sample_indices_
        ),
    )

    save_silhouette_distribution(
        scaled_features=scaled_features,
        cluster_labels=cluster_labels,
    )

    print("\nDBSCAN Evaluation")

    for metric_name, metric_value in (
        metrics.items()
    ):
        if isinstance(
            metric_value,
            float,
        ):
            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )
        else:
            print(
                f"{metric_name}: "
                f"{metric_value}"
            )

    valid_candidates = (
        parameter_table
        .dropna(
            subset=[
                "silhouette_score_excluding_noise"
            ]
        )
    )

    if not valid_candidates.empty:
        best_candidate = valid_candidates.loc[
            valid_candidates[
                "silhouette_score_excluding_noise"
            ].idxmax()
        ]

        print(
            "\nBest tested parameters by "
            "non-noise silhouette score:"
        )

        print(
            f"eps: {best_candidate['eps']}"
        )

        print(
            "min_samples: "
            f"{int(best_candidate['min_samples'])}"
        )

        print(
            "clusters: "
            f"{int(best_candidate['cluster_count'])}"
        )

        print(
            "noise percentage: "
            f"{best_candidate['noise_percentage']:.2f}%"
        )

        print(
            "silhouette score: "
            f"{best_candidate['silhouette_score_excluding_noise']:.4f}"
        )

    print(
        "\nEvaluation outputs saved successfully."
    )


if __name__ == "__main__":
    main()
