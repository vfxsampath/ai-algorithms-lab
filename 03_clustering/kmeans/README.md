# K-Means Clustering

## Overview

K-Means is an unsupervised machine-learning algorithm that divides observations into a specified number of clusters.

The algorithm attempts to create groups whose members are similar to one another while remaining different from members of other groups.

K-Means does not use target labels during training.

## Business Problem

A product-analysis team wants to segment products according to measurable characteristics.

The resulting clusters could support:

- product portfolio segmentation;
- differentiated marketing strategies;
- product-positioning analysis;
- quality-profile identification;
- customer or product grouping;
- exploratory pattern discovery.

This project uses the scikit-learn Wine dataset as a self-contained product-profile segmentation example.

## Dataset

The dataset contains 13 numerical chemical measurements for 178 wine observations.

The original class labels are not used during model training.

## Core Algorithm

K-Means repeatedly performs these steps:

1. Initialize `k` cluster centroids.
2. Assign every observation to its nearest centroid.
3. Recalculate each centroid using assigned observations.
4. Repeat until cluster assignments stabilize or the iteration limit is reached.

## Objective

K-Means minimizes within-cluster squared distances.

This measure is reported as inertia.

Lower inertia means observations are closer to their assigned centroids.

However, inertia always decreases as the number of clusters increases, so it cannot select the best `k` by itself.

## Model Pipeline

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clusterer",
            KMeans(
                n_clusters=3,
                init="k-means++",
                n_init=20,
                max_iter=300,
                tol=1e-4,
                random_state=42,
            ),
        ),
    ]
)

Why Scaling Is Required

K-Means uses Euclidean distance.

Features measured on larger numerical scales could dominate cluster assignments.

StandardScaler places features on comparable standardized scales.

Hyperparameters
n_clusters=3

The algorithm creates three groups.

This value is evaluated against candidate values from 2 through 10.

init="k-means++"

Initial centroids are selected to improve their spread and reduce poor initialization risk.

n_init=20

The algorithm runs 20 times using different centroid initializations.

The result with the lowest inertia is retained.

max_iter=300

Each initialization may perform up to 300 iterations.

random_state=42

Provides reproducible initialization.

Evaluation Metrics

Because clustering has no required target labels, internal metrics are used.

Inertia

Measures total squared distance from observations to assigned centroids.

Lower is better, but comparisons should use the same dataset and feature scaling.

Silhouette Score

Measures cluster cohesion and separation.

Near 1: well-separated clusters
Near 0: overlapping clusters
Below 0: potentially poor assignments
Davies–Bouldin Score

Measures cluster similarity.

Lower values are generally better.

Calinski–Harabasz Score

Measures between-cluster separation relative to within-cluster compactness.

Higher values are generally better.

Selecting the Number of Clusters

The project evaluates candidate values from k=2 through k=10 using:

the elbow method;
silhouette score;
domain interpretability;
cluster-size balance;
cluster-profile usefulness.

The highest silhouette score does not automatically produce the most useful business segmentation.

PCA Visualization

PCA reduces the standardized feature space to two dimensions for visualization.

PCA is used only for the chart.

The K-Means model itself is trained using all standardized features.

Output Files
outputs/
├── figures/
│   ├── elbow_plot.png
│   ├── silhouette_comparison.png
│   ├── silhouette_distribution.png
│   └── pca_cluster_plot.png
├── metrics/
│   ├── metrics.json
│   ├── training_summary.json
│   ├── cluster_sizes.csv
│   ├── cluster_profiles.csv
│   ├── cluster_centroids.csv
│   └── candidate_k_scores.csv
└── predictions/
    └── sample_silhouette_scores.csv
Strengths
Simple and computationally efficient.
Easy to apply to numerical datasets.
Produces clear centroid-based groups.
Scales relatively well.
Useful for exploratory segmentation.
New observations can be assigned to existing clusters.
Limitations
Requires the number of clusters in advance.
Sensitive to feature scaling.
Sensitive to initialization.
Sensitive to outliers.
Assumes roughly compact and similarly sized clusters.
Performs poorly with irregular cluster shapes.
Cluster numbers have no inherent meaning.
Results do not automatically represent business segments.