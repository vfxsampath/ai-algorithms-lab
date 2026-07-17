# DBSCAN Clustering

## Overview

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.

It is an unsupervised clustering algorithm that identifies dense groups of observations and labels observations in low-density regions as noise.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified before training.

## Business Problem

A product-analysis team wants to identify:

- naturally dense product groups;
- irregularly shaped product segments;
- unusual or isolated product profiles;
- observations that do not fit established groups.

Potential applications include:

- product segmentation;
- customer segmentation;
- fraud pattern discovery;
- operational anomaly identification;
- geographic hotspot analysis;
- quality-control analysis.

This project uses the scikit-learn Wine dataset as a product-profile clustering example.

## Dataset

The Wine dataset contains 178 observations and 13 numerical chemical features.

The original target classes are not used during model training.

## Core DBSCAN Concepts

DBSCAN separates observations into three types.

### Core Samples

A core sample has at least `min_samples` observations within its `eps` neighborhood, including itself.

Core samples form the dense foundation of clusters.

### Border Samples

A border sample lies within the neighborhood of a core sample but does not independently satisfy the density requirement.

### Noise Samples

A noise sample cannot be connected to a sufficiently dense region.

DBSCAN assigns the label:

```text
-1

o noise observations.

Model Pipeline
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clusterer",
            DBSCAN(
                eps=2.3,
                min_samples=5,
                metric="euclidean",
                algorithm="auto",
                leaf_size=30,
                n_jobs=-1,
            ),
        ),
    ]
)
Why Scaling Is Required

DBSCAN uses distances between observations.

Features with larger numerical ranges can dominate the distance calculation.

StandardScaler places the features on comparable standardized scales.

Main Parameters
eps=2.3

eps defines the maximum distance between two observations for them to be considered neighbors.

A smaller value may create:

more noise;
smaller clusters;
fragmented dense regions.

A larger value may create:

fewer noise observations;
larger merged clusters;
reduced separation.
min_samples=5

This specifies the minimum number of observations required within an eps neighborhood for an observation to become a core sample.

A higher value generally requires denser regions and may classify more observations as noise.

metric="euclidean"

Euclidean distance is used in the standardized feature space.

How DBSCAN Works
Select an unvisited observation.
Identify observations within its eps radius.
Determine whether it is a core sample.
Expand a cluster through connected core neighborhoods.
Assign border observations to reachable clusters.
Label unconnected low-density observations as noise.
Continue until all observations are processed.
Evaluation

Because this is unsupervised learning, no target labels are required.

The project evaluates:

number of clusters;
noise count;
noise percentage;
core-sample count;
border-sample count;
silhouette score excluding noise;
Davies–Bouldin score excluding noise;
Calinski–Harabasz score excluding noise;
cluster profiles;
parameter sensitivity.
K-Distance Plot

The k-distance plot displays each observation’s distance to its min_samplesth nearest neighbor.

A visible bend can help identify a reasonable starting value for eps.

The selected value should also be evaluated through:

cluster count;
noise percentage;
silhouette score;
profile usefulness;
domain interpretation.
Parameter Comparison

This project tests multiple combinations of:

eps;
min_samples.

The comparison table records:

discovered cluster count;
noise count;
noise percentage;
silhouette score excluding noise.

The mathematically highest silhouette score is not automatically the best business solution.

PCA Visualization

PCA reduces the standardized feature space to two dimensions for visualization.

The chart distinguishes:

core samples;
border samples;
noise observations.

DBSCAN itself is trained using all standardized features.

Approximate New-Observation Assignment

Standard DBSCAN does not provide native prediction for unseen observations.

The provided predict.py performs an approximation:

scale the new observation;
locate its nearest trained core sample;
compare the distance with eps;
assign the core sample’s cluster when within eps;
otherwise label the observation as noise.

This approximation does not retrain or expand the DBSCAN clusters.

Output Files
outputs/
├── figures/
│   ├── k_distance_plot.png
│   ├── parameter_cluster_count.png
│   ├── pca_cluster_plot.png
│   └── silhouette_distribution.png
├── metrics/
│   ├── metrics.json
│   ├── training_summary.json
│   ├── cluster_sizes.csv
│   ├── cluster_profiles.csv
│   ├── core_samples.csv
│   ├── k_distances.csv
│   └── parameter_comparison.csv
└── predictions/
    └── sample_silhouette_scores.csv
Strengths
Does not require the number of clusters in advance.
Detects irregular cluster shapes.
Explicitly identifies noise.
Can discover dense local structures.
Does not rely on centroids.
Can be useful for spatial and anomaly-oriented analysis.
Limitations
Sensitive to eps.
Sensitive to min_samples.
Sensitive to feature scaling.
Can struggle with clusters of different densities.
May label many observations as noise.
Can merge nearby dense regions.
Standard DBSCAN does not naturally predict new observations.
Distance becomes less informative in high-dimensional spaces.
