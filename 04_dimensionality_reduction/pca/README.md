# Principal Component Analysis

## Overview

Principal Component Analysis, or PCA, is an unsupervised linear dimensionality-reduction technique.

PCA transforms correlated numerical features into a smaller set of uncorrelated variables called principal components.

The first principal component captures the greatest possible variance. Each following component captures the greatest remaining variance while remaining orthogonal to the earlier components.

## Business Problem

An analytics team has a dataset containing many correlated product measurements.

Using all original features may create:

- redundant information;
- difficult visualizations;
- slower downstream models;
- multicollinearity;
- more complex analysis.

PCA can reduce the feature space while retaining most of the available variance.

Potential applications include:

- data compression;
- exploratory visualization;
- feature extraction;
- multicollinearity reduction;
- preprocessing for clustering;
- preprocessing for predictive models;
- anomaly-detection support.

## Dataset

The project uses the numerical features from the scikit-learn Wine dataset.

The target labels are not used when fitting PCA.

## Correct Train-Test Design

The full feature dataset is split into:

- 80% training data;
- 20% held-out test data.

Only the training data are used to fit:

1. `StandardScaler`;
2. PCA.

The fitted pipeline then transforms the held-out test data.

This prevents the test-set feature distribution from influencing the fitted scaling parameters or principal components.

## Model Pipeline

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=0.95,
                svd_solver="full",
            ),
        ),
    ]
)

Why Scaling Is Required

PCA is sensitive to feature variance.

Features measured on larger numerical scales can dominate the principal components.

StandardScaler transforms features using training-set means and standard deviations before PCA is fitted.

Selecting the Number of Components

The model uses:

n_components = 0.95

This instructs PCA to retain the minimum number of components needed to explain at least 95% of the training-set variance.

The actual number of retained components is determined during fitting.

Principal Components

Each component is a weighted combination of the original standardized features.

Component scores describe where each observation lies in the reduced-dimensional space.

Component loadings describe how strongly original features contribute to each principal component.

Explained Variance

Explained variance ratio reports the proportion of total standardized variance represented by each principal component.

Cumulative explained variance shows how much variance is retained as components are added.

Reconstruction

The reduced component representation can be approximately transformed back into the original feature space.

Reconstruction error measures the information lost through dimensionality reduction.

Lower reconstruction error indicates that the retained components preserve the original observations more accurately.

Evaluation Design

The project evaluates:

retained component count;
cumulative explained variance;
dimensionality-reduction percentage;
training reconstruction error;
held-out test reconstruction error;
train-to-test reconstruction comparison;
feature-level reconstruction error;
component loadings;
train and test projections.
Output Files
outputs/
├── figures/
│   ├── explained_variance.png
│   ├── cumulative_explained_variance.png
│   ├── train_test_pca_projection.png
│   └── component_loadings.png
├── metrics/
│   ├── training_summary.json
│   ├── metrics.json
│   ├── explained_variance.csv
│   ├── component_loadings.csv
│   └── test_reconstruction_error_by_feature.csv
├── transformed_data/
│   ├── train_transformed.csv
│   ├── test_transformed.csv
│   ├── test_reconstructed.csv
│   └── test_reconstruction_errors.csv
└── predictions/
    ├── pca_transformed_predictions.csv
    ├── pca_reconstructed_predictions.csv
    └── pca_prediction_reconstruction_errors.csv