# Gaussian Naive Bayes Classification

## Overview

Gaussian Naive Bayes is a supervised probabilistic classification algorithm.

It applies Bayes' theorem while making two important assumptions:

1. Features are conditionally independent within each class.
2. Numerical feature values follow Gaussian distributions within each class.

The model calculates the probability of each class given the observed feature values and predicts the class with the highest posterior probability.

## Business Problem

A healthcare decision-support system needs to classify diagnostic observations as malignant or benign using numerical measurements.

Gaussian Naive Bayes can support:

- rapid preliminary classification;
- probability-based prioritization;
- lightweight model deployment;
- baseline model comparison;
- incremental or streaming-learning applications.

This project is an educational demonstration and not a clinically validated diagnostic system.

## Dataset

The Breast Cancer Wisconsin dataset from scikit-learn is used.

| Target | Meaning |
|---|---|
| 0 | Malignant |
| 1 | Benign |

The same dataset is used across several classification algorithms to allow fair comparison.

## Core Principle

The model estimates:

```text
P(Class | Features)
The independence assumption allows the likelihood to be represented as the product of individual feature likelihoods.

Gaussian Assumption

For every feature and class, the algorithm estimates:

the class-specific mean;
the class-specific variance.

It then uses a Gaussian probability distribution to estimate how likely an observed feature value is for each class.

Model Configuration
GaussianNB(
    var_smoothing=1e-9,
)
Variance Smoothing

var_smoothing adds a small amount of numerical stability to feature variances.

This helps avoid division by extremely small variance values.

Why Scaling Was Not Applied

Gaussian Naive Bayes estimates a separate mean and variance for every feature within each class.

Unlike KNN and SVM, it does not calculate one combined Euclidean distance or margin using all raw feature scales.

Therefore, feature scaling is not required for this implementation.

Learning Objectives

This project demonstrates:

probabilistic classification;
Bayes' theorem;
conditional probability;
conditional independence;
class priors;
Gaussian likelihood estimation;
posterior probabilities;
probability calibration analysis;
class-specific feature statistics;
reusable training and inference workflows.
Evaluation Metrics
Accuracy
Balanced accuracy
Precision
Recall
F1-score
ROC-AUC
Log loss
Confusion matrix
Precision–recall curve
Calibration curve
Output Files
outputs/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── calibration_curve.png
│   └── probability_distribution.png
├── metrics/
│   ├── metrics.json
│   ├── training_summary.json
│   ├── classification_report.txt
│   ├── confusion_matrix_values.csv
│   └── class_parameters.csv
└── predictions/
    └── test_predictions.csv
Strengths
Very fast training.
Very fast prediction.
Simple probabilistic interpretation.
Works well as a baseline.
Supports probability predictions.
Requires relatively little training data.
Supports incremental learning.
Can perform well when assumptions are approximately valid.
Limitations
Assumes conditional feature independence.
Assumes Gaussian numerical feature distributions.
Correlated features may reduce performance.
Probabilities may be overconfident.
Complex nonlinear feature interactions are not learned.
Distribution assumptions may not match real data.