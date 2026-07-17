# Support Vector Machine Classification

## Overview

Support Vector Machine, or SVM, is a supervised machine-learning algorithm used for classification and regression.

For classification, SVM attempts to identify a decision boundary that separates classes while maximizing the margin between the boundary and the closest training observations.

The closest influential observations are known as support vectors.

This implementation uses a nonlinear Radial Basis Function kernel.

## Business Problem

A healthcare decision-support system needs to classify observations as malignant or benign using numerical diagnostic measurements.

The model can support:

- preliminary risk classification;
- case prioritization;
- nonlinear pattern detection;
- model benchmarking;
- probability-based decision support.

This project is an educational demonstration and is not a clinically validated diagnostic system.

## Dataset

The Breast Cancer Wisconsin dataset from scikit-learn is used.

| Target | Meaning |
|---|---|
| 0 | Malignant |
| 1 | Benign |

The same dataset is used for several classification algorithms in this repository to support fair model comparison.

## Core Idea

SVM attempts to identify a decision boundary with the largest practical separation between classes.

Observations closest to the boundary have the greatest influence on its position.

An RBF kernel allows SVM to create nonlinear decision boundaries by measuring similarity between observations.

## Model Pipeline

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
                method="sigmoid",
                cv=5,
            ),
        ),
    ]
)

Why Scaling Is Required

SVM uses feature values when calculating margins and kernel similarity.

Features with much larger numerical scales can dominate the model.

StandardScaler transforms each feature to have approximately:

mean of zero;
standard deviation of one.

The scaler is included in the pipeline to ensure that identical preprocessing is applied during training, evaluation, and prediction.

Hyperparameters
kernel="rbf"

Uses the Radial Basis Function kernel to represent nonlinear class boundaries.

C=2.0

Controls the trade-off between:

creating a wider margin;
reducing classification errors.

A smaller C permits more training errors and creates stronger regularization.

A larger C penalizes errors more heavily and may create a more complex boundary.

gamma="scale"

Controls the influence range of individual training observations.

A small gamma creates smoother boundaries.

A large gamma creates more localized and complex boundaries.

class_weight="balanced"

Adjusts class weights according to class frequency.

Sigmoid probability calibration

The SVM is wrapped in CalibratedClassifierCV to produce class-probability estimates.

Five-fold internal calibration is used.

Learning Objectives

This implementation demonstrates:

maximum-margin classification;
support-vector concepts;
nonlinear kernel methods;
RBF kernels;
feature scaling;
model regularization;
class weighting;
probability calibration;
reusable training, evaluation, and inference workflows.
Evaluation Metrics

The evaluation includes:

accuracy;
balanced accuracy;
class-specific precision;
class-specific recall;
class-specific F1-score;
ROC-AUC;
confusion matrix;
ROC curve;
precision-recall curve;
calibration curve.
Output Files
outputs/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── calibration_curve.png
├── metrics/
│   ├── metrics.json
│   ├── training_summary.json
│   ├── classification_report.txt
│   └── confusion_matrix_values.csv
└── predictions/
    └── test_predictions.csv
Strengths
Effective in high-dimensional feature spaces.
Supports nonlinear decision boundaries.
Strong regularization controls.
Can perform well with clear class separation.
Uses influential boundary observations.
Supports class weighting.
Can be combined with probability calibration.
Limitations
Sensitive to feature scaling.
Sensitive to C and gamma.
Training can be slow for large datasets.
Less interpretable than Logistic Regression or Decision Trees.
RBF models do not provide simple feature coefficients.
Probability calibration adds computational cost.
Performance can decline with noisy or overlapping classes.