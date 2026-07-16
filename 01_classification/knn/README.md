# K-Nearest Neighbors Classification

## Overview

K-Nearest Neighbors, commonly called KNN, is a supervised machine-learning algorithm that classifies a new observation according to the classes of nearby training observations.

Unlike Logistic Regression, Decision Trees, and Random Forests, KNN does not learn a conventional parametric model. It stores the training examples and performs most of its computation during prediction.

## Business Problem

A healthcare decision-support system needs to classify diagnostic observations as malignant or benign using numerical measurements.

KNN can support:

- similarity-based case classification;
- identification of comparable historical observations;
- preliminary screening;
- model comparison;
- neighborhood-based explanations.

This project is an educational demonstration and not a clinically validated diagnostic system.

## Dataset

The Breast Cancer Wisconsin dataset from scikit-learn is used.

| Target | Meaning |
|---|---|
| 0 | Malignant |
| 1 | Benign |

The same dataset is used for Logistic Regression, Decision Tree, Random Forest, and KNN to support fair comparison.

## Core Idea

For each new observation:

1. Calculate its distance from training observations.
2. Select the `k` closest observations.
3. Examine their classes.
4. Combine their votes.
5. Return the final class and probability.

## Model Configuration

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="minkowski",
                p=2,
                algorithm="auto",
                n_jobs=-1,
            ),
        ),
    ]
)

Why Feature Scaling Is Required

KNN uses distances between observations.

If one feature uses values in thousands while another uses values between zero and one, the large-scale feature may dominate the distance calculation.

StandardScaler transforms each feature using its training-set mean and standard deviation.

The scaler and classifier are placed in one pipeline so the same preprocessing is applied during training, evaluation, and prediction.

Hyperparameters
n_neighbors=7

The prediction is based on seven nearby training observations.

A small value may make the model sensitive to noise.

A large value may produce smoother but less local predictions.

weights="distance"

Closer neighbors receive more influence than distant neighbors.

metric="minkowski" and p=2

This configuration produces Euclidean distance.

algorithm="auto"

Scikit-learn automatically chooses an appropriate neighbor-search method.

Learning Objectives

This project demonstrates:

instance-based learning;
similarity-based classification;
Euclidean distance;
feature scaling;
weighted neighbor voting;
probability prediction;
class-specific evaluation;
neighbor-distance inspection;
reusable training and inference workflows.
Evaluation Metrics
Accuracy
Balanced accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion matrix
Precision–recall curve
Output Files
outputs/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
├── metrics/
│   ├── metrics.json
│   ├── training_summary.json
│   ├── classification_report.txt
│   └── confusion_matrix_values.csv
└── predictions/
    ├── test_predictions.csv
    └── example_neighbors.csv
Strengths
Simple to understand.
No complex training process.
Handles nonlinear decision boundaries.
Supports multiclass classification.
Provides intuitive similarity-based reasoning.
Can work well on small, well-scaled datasets.
Limitations
Prediction can be slow on large datasets.
Requires storing the training data.
Sensitive to feature scaling.
Sensitive to irrelevant features.
Performance depends strongly on the selected value of k.
May perform poorly in high-dimensional spaces.
Sensitive to the chosen distance metric.
Run the Project
python 01_classification/knn/src/train.py
python 01_classification/knn/src/evaluate.py
python 01_classification/knn/src/predict.py
Results
Confusion Matrix

ROC Curve

Precision–Recall Curve

Additional Documentation
Detailed Result Interpretation
Disclaimer

This repository is for educational and portfolio purposes. It must not be used for real medical diagnosis.