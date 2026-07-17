# Isolation Forest Anomaly Detection

## Overview

Isolation Forest is an unsupervised anomaly-detection algorithm.

Instead of modelling normal observations directly, it repeatedly partitions the feature space using random features and random split values.

Unusual observations tend to be isolated using fewer partitions than common observations.

## Business Problem

An organization wants to identify unusual records within operational data.

Potential applications include:

- transaction anomaly detection;
- fraud-screening support;
- equipment-condition monitoring;
- unusual customer behaviour;
- process-deviation detection;
- data-quality monitoring;
- cybersecurity-event prioritization.

This project uses the scikit-learn Wine dataset as a numerical anomaly-detection demonstration.

## Correct Train-Test Design

The original feature dataset is divided into:

- training data;
- held-out normal test data.

The scaler and Isolation Forest are fitted only on the training set.

Controlled synthetic anomalies are added only to the held-out evaluation set.

The synthetic anomalies are never used to fit the model.

## Evaluation Dataset

The evaluation data contain:

- genuine held-out observations from the original dataset;
- controlled synthetic anomalies created by strongly perturbing selected features.

The synthetic labels allow educational calculation of:

- precision;
- recall;
- F1-score;
- ROC-AUC;
- average precision;
- confusion matrix.

These results do not represent performance on real-world anomalies.

## Model Pipeline

```python
Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "detector",
            IsolationForest(
                n_estimators=300,
                contamination=0.05,
                max_samples="auto",
                max_features=1.0,
                bootstrap=False,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)
Main Parameters
n_estimators=300

Builds 300 isolation trees.

More trees generally produce a more stable average anomaly score, with additional computational cost.

contamination=0.05

Defines the expected proportion of anomalies used to set the decision threshold.

This does not mean that exactly 5% of future records are truly anomalous.

max_samples="auto"

Selects the sample size used to build each isolation tree according to scikit-learn's automatic setting.

max_features=1.0

Each tree can draw from all available features.

bootstrap=False

Samples are drawn without replacement when constructing trees.

Prediction Labels

Scikit-learn returns:

Raw prediction	Meaning
1	Inlier
-1	Outlier

This project converts the labels to:

Project label	Meaning
0	Normal
1	Anomaly
Decision Function

The decision function is positive for observations on the inlier side of the learned threshold and negative for observations on the outlier side.

For easier reporting, this project defines:

anomaly_score = -decision_function

Therefore, larger project anomaly scores indicate more anomalous observations.

Evaluation Metrics

The project reports:

accuracy;
balanced accuracy;
anomaly precision;
anomaly recall;
anomaly F1-score;
normal precision;
normal recall;
ROC-AUC;
average precision;
confusion matrix;
anomaly-score distributions.
Output Files
outputs/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── anomaly_score_distribution.png
│   └── pca_anomaly_visualization.png
├── metrics/
│   ├── training_summary.json
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix_values.csv
│   └── anomaly_score_summary.csv
└── predictions/
    └── evaluation_predictions.csv