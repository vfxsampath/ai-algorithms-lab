# Random Forest Classification

## Overview

Random Forest is an ensemble machine-learning algorithm that combines predictions from multiple Decision Trees.

Each tree is trained using a bootstrap sample of the training data and a randomized subset of features. The final classification is based on aggregated predictions across the forest.

This reduces the instability and overfitting risk commonly associated with a single Decision Tree.

## Business Problem

A healthcare decision-support system needs to classify observations as malignant or benign using numerical diagnostic measurements.

The model can support:

- preliminary risk screening;
- case prioritization;
- expert decision support;      
- feature-importance analysis;
- comparison with simpler classification models.

This project is an educational machine-learning demonstration and not a clinically validated diagnostic system.

## Dataset

The project uses the Breast Cancer Wisconsin dataset provided by scikit-learn.

Target classes:

| Value | Class |
|---|---|
| 0 | Malignant |
| 1 | Benign |

The same dataset is used for Logistic Regression, Decision Tree, and Random Forest so their performance can be compared fairly.

## Learning Objectives

This project demonstrates:

# Random Forest Classification

## Overview

Random Forest is an ensemble machine-learning algorithm that combines predictions from multiple Decision Trees.

Each tree is trained using a bootstrap sample of the training data and a randomized subset of features. The final classification is based on aggregated predictions across the forest.

This reduces the instability and overfitting risk commonly associated with a single Decision Tree.

## Business Problem

A healthcare decision-support system needs to classify observations as malignant or benign using numerical diagnostic measurements.

The model can support:

- preliminary risk screening;
- case prioritization;
- expert decision support;
- feature-importance analysis;
- comparison with simpler classification models.

This project is an educational machine-learning demonstration and not a clinically validated diagnostic system.

## Dataset

The project uses the Breast Cancer Wisconsin dataset provided by scikit-learn.

Target classes:

| Value | Class |
|---|---|
| 0 | Malignant |
| 1 | Benign |

The same dataset is used for Logistic Regression, Decision Tree, and Random Forest so their performance can be compared fairly.

## Learning Objectives

This project demonstrates:

- ensemble learning;
- bootstrap aggregation;
- randomized feature selection;
- multiple Decision Trees;
- class weighting;
- out-of-bag evaluation;
- probability prediction;
- class-specific evaluation;
- feature importance;
- reusable training, evaluation, and prediction workflows.

## Model Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
Why These Parameters Were Used
n_estimators=300

Builds 300 Decision Trees.

More trees generally provide more stable ensemble predictions, although they increase training and inference cost.

max_features="sqrt"

Each split considers the square root of the total number of features.

This introduces diversity among trees.

bootstrap=True

Each tree is trained on a sample drawn with replacement from the training data.

oob_score=True

Training records not selected in a tree's bootstrap sample are used to estimate out-of-bag performance.

class_weight="balanced"

Class weights are adjusted inversely according to class frequencies.

min_samples_split=5

A node needs at least five samples before it can be divided.

min_samples_leaf=2

Each terminal leaf must contain at least two observations.

Workflow
Load the dataset.
Separate features and target.
Create stratified training and test sets.
Train the Random Forest.
Save the trained model.
Save held-out test data.
Calculate out-of-bag performance.
Load the model independently.
Generate predictions and probabilities.
Calculate classification metrics.
Create evaluation visualizations.
Save feature-importance results.
Generate a prediction for a new record.
Evaluation Metrics

The evaluation includes:

accuracy;
balanced accuracy;
precision;
recall;
F1-score;
ROC-AUC;
confusion matrix;
precision–recall curve;
class-specific malignant and benign metrics;
out-of-bag score.
Output Files
outputs/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
├── metrics/
│   ├── metrics.json
│   ├── training_summary.json
│   ├── classification_report.txt
│   ├── confusion_matrix_values.csv
│   └── feature_importance.csv
└── predictions/
    └── test_predictions.csv
Strengths
Strong predictive performance.
Handles nonlinear relationships.
Captures feature interactions.
Reduces variance compared with one Decision Tree.
Requires little feature scaling.
Supports class probabilities.
Provides feature-importance scores.
Can estimate out-of-bag performance.
Works with many features and observations.
Limitations
Less interpretable than one Decision Tree.
Larger model size.
Slower prediction than simpler models.
Impurity-based importance can be biased.
Probability outputs may require calibration.
Many trees can increase computational cost.
Strong performance does not establish causality.
Run the Project

Run the files in this order:

python 01_classification/random_forest/src/train.py
python 01_classification/random_forest/src/evaluate.py
python 01_classification/random_forest/src/predict.py
Results
Confusion Matrix

ROC Curve

Precision–Recall Curve

Feature Importance

Additional Documentation
Detailed Result Interpretation
Comparison Context

This model should be compared with:

Logistic Regression;
Decision Tree Classification;
K-Nearest Neighbors;
Support Vector Machine;
XGBoost.
Disclaimer

This repository is for educational and portfolio purposes. The model has not been externally or clinically validated and must not be used for real medical diagnosis.