# Logistic Regression

## Overview

Logistic Regression is a supervised machine-learning algorithm used primarily for binary and multiclass classification.

Despite its name, it is a classification algorithm. It estimates the probability that an observation belongs to a particular class.

## Business Problem

A telecommunications company wants to predict whether a customer is likely to leave its service.

The model can classify customers as:

- 0: Customer is unlikely to leave
- 1: Customer is likely to leave

The company could use the prediction to prioritize retention campaigns.

## Learning Objectives

This example demonstrates:

- Binary classification
- Data preprocessing
- Feature scaling
- Train-test splitting
- Probability prediction
- Threshold-based classification
- Confusion-matrix interpretation
- Precision, recall and F1-score
- ROC-AUC evaluation

## Dataset

Initial version:

- Scikit-learn Breast Cancer Wisconsin dataset

Future business version:

- Customer churn dataset

## Algorithm Workflow

1. Load the dataset
2. Inspect data quality
3. Separate features and target
4. Create training and testing sets
5. Standardize numerical features
6. Train Logistic Regression
7. Generate class probabilities
8. Evaluate model performance
9. Visualize the confusion matrix and ROC curve
10. Interpret results for a business audience

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Strengths

- Easy to understand
- Fast to train
- Produces probabilities
- Suitable as a baseline model
- Coefficients can support interpretation

## Limitations

- Assumes a relatively linear decision boundary
- Sensitive to highly correlated predictors
- May underperform on complex nonlinear relationships
- Requires appropriate preprocessing

## Technology

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Files

```text
logistic_regression/
├── README.md
├── notebook.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
└── outputs/