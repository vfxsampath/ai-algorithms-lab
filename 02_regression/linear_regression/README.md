# Linear Regression

## Overview

Linear Regression is a supervised machine-learning algorithm used to predict a continuous numerical value.

The algorithm models the relationship between one or more input features and a continuous target variable by fitting a linear equation to the observed data.

## Business Problem

A property company wants to estimate house prices using characteristics such as:

- Median income
- Number of rooms
- Population
- Geographic location
- Housing age

The predicted values could support:

- Property valuation
- Investment decisions
- Pricing strategy
- Market analysis
- Risk assessment

## Learning Objectives

This example demonstrates:

- Regression modelling
- Train-test splitting
- Feature preprocessing
- Model training
- Continuous-value prediction
- Residual analysis
- Feature coefficient interpretation
- Regression evaluation metrics
- Saving model outputs

## Dataset

This implementation uses the California Housing dataset from scikit-learn.

The target variable represents median house value.

## Algorithm Workflow

1. Load the dataset
2. Inspect features and target
3. Split the data into training and testing sets
4. Standardize input features
5. Train the Linear Regression model
6. Generate predictions
7. Calculate evaluation metrics
8. Analyze residuals
9. Visualize predicted versus actual values
10. Interpret model coefficients

## Evaluation Metrics

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R-squared
- Adjusted R-squared

## Strengths

- Easy to understand and explain
- Fast to train
- Useful as a baseline model
- Coefficients support interpretation
- Works well when relationships are approximately linear

## Limitations

- Assumes a linear relationship
- Sensitive to influential outliers
- Can be affected by multicollinearity
- May underperform when relationships are nonlinear
- Assumes relatively stable error variance

## Business Interpretation

The model coefficients indicate how the predicted target changes when one feature increases while the other features remain constant.

However, coefficients should not automatically be interpreted as causal effects.

## Technology

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Files

```text
linear_regression/
├── README.md
├── src/
│   ├── train.py
│   └── predict.py
├── data/
└── outputs/
    ├── figures/
    └── metrics/