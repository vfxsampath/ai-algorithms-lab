# XGBoost Time-Series — Result Interpretation

## 1. Purpose

This document explains how XGBoost is used for forecasting, how lag features work, how recursive forecasting behaves, and how the outputs should be interpreted.

## 2. XGBoost Is Not Automatically a Time-Series Model

XGBoost accepts rows of features and target values.

It does not automatically know that one observation occurred before another.

Time-series structure is supplied through:

- lag features;
- rolling statistics;
- calendar features;
- chronological splitting.

## 3. Lag Features

Lag features contain earlier observations.

For example:

```text
lag_1 = one month earlier
lag_12 = twelve months earlier
```

A high importance for `lag_1` suggests that the previous observation strongly contributes to forecasts.

A high importance for `lag_12` suggests that annual temporal structure may be useful.

## 4. Rolling Mean

A rolling mean summarizes the recent level.

It can help the model recognize:

- local trend;
- recent baseline;
- short-term smoothing;
- gradual changes.

## 5. Rolling Standard Deviation

Rolling standard deviation measures recent variability.

A large value means the recent observations fluctuate more strongly.

It can help the model distinguish stable and volatile periods.

## 6. Rolling Minimum and Maximum

These features describe the recent range.

They can help the model detect:

- recent peaks;
- recent low values;
- changing local boundaries.

## 7. Cyclical Month Features

Month numbers alone incorrectly make December and January appear far apart.

Sine and cosine transformations represent month as a cycle.

This allows the model to learn recurring yearly patterns more naturally.

## 8. Time Index

The time index increases throughout the series.

It allows the model to learn broad historical progression.

Tree models cannot extrapolate linear trends as naturally as dedicated trend models, so time-index behaviour should be interpreted cautiously.

## 9. Chronological Validation

Candidate configurations are selected using the latest part of the training period.

Earlier feature rows are used for candidate fitting.

Later training rows are used for validation.

The final test period remains untouched.

## 10. Recursive Forecasting

Recursive forecasting predicts one period at a time.

For the first forecast, all lag values are real historical observations.

For later forecasts, some lag values are earlier model predictions.

Therefore, an early forecast error can influence later forecasts.

## 11. Error Accumulation

The error-by-horizon chart shows whether forecast error increases farther into the future.

Increasing errors may result from:

- recursive prediction dependence;
- trend extrapolation difficulty;
- structural changes;
- insufficient lag features;
- missing explanatory variables.

## 12. Candidate Model Selection

Candidate models differ in:

- tree count;
- learning rate;
- tree depth;
- row sampling;
- feature sampling;
- regularization.

The candidate with the lowest chronological validation MAE is selected.

The test period is used only after model selection.

## 13. Number of Estimators

`n_estimators` controls the number of boosted trees.

More trees can improve fitting but increase:

- training time;
- model complexity;
- overfitting risk.

## 14. Learning Rate

The learning rate controls how much each tree contributes.

A smaller value usually requires more trees.

A larger value learns faster but may overfit or miss an optimal solution.

## 15. Maximum Depth

`max_depth` controls tree complexity.

Deeper trees can model more complex interactions.

They can also increase overfitting.

## 16. Subsample

`subsample` controls the proportion of training rows used by each tree.

Values below one introduce randomness and can reduce overfitting.

## 17. Column Sampling

`colsample_bytree` controls the proportion of features used by each tree.

It can reduce feature dependence and improve generalization.

## 18. Regularization

`reg_alpha` applies L1 regularization.

`reg_lambda` applies L2 regularization.

Regularization discourages overly complex tree weights.

## 19. Feature Importance

Feature importance shows how strongly features contributed to the fitted model.

It does not show:

- causal impact;
- whether an increase raises or lowers the prediction;
- reliable individual-record explanations.

Possible improvements include:

- permutation importance;
- SHAP values;
- partial-dependence analysis.

## 20. MAE

Mean Absolute Error represents the average absolute forecast miss.

It remains in the original target units.

## 21. RMSE

Root Mean Squared Error gives more weight to large forecast errors.

It is useful when large mistakes are especially costly.

## 22. MAPE

MAPE expresses errors as percentages.

It can become unreliable when actual values approach zero.

## 23. sMAPE

sMAPE uses the magnitudes of both actual and predicted values.

It reduces some asymmetry in ordinary MAPE.

## 24. Seasonal MASE

Seasonal MASE compares model error with an in-sample seasonal-naïve benchmark.

General interpretation:

- below 1: better than the seasonal-naïve scaling benchmark;
- around 1: similar;
- above 1: worse.

## 25. Baseline Comparison

The model should be compared with:

- ordinary naïve forecasting;
- seasonal-naïve forecasting.

A complex boosted-tree model is not justified when it cannot outperform a simple baseline.

## 26. One-Step Example

`predict.py` forecasts the first held-out month.

All lag and rolling features in this example come from actual historical training observations.

It prints:

- forecast date;
- forecast value;
- actual value;
- forecast error;
- exact feature values used.

## 27. XGBoost vs ARIMA

| Aspect | XGBoost | ARIMA |
|---|---|---|
| Main mechanism | Boosted trees | Autoregression and errors |
| Feature engineering | Required | Mostly internal |
| Nonlinear patterns | Strong | Limited |
| External variables | Easy to add | Requires ARIMAX/SARIMAX |
| Trend extrapolation | Often difficult | Differencing-based |
| Forecast intervals | Not included here | Native |
| Interpretation | Feature-based | Lag-order based |

## 28. XGBoost vs Prophet

| Aspect | XGBoost | Prophet |
|---|---|---|
| Main structure | Engineered features | Trend and seasonality |
| Calendar features | Manually created | Built in |
| Nonlinear interactions | Strong | More structured |
| Changepoints | Not explicit | Explicit |
| Recursive forecasting | Used here | Not required |
| External regressors | Supported | Supported |

## 29. Strengths

XGBoost provides:

- nonlinear modelling;
- feature interactions;
- flexible external predictors;
- regularization;
- strong tabular-data performance;
- reusable supervised-learning workflows.

## 30. Limitations

Its limitations include:

- required lag engineering;
- recursive-error accumulation;
- limited trend extrapolation;
- no native awareness of chronology;
- point forecasts without uncertainty;
- possible overfitting;
- sensitivity to feature design.

## 31. Recommended Improvements

1. Add expanding-window validation.
2. Add rolling-origin validation.
3. Compare recursive and direct forecasting.
4. Add external explanatory variables.
5. Add early stopping.
6. Add SHAP analysis.
7. Add conformal prediction intervals.
8. Compare alternative lag sets.
9. Add seasonal differencing features.
10. Compare ARIMA, Prophet, SARIMA, Holt-Winters, and XGBoost.
11. Build a forecasting comparison dashboard.
12. Add data-drift monitoring.

## 32. Final Conclusion

XGBoost converts forecasting into supervised regression using historical features.

This project demonstrates:

- chronological splitting;
- lag engineering;
- rolling features;
- time-aware validation;
- candidate tuning;
- recursive forecasting;
- baseline comparison;
- feature importance;
- one-step held-out inference.

The model should be selected only when its held-out performance and operational flexibility justify its additional complexity.