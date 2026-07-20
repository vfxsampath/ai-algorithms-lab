# Prophet — Result Interpretation

## 1. Purpose

This document explains how to interpret Prophet forecasts, components, metrics, and limitations.

The model is fitted using historical observations and evaluated using future held-out periods.

---

## 2. Chronological Separation

Earlier observations are used for fitting.

The latest 24 monthly observations are held out for evaluation.

Future observations are never included during model training.

This prevents time-based data leakage.

---

## 3. Forecast

The `yhat` value is Prophet's point forecast.

It represents the model's central estimate for a future date.

It is not guaranteed to equal the eventual observation.

---

## 4. Forecast Interval

`yhat_lower` and `yhat_upper` represent the forecast uncertainty interval.

A 95% interval does not mean that each individual forecast has a guaranteed 95% probability of being correct.

Coverage should be evaluated across multiple future periods.

---

## 5. Trend

The trend component represents long-term movement in the series.

It excludes the seasonal contribution.

An increasing trend means the underlying level is estimated to rise over time.

---

## 6. Changepoints

Changepoints are locations where the trend slope may change.

They do not automatically represent known business events.

They indicate that the fitted model found evidence supporting a change in trend behaviour.

---

## 7. Changepoint Prior Scale

The model uses:

```text
changepoint_prior_scale = 0.05
```

A smaller value creates a smoother and less flexible trend.

A larger value allows the trend to follow historical changes more closely.

Too little flexibility may underfit.

Too much flexibility may overfit temporary variation.

---

## 8. Yearly Seasonality

Yearly seasonality represents patterns that repeat approximately every year.

For monthly data, this may capture changes associated with months or seasons.

It assumes that seasonal structure observed in training continues into the future.

---

## 9. Additive Seasonality

The model uses additive seasonality.

Conceptually:

```text
Forecast =
Trend + Seasonal Effect
```

A seasonal contribution of `+1.5` means the seasonal component raises the forecast by approximately 1.5 target units relative to the trend.

---

## 10. Forecast Error

Forecast error is calculated as:

```text
Actual value - Forecast value
```

Positive error means the model forecast was too low.

Negative error means it was too high.

---

## 11. MAE

Mean Absolute Error reports the average absolute forecast error.

It remains in the original target units.

Lower values are preferred when comparing models on the same dataset.

---

## 12. RMSE

Root Mean Squared Error gives more weight to larger forecast misses.

It is useful when occasional large errors are particularly undesirable.

---

## 13. MAPE

Mean Absolute Percentage Error expresses average forecast errors as percentages.

It is easily communicated but can be unstable when actual values are zero or close to zero.

---

## 14. sMAPE

Symmetric MAPE uses the combined magnitude of actual and forecast values.

It reduces some asymmetry found in standard MAPE.

---

## 15. MASE

Mean Absolute Scaled Error compares Prophet's error with a naïve in-sample benchmark.

General interpretation:

- below `1`: better than the naïve scaling benchmark;
- around `1`: similar;
- above `1`: worse.

---

## 16. Naïve Comparison

The naïve forecast repeats the final training observation across the future horizon.

A sophisticated forecasting model should be compared with this simple alternative.

When Prophet cannot outperform it, additional model complexity may not be justified.

---

## 17. Interval Coverage

Interval coverage measures the proportion of actual held-out observations inside the forecast interval.

A nominal 95% interval should ideally achieve reasonably close long-term coverage.

A 24-period test set is too small to prove reliable interval calibration.

---

## 18. Forecast Components

The component chart separates:

- trend;
- yearly seasonality;
- any configured event or holiday effects.

This improves interpretability by showing how the final forecast was constructed.

---

## 19. Residuals

Held-out residuals represent the difference between actual future observations and Prophet forecasts.

Useful residual behaviour includes:

- average close to zero;
- no obvious systematic trend;
- no persistent underprediction or overprediction;
- reasonably stable variation.

Patterns in residuals indicate information the model has not captured.

---

## 20. One-Period Example

`predict.py` selects the first held-out month.

It reports:

- date;
- point forecast;
- actual value;
- trend estimate;
- forecast error;
- uncertainty interval;
- whether the actual value falls inside the interval.

Overall performance must be judged using the entire held-out horizon.

---

## 21. Prophet vs ARIMA

| Aspect | Prophet | ARIMA |
|---|---|---|
| Main structure | Trend and seasonality components | Autoregression and error dependence |
| Trend changes | Explicit changepoints | Differencing and temporal terms |
| Seasonality | Built-in additive or multiplicative | SARIMA required |
| Holiday effects | Native support | Requires external regressors |
| Stationarity requirement | Less explicit | Usually important |
| Interpretability | Components and changepoints | ARIMA orders and coefficients |
| Best use | Business series with trend and seasonality | Strong autocorrelation structure |

Neither method is universally better.

Both should be evaluated on the same future periods.

---

## 22. Strengths

Prophet provides:

- clear trend decomposition;
- interpretable seasonal effects;
- changepoint flexibility;
- uncertainty intervals;
- holiday support;
- accessible forecasting workflows.

---

## 23. Limitations

The main limitations are:

- dependence on historical trend continuation;
- possible overfitting or underfitting of changepoints;
- possible underperformance against simpler methods;
- limited modelling of complex short-term autocorrelation;
- sensitivity to structural breaks;
- uncertainty assumptions that require validation.

---

## 24. Recommended Improvements

Future improvements should include:

1. Prophet historical cross-validation.
2. Changepoint-prior tuning.
3. Seasonality-prior tuning.
4. Additive versus multiplicative comparison.
5. ARIMA and Prophet side-by-side evaluation.
6. Rolling-origin evaluation.
7. Holiday-effect demonstration.
8. Additional regressors.
9. Forecast-interval calibration.
10. Structural-break analysis.
11. Model comparison dashboard.
12. Ensemble forecasting.

---

## 25. Final Conclusion

Prophet forecasts time series through interpretable trend and seasonal components.

This implementation demonstrates:

- chronological data preparation;
- held-out future evaluation;
- yearly seasonality;
- trend changepoints;
- forecast intervals;
- baseline comparison;
- residual analysis;
- one-period held-out forecasting.

Prophet should be selected because it performs well for the real forecasting requirement, not merely because it is easy to use.