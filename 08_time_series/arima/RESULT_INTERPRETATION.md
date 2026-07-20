# ARIMA — Result Interpretation

## 1. Purpose

This document explains how to interpret the ARIMA forecasting outputs.

The model is fitted using historical observations and evaluated using future held-out periods.

---

## 2. Chronological Train-Test Separation

The dataset is split according to time.

Earlier observations are used for model fitting.

Later observations are reserved for testing.

This prevents future information from influencing the historical model.

Random train-test splitting is not appropriate for ordinary forecasting because it can mix future information into the training data.

---

## 3. ARIMA Order

The selected model is represented as:

```text
ARIMA(p, d, q)
```

### `p`

The number of autoregressive lags.

The model uses previous observations as predictive inputs.

### `d`

The number of differencing operations.

Differencing helps remove trend and improve stationarity.

### `q`

The number of moving-average error terms.

The model uses previous forecast errors to improve predictions.

---

## 4. Candidate Order Comparison

The file:

```text
outputs/metrics/candidate_orders.csv
```

contains the tested model orders and their information criteria.

Lower AIC and BIC values are generally preferred when comparing models fitted to the same training series.

A lower information criterion does not guarantee the best held-out forecast performance.

Therefore, the selected model must still be evaluated on the future test period.

---

## 5. Stationarity

A stationary series has statistical properties that remain relatively stable over time.

ARIMA commonly uses differencing when the original series contains trend.

The project saves Augmented Dickey-Fuller results for:

- the original training series;
- the first-differenced series.

A p-value below 0.05 is commonly interpreted as evidence against the unit-root null hypothesis.

Stationarity should not be judged using one statistical test alone.

The time plot, ACF, PACF, trend, and domain context should also be considered.

---

## 6. ACF

The autocorrelation function measures correlation between a series and its lagged values.

It helps reveal:

- serial dependence;
- repeating temporal structure;
- possible moving-average behaviour;
- possible seasonality.

The ACF plot in this project uses the first-differenced training series.

---

## 7. PACF

The partial autocorrelation function measures the relationship between the series and a lag after accounting for shorter lags.

It can help suggest autoregressive structure.

ACF and PACF provide guidance rather than guaranteed ARIMA orders.

Final orders should also be evaluated through diagnostics and held-out forecasts.

---

## 8. AIC

Akaike Information Criterion balances:

- model fit;
- model complexity.

Lower AIC is generally preferred among candidate models fitted to the same data.

AIC is not an accuracy percentage.

It should not be compared across unrelated datasets.

---

## 9. BIC

Bayesian Information Criterion also balances fit and complexity.

BIC usually penalizes additional parameters more strongly than AIC.

A model with lower BIC may be more parsimonious.

The model with the lowest AIC and the model with the lowest BIC may differ.

---

## 10. Forecast Values

The forecast represents the model's estimated future values.

The held-out forecast table contains:

- actual future value;
- ARIMA forecast;
- naïve forecast;
- lower interval bound;
- upper interval bound;
- forecast error.

Forecasts should be evaluated over the complete held-out horizon rather than using only one period.

---

## 11. Forecast Error

Forecast error is calculated as:

```text
Forecast error =
Actual value - Forecast value
```

A positive error means the model forecast was below the actual value.

A negative error means the model forecast was above the actual value.

---

## 12. MAE

Mean Absolute Error reports the average absolute difference between actual and forecast values.

It remains in the original measurement units.

Lower values are better when comparing models on the same series.

---

## 13. RMSE

Root Mean Squared Error gives greater weight to large forecasting errors.

It is useful when large errors are especially costly.

RMSE is expressed in the original target units.

---

## 14. MAPE

Mean Absolute Percentage Error expresses forecast error as a percentage of actual values.

It is easy to communicate but becomes unreliable when actual values are zero or very close to zero.

---

## 15. sMAPE

Symmetric Mean Absolute Percentage Error uses both actual and forecast magnitudes in its denominator.

It reduces some asymmetry of ordinary MAPE.

It can still behave unusually when both values are close to zero.

---

## 16. MASE

Mean Absolute Scaled Error compares the model's absolute forecast error with the average in-sample naïve error.

General interpretation:

- below `1`: better than the naïve scaling benchmark;
- around `1`: similar to the benchmark;
- above `1`: worse than the benchmark.

MASE is useful for comparing performance across series with different units.

---

## 17. Naïve Forecast Comparison

The naïve baseline predicts future periods using the final training observation.

The ARIMA model should ideally improve on the baseline.

If ARIMA performs worse, possible reasons include:

- unsuitable ARIMA order;
- strong seasonality;
- structural change;
- insufficient training history;
- poor stationarity treatment;
- a test horizon dominated by trend changes.

---

## 18. Forecast Intervals

The 95% forecast interval expresses uncertainty around the point forecast.

The actual value may fall outside the interval.

Forecast intervals usually widen as the forecasting horizon increases because uncertainty accumulates.

Interval coverage should be examined across many held-out periods.

---

## 19. Interval Coverage

Forecast interval coverage is the proportion of actual held-out values falling inside the forecast interval.

For a nominal 95% interval, long-run coverage should ideally be reasonably close to 95%.

A very small test set cannot establish reliable interval calibration.

---

## 20. Residuals

Residuals are the differences between observed training values and fitted model estimates.

A well-specified model should leave residuals that:

- fluctuate around zero;
- show limited autocorrelation;
- have relatively stable variability;
- contain no obvious remaining temporal pattern.

Residuals do not need to be perfectly normally distributed for every forecasting use.

---

## 21. Ljung-Box Test

The Ljung-Box test examines whether groups of residual autocorrelations differ from zero.

A small p-value may indicate remaining serial dependence.

That suggests the model has not captured all temporal structure.

A large p-value does not prove the model is perfect.

---

## 22. One-Step Held-Out Example

`predict.py` forecasts the first held-out future period.

It prints:

- forecast date;
- predicted value;
- actual held-out value;
- forecast error;
- forecast interval;
- whether the actual value lies within the interval.

The example demonstrates inference using a saved model.

Overall model quality must be judged using the entire test horizon.

---

## 23. ARIMA vs SARIMA

Ordinary ARIMA models non-seasonal temporal dependence.

SARIMA extends ARIMA using seasonal terms:

```text
SARIMA(p, d, q)(P, D, Q, s)
```

SARIMA should be considered when a series has repeating seasonal structure.

---

## 24. Strengths

ARIMA provides:

- interpretable statistical structure;
- support for trend differencing;
- autocorrelation modelling;
- forecast intervals;
- information criteria;
- established residual diagnostics;
- strong univariate baselines.

---

## 25. Limitations

Its main limitations are:

- linear temporal assumptions;
- sensitivity to order selection;
- ordinary ARIMA's lack of seasonal terms;
- reduced reliability after structural breaks;
- difficulty with complex external drivers;
- increasing uncertainty at longer horizons.

---

## 26. Recommended Improvements

Future improvements should include:

1. Rolling-origin evaluation.
2. Expanding-window forecasts.
3. Additional candidate orders.
4. Automated order selection.
5. Seasonal ARIMA.
6. Exponential-smoothing comparison.
7. Prophet comparison.
8. Forecast interval calibration.
9. Structural-break analysis.
10. Exogenous-variable support.
11. Time-series cross-validation.
12. Model-comparison dashboard.

---

## 27. Final Conclusion

ARIMA models future values using:

- previous observations;
- differencing;
- previous forecast errors.

This project demonstrates:

- chronological data splitting;
- training-only model fitting;
- candidate-order comparison;
- stationarity testing;
- ACF and PACF analysis;
- held-out future forecasting;
- forecast intervals;
- residual diagnostics;
- naïve baseline comparison.

The model should be selected according to held-out forecast performance and diagnostics, not only the lowest training AIC.