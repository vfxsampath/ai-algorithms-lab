# SARIMA — Result Interpretation

## 1. Purpose

This document explains the seasonal model structure, forecasts, metrics, diagnostics, and practical limitations.

## 2. Model Structure

SARIMA is expressed as:

```text
SARIMA(p, d, q)(P, D, Q, s)
```

The first group models non-seasonal temporal relationships.

The second group models repeating seasonal relationships.

## 3. Seasonal Period

This project uses:

```text
s = 12
```

because the data are monthly and one yearly cycle contains 12 months.

The seasonal period should come from the data frequency and domain, not arbitrary tuning alone.

## 4. Seasonal Autoregression

`P` represents seasonal autoregressive terms.

For monthly data, a seasonal lag may connect the current month with observations 12 months earlier.

## 5. Seasonal Differencing

`D` represents seasonal differencing.

With `D=1` and `s=12`, the model uses differences between an observation and the corresponding observation one year earlier.

Seasonal differencing can reduce repeating yearly patterns.

Excessive differencing may remove meaningful information.

## 6. Seasonal Moving Average

`Q` represents seasonal moving-average error terms.

These terms model relationships between the current outcome and forecasting errors from previous seasonal cycles.

## 7. Candidate Models

The candidate-model file contains:

- ordinary order;
- seasonal order;
- AIC;
- BIC;
- likelihood;
- convergence status.

The candidate with the lowest successful AIC is selected.

That selection must still be verified using held-out forecasts.

## 8. Seasonal-Naïve Baseline

Seasonal-naïve forecasting repeats the value from the same month of the previous year.

For strongly seasonal data, this is usually a more meaningful baseline than repeating only the final training value.

SARIMA should ideally outperform the seasonal-naïve model.

## 9. Seasonal MASE

Seasonal MASE compares SARIMA’s forecast error with the average seasonal-naïve error inside the training data.

General interpretation:

- below `1` — better than the seasonal-naïve scaling benchmark;
- around `1` — similar;
- above `1` — worse.

## 10. Forecast Intervals

The lower and upper values describe the estimated uncertainty around each forecast.

Intervals generally widen as the forecast horizon increases.

Actual coverage should be measured across the full test horizon.

## 11. Residuals

A well-specified seasonal model should leave residuals with:

- mean near zero;
- limited remaining autocorrelation;
- limited seasonal structure;
- reasonably stable variance.

Remaining seasonal residual patterns indicate incomplete modelling.

## 12. Ljung-Box Test

The Ljung-Box test examines whether groups of residual autocorrelations differ from zero.

A small p-value suggests remaining temporal dependence.

This can indicate that additional non-seasonal or seasonal structure remains unmodelled.

## 13. ACF and PACF

The ACF and PACF charts are created after ordinary and seasonal differencing.

Seasonal peaks near lags such as 12, 24, or 36 may indicate repeating yearly dependence.

These charts guide model design but do not guarantee the optimal order.

## 14. SARIMA vs ARIMA

| Aspect | ARIMA | SARIMA |
|---|---|---|
| Trend differencing | Yes | Yes |
| Seasonal differencing | No explicit seasonal term | Yes |
| Seasonal AR terms | No | Yes |
| Seasonal MA terms | No | Yes |
| Complexity | Lower | Higher |
| Best use | Non-seasonal series | Seasonal series |

SARIMA is not automatically better. It should be selected only when seasonal terms improve future performance.

## 15. SARIMA vs Prophet

| Aspect | SARIMA | Prophet |
|---|---|---|
| Main mechanism | Lag and error relationships | Trend and seasonal decomposition |
| Seasonal structure | Seasonal ARIMA terms | Fourier-based components |
| Trend changes | Indirect | Explicit changepoints |
| Stationarity | Important | Less explicit |
| Holiday effects | Requires regressors | Native support |
| Diagnostics | Strong residual tools | Component-focused |

## 16. One-Step Example

`predict.py` forecasts the first held-out future observation.

It shows:

- forecast date;
- estimated value;
- actual value;
- forecast error;
- 95% interval;
- interval coverage.

The complete test horizon is more important than one example.

## 17. Strengths

SARIMA provides:

- explicit seasonal modelling;
- interpretable seasonal orders;
- forecast intervals;
- established statistical diagnostics;
- strong seasonal baselines;
- support for trend and seasonal differencing.

## 18. Limitations

Its limitations include:

- computational model selection;
- parameter sensitivity;
- linearity assumptions;
- required seasonal history;
- possible over-differencing;
- reduced performance after structural changes;
- complex interpretation for large orders.

## 19. Recommended Improvements

1. Rolling-origin evaluation.
2. Larger candidate-order search.
3. Compare AIC and held-out selection.
4. Automatic seasonal-order selection.
5. Exponential-smoothing comparison.
6. Direct ARIMA, SARIMA, and Prophet dashboard.
7. Exogenous-variable modelling with SARIMAX.
8. Forecast-interval calibration.
9. Structural-break analysis.
10. Time-series cross-validation.

## 20. Final Conclusion

SARIMA extends ARIMA by modelling repeating seasonal relationships.

This project demonstrates:

- chronological splitting;
- ordinary and seasonal differencing;
- seasonal parameter selection;
- held-out future forecasts;
- seasonal-naïve comparison;
- uncertainty intervals;
- residual diagnostics;
- one-period inference.

The final decision should be based on future forecast performance, residual quality, simplicity, and operational usefulness.