# SARIMA Time-Series Forecasting

## Overview

SARIMA stands for Seasonal AutoRegressive Integrated Moving Average.

It extends ARIMA by adding explicit seasonal autoregressive, differencing, and moving-average terms.

The model is represented as:

```text
SARIMA(p, d, q)(P, D, Q, s)
```

## Components

### Non-Seasonal Terms

- `p` — autoregressive order;
- `d` — differencing order;
- `q` — moving-average order.

### Seasonal Terms

- `P` — seasonal autoregressive order;
- `D` — seasonal differencing order;
- `Q` — seasonal moving-average order;
- `s` — number of periods in one seasonal cycle.

For monthly data:

```text
s = 12
```

represents yearly seasonality.

## Business Applications

SARIMA can support:

- monthly sales forecasting;
- seasonal demand planning;
- inventory forecasting;
- energy-demand forecasting;
- passenger-volume forecasting;
- workforce planning;
- environmental forecasting.

## Dataset

This implementation uses monthly atmospheric CO₂ data from statsmodels.

The same data and 24-month test horizon are used in the ARIMA and Prophet folders so the models can later be compared fairly.

## Correct Time-Series Split

The time series is divided chronologically:

```text
Historical observations → training data
Latest 24 observations → held-out future data
```

No random shuffling is used.

## Model Selection

Several candidate non-seasonal and seasonal orders are fitted using training data only.

The candidate with the lowest AIC is selected.

Held-out future performance is then evaluated separately.

## Seasonal Differencing

Seasonal differencing calculates:

```text
Current value - Value from 12 months earlier
```

This can reduce repeating yearly structure before autoregressive and moving-average terms are estimated.

## Forecasting Baselines

SARIMA is compared with:

1. ordinary naïve forecasting;
2. seasonal-naïve forecasting.

The seasonal-naïve model predicts each future month using the value from the corresponding month of the previous year.

## Evaluation Metrics

- MAE
- MSE
- RMSE
- MAPE
- sMAPE
- Seasonal MASE
- Forecast-interval coverage
- Improvement over seasonal-naïve forecasting

## Residual Diagnostics

The project examines:

- residual values over time;
- residual distribution;
- residual mean;
- Ljung-Box tests;
- remaining temporal dependence.

## Output Files

```text
outputs/
├── figures/
│   ├── train_test_split.png
│   ├── held_out_forecast.png
│   ├── baseline_comparison.png
│   ├── seasonal_differencing.png
│   ├── acf.png
│   ├── pacf.png
│   ├── residuals_over_time.png
│   └── residual_distribution.png
├── metrics/
│   ├── training_summary.json
│   ├── candidate_models.csv
│   ├── metrics.json
│   ├── residual_summary.json
│   └── ljung_box_test.csv
└── predictions/
    └── test_forecasts.csv
```

## Run

```powershell
python 08_time_series/sarima/src/train.py
python 08_time_series/sarima/src/evaluate.py
python 08_time_series/sarima/src/predict.py
```

## Results

### Held-Out Forecast

![Held-Out Forecast](outputs/figures/held_out_forecast.png)

### Baseline Comparison

![Baseline Comparison](outputs/figures/baseline_comparison.png)

### Seasonal Differencing

![Seasonal Differencing](outputs/figures/seasonal_differencing.png)

### ACF

![ACF](outputs/figures/acf.png)

### PACF

![PACF](outputs/figures/pacf.png)

## Strengths

- Explicitly models seasonal autocorrelation.
- Supports trend and seasonal differencing.
- Produces uncertainty intervals.
- Offers interpretable orders.
- Provides strong residual diagnostics.
- Can outperform ordinary ARIMA on seasonal series.

## Limitations

- More parameters than ordinary ARIMA.
- Candidate fitting can be computationally expensive.
- Sensitive to seasonal-period assumptions.
- Requires sufficient seasonal history.
- Primarily models linear temporal relationships.
- Structural changes can reduce accuracy.
- Incorrect differencing may remove useful information.

## Additional Documentation

- [Detailed Result Interpretation](RESULT_INTERPRETATION.md)