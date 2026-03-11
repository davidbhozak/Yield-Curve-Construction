# Yield Curve Construction & Forecasting

Built a yield curve model using real US Treasury data, then forecasted future curve shapes using VAR and LSTM.

The yield curve is one of the most watched indicators in fixed income — it shows the relationship between interest rates and maturity. An inverted curve (short rates > long rates) has preceded every US recession since 1960.

## What it does
- Fits the Nelson-Siegel-Svensson model to real Treasury yields
- Forecasts curve evolution using VAR(10) and LSTM (PyTorch)
- Captures level, slope, and curvature factors separately

## Results
Latest curve (Dec 2025): 10Y = 4.18%, 30Y = 4.84% — normal upward slope after 2022-2023 inversion.

## Stack
Python, pandas, numpy, PyTorch, scipy, matplotlib

