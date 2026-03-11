import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from fredapi import Fred

# CONFIGURATION
FRED_API_KEY = "your_fred_api_key_here"
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

fred = Fred(api_key=FRED_API_KEY)

# STEP 1: Fetch Treasury yields
# Maturities: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
series_ids = {
    0.083: "DGS1MO",   # 1 month
    0.25:  "DGS3MO",   # 3 month
    0.5:   "DGS6MO",   # 6 month
    1:     "DGS1",     # 1 year
    2:     "DGS2",     # 2 year
    3:     "DGS3",     # 3 year
    5:     "DGS5",     # 5 year
    7:     "DGS7",     # 7 year
    10:    "DGS10",    # 10 year
    20:    "DGS20",    # 20 year
    30:    "DGS30",    # 30 year
}

# Fetch all series
yields = pd.DataFrame()
for maturity, series_id in series_ids.items():
    data = fred.get_series(series_id, START_DATE, END_DATE)
    yields[maturity] = data

yields = yields.dropna()
yields = yields / 100  # convert from % to decimal

print(f"Fetched {len(yields)} trading days of yield data")
print(f"\nLatest yield curve ({yields.index[-1].date()}):")
for maturity, col in zip(series_ids.keys(), yields.columns):
    print(f"  {maturity:>5}Y: {yields[col].iloc[-1]:.4f}")

# STEP 2: Plot the raw yield curve
maturities = np.array(list(series_ids.keys()))

plt.figure(figsize=(12, 6))

# Plot 4 snapshots: 2020, 2022, 2023, and latest
dates_to_plot = {
    "Jan 2020 (pre-COVID)":  "2020-01-02",
    "Jan 2022 (pre-hikes)":  "2022-01-03",
    "Dec 2023 (peak rates)": "2023-12-29",
    "Dec 2025 (current)":    yields.index[-1],
}

for label, date in dates_to_plot.items():
    row = yields.loc[date]
    plt.plot(maturities, row.values * 100, marker="o", label=label)

plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.title("U.S. Treasury Yield Curve — Historical Snapshots")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("yield_curve_snapshots.png")
plt.show()

# STEP 3: Nelson-Siegel-Svensson Model
# NSS formula: models yield curve with 6 parameters
# beta0 = long-term level
# beta1 = short-term slope
# beta2 = medium-term curvature 1
# beta3 = medium-term curvature 2
# tau1, tau2 = decay factors

def nss(maturity, beta0, beta1, beta2, beta3, tau1, tau2):
    t = maturity
    f1 = (1 - np.exp(-t / tau1)) / (t / tau1)
    f2 = f1 - np.exp(-t / tau1)
    f3 = (1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
    return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3

# Fit NSS to the latest yield curve
maturities = np.array(list(series_ids.keys()))
latest_yields = yields.iloc[-1].values

def nss_error_row(observed):
    def error(params):
        fitted = nss(maturities, *params)
        return np.sum((fitted - observed) ** 2)
    return error

# Initial guess for parameters
x0 = [0.04, -0.01, 0.01, 0.01, 1.0, 5.0]
bounds = [
    (0, 0.2),    # beta0
    (-0.2, 0.2), # beta1
    (-0.2, 0.2), # beta2
    (-0.2, 0.2), # beta3
    (0.1, 10),   # tau1
    (0.1, 30),   # tau2
]

result = minimize(nss_error_row(latest_yields), x0, method="L-BFGS-B", bounds=bounds)
beta0, beta1, beta2, beta3, tau1, tau2 = result.x

print("\nNSS Parameters (latest curve):")
print(f"  beta0 (long-term level):  {beta0:.4f}")
print(f"  beta1 (slope):            {beta1:.4f}")
print(f"  beta2 (curvature 1):      {beta2:.4f}")
print(f"  beta3 (curvature 2):      {beta3:.4f}")
print(f"  tau1 (decay 1):           {tau1:.4f}")
print(f"  tau2 (decay 2):           {tau2:.4f}")

# Plot fitted vs actual
maturity_fine = np.linspace(0.083, 30, 300)
fitted_curve = nss(maturity_fine, beta0, beta1, beta2, beta3, tau1, tau2)

plt.figure(figsize=(12, 6))
plt.plot(maturities, latest_yields * 100, "o", label="Actual yields", markersize=8)
plt.plot(maturity_fine, fitted_curve * 100, "-", label="NSS fitted curve", linewidth=2)
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.title(f"NSS Model Fit — {yields.index[-1].date()}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("nss_fit.png")
plt.show()

# STEP 4: Fit NSS to every day (extract parameter time series)
from tqdm import tqdm

params_list = []
failed = 0

for date, row in yields.iterrows():
    try:
        result = minimize(nss_error_row(row.values), x0, 
                         method="L-BFGS-B", bounds=bounds)
        params_list.append([date] + list(result.x))
    except:
        failed += 1
        continue

print(f"Failed fits: {failed}")

# Convert to DataFrame
params_df = pd.DataFrame(params_list, columns=["date", "beta0", "beta1", "beta2", "beta3", "tau1", "tau2"])
params_df = params_df.set_index("date")

print("\nNSS Parameters (last 5 rows):")
print(params_df.tail())

# Plot parameters over time
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("NSS Parameters Over Time (2015-2025)", fontsize=14)

for ax, col in zip(axes.flatten(), ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]):
    ax.plot(params_df.index, params_df[col])
    ax.set_title(col)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("nss_parameters.png")
plt.show()

# Use only beta0, beta1, beta2 for VAR (tau params are too unstable)
var_params = params_df[["beta0", "beta1", "beta2"]].copy()
print("\nCorrelation between NSS parameters:")
print(var_params.corr().round(3))

# STEP 5: VAR Model to forecast NSS parameters
from statsmodels.tsa.vector_ar.var_model import VAR

# Train/test split — use last 60 days as test
train = var_params.iloc[:-60]
test = var_params.iloc[-60:]

# Fit VAR model — lag order selected automatically
model = VAR(train)
lag_result = model.select_order(maxlags=10)
best_lag = lag_result.aic
print(f"\nOptimal VAR lag order (AIC): {best_lag}")

var_fitted = model.fit(best_lag)
print(var_fitted.summary())

# STEP 6: Forecast next 30 trading days
forecast = var_fitted.forecast(var_params.values[-best_lag:], steps=30)
forecast_df = pd.DataFrame(forecast, columns=["beta0", "beta1", "beta2"])

# Reconstruct forecasted yield curves
print("\nForecasted NSS parameters (next 30 days):")
print(forecast_df.tail())

# Plot forecasted curve vs current
maturity_fine = np.linspace(0.083, 30, 300)
current_curve = nss(maturity_fine, beta0, beta1, beta2, beta3, tau1, tau2)
forecast_curve = nss(maturity_fine, 
                     forecast_df["beta0"].iloc[-1],
                     forecast_df["beta1"].iloc[-1],
                     forecast_df["beta2"].iloc[-1],
                     beta3, tau1, tau2)

plt.figure(figsize=(12, 6))
plt.plot(maturity_fine, current_curve * 100, label="Current (Dec 2025)", linewidth=2)
plt.plot(maturity_fine, forecast_curve * 100, "--", label="VAR Forecast (30 days)", linewidth=2)
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.title("VAR Yield Curve Forecast — 30 Trading Days Ahead")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("var_forecast.png")
plt.show()

# STEP 7: LSTM Model
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Filter to recent regime only — post rate hike era
SEQUENCE_LENGTH = 60
regime_start = "2022-01-01"
recent_params = var_params.loc[regime_start:]
print(f"Training on {len(recent_params)} days from {regime_start}")

# Normalize on recent data only
scaler = MinMaxScaler()
scaled_params = scaler.fit_transform(recent_params.values)

# Create sequences — 60 day lookback window
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_params, SEQUENCE_LENGTH)

# Train/test split
split = int(len(X) * 0.8)
X_train = torch.tensor(X[:split], dtype=torch.float32)
y_train = torch.tensor(y[:split], dtype=torch.float32)
X_test  = torch.tensor(X[split:], dtype=torch.float32)
y_test  = torch.tensor(y[split:], dtype=torch.float32)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# ── Define LSTM network ──────────────────────────────────────────
class YieldCurveLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)

model_lstm = YieldCurveLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

# ── Train ────────────────────────────────────────────────────────
EPOCHS = 100
print("\nTraining LSTM...")
for epoch in range(EPOCHS):
    model_lstm.train()
    optimizer.zero_grad()
    output = model_lstm(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {loss.item():.6f}")

print("Training complete!")

# STEP 8: LSTM Forecast & Comparison
model_lstm.eval()

# Roll forward 30 days
last_sequence = torch.tensor(scaled_params[-SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0)
lstm_forecast = []

with torch.no_grad():
    for _ in range(30):
        pred = model_lstm(last_sequence)
        lstm_forecast.append(pred.squeeze().numpy())
        last_sequence = torch.cat([last_sequence[:, 1:, :], pred.unsqueeze(1)], dim=1)

# Inverse transform back to original scale
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast))
lstm_forecast_df = pd.DataFrame(lstm_forecast, columns=["beta0", "beta1", "beta2"])

print("\nLSTM Forecasted Parameters (next 30 days):")
print(lstm_forecast_df.tail())

# Final comparison plot
maturity_fine = np.linspace(0.083, 30, 300)

current_curve = nss(maturity_fine, beta0, beta1, beta2, beta3, tau1, tau2)

var_curve = nss(maturity_fine,
                forecast_df["beta0"].iloc[-1],
                forecast_df["beta1"].iloc[-1],
                forecast_df["beta2"].iloc[-1],
                beta3, tau1, tau2)

lstm_curve = nss(maturity_fine,
                 lstm_forecast_df["beta0"].iloc[-1],
                 lstm_forecast_df["beta1"].iloc[-1],
                 lstm_forecast_df["beta2"].iloc[-1],
                 beta3, tau1, tau2)

plt.figure(figsize=(12, 6))
plt.plot(maturity_fine, current_curve * 100, label="Current (Dec 2025)", linewidth=2)
plt.plot(maturity_fine, var_curve * 100, "--", label="VAR Forecast (30 days)", linewidth=2)
plt.plot(maturity_fine, lstm_curve * 100, "--", label="LSTM Forecast (30 days)", linewidth=2)
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.title("Yield Curve Forecast — VAR vs LSTM (30 Trading Days)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("var_vs_lstm_forecast.png")
plt.show()