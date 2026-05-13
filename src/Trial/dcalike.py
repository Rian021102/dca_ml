import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit

# =====================================================
# 1. Load same dataset
# =====================================================
path = "/home/rian/python_project/myvenv/dca_ml/data/selected_data_02.csv"

df = pd.read_csv(path)
df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
df = df.sort_values("TEST_DATE").dropna(subset=["OIL"]).reset_index(drop=True)
df = df[df["OIL"] > 0].reset_index(drop=True)

# Smooth oil rate to capture decline trend
df["OIL_SMOOTH"] = df["OIL"].rolling(window=7, min_periods=1).median()

q = df["OIL_SMOOTH"].values.astype(float)
dates = df["TEST_DATE"]

# Time index in days from first date
df["t"] = (df["TEST_DATE"] - df["TEST_DATE"].iloc[0]).dt.days
t = df["t"].values.astype(float)

# =====================================================
# 2. Random Forest feature engineering
# =====================================================
def build_features(q_series, idx):
    arr = np.asarray(q_series[: idx + 1], dtype=float)
    arr = np.clip(arr, 1e-6, None)
    logq = np.log(arr)

    features = {
        "t": idx,
        "q_current": arr[-1],
        "logq_current": logq[-1],
    }

    for w in [3, 7, 14, 30]:
        sub = arr[-w:]
        lsub = logq[-w:]

        features[f"roll_mean_{w}"] = np.mean(sub)
        features[f"roll_median_{w}"] = np.median(sub)
        features[f"roll_std_{w}"] = np.std(sub) if len(sub) > 1 else 0.0

        if len(lsub) >= 2:
            x = np.arange(len(lsub))
            features[f"log_slope_{w}"] = np.polyfit(x, lsub, 1)[0]
        else:
            features[f"log_slope_{w}"] = 0.0

    if len(arr) >= 2:
        ratios = arr[1:] / arr[:-1]
        for w in [3, 7, 14, 30]:
            r = ratios[-w:]
            features[f"ratio_mean_{w}"] = np.mean(r)
            features[f"ratio_median_{w}"] = np.median(r)
    else:
        for w in [3, 7, 14, 30]:
            features[f"ratio_mean_{w}"] = 1.0
            features[f"ratio_median_{w}"] = 1.0

    return features

# =====================================================
# 3. Build ML training data
# =====================================================
rows = []
targets = []
target_indices = []

minimum_history = 30

for i in range(minimum_history, len(q) - 1):
    rows.append(build_features(q, i))
    targets.append(np.log(q[i + 1] / q[i]))
    target_indices.append(i + 1)

X = pd.DataFrame(rows)
y = np.array(targets)
target_indices = np.array(target_indices)

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y[:split]
y_test = y[split:]

# =====================================================
# 4. Train Random Forest
# =====================================================
rf = RandomForestRegressor(
    n_estimators=800,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =====================================================
# 5. Random Forest test prediction
# =====================================================
pred_log_decline_test = rf.predict(X_test)

test_current_q = q[target_indices[split:] - 1]
test_actual_q = q[target_indices[split:]]
test_pred_q = test_current_q * np.exp(pred_log_decline_test)

mae = mean_absolute_error(test_actual_q, test_pred_q)
rmse = mean_squared_error(test_actual_q, test_pred_q) ** 0.5
r2 = r2_score(test_actual_q, test_pred_q)

# =====================================================
# 6. Recursive Random Forest forecast
# =====================================================
forecast_horizon = 100
future_q_rf = list(q.copy())

recent_log_declines = np.diff(np.log(np.clip(q[-30:], 1e-6, None)))
negative_recent_declines = recent_log_declines[recent_log_declines < 0]

if len(negative_recent_declines) > 0:
    terminal_decline = np.median(negative_recent_declines)
else:
    terminal_decline = np.median(recent_log_declines)

for step in range(forecast_horizon):
    idx = len(future_q_rf) - 1

    X_future = pd.DataFrame([build_features(future_q_rf, idx)])
    X_future = X_future[X.columns]

    predicted_log_decline = float(rf.predict(X_future)[0])

    # Decline-only constraint
    predicted_log_decline = min(predicted_log_decline, 0.0)

    # Blend ML decline with recent terminal decline
    if terminal_decline < 0:
        predicted_log_decline = 0.7 * predicted_log_decline + 0.3 * terminal_decline

    # Avoid unrealistic collapse or flat forecast
    predicted_log_decline = np.clip(predicted_log_decline, -0.05, -1e-5)

    next_q = future_q_rf[-1] * np.exp(predicted_log_decline)
    future_q_rf.append(next_q)

rf_forecast = np.array(future_q_rf[-forecast_horizon:])

# =====================================================
# 7. Exponential Arps fitting and forecasting
# =====================================================
def exponential_arps(t, qi, D):
    return qi * np.exp(-D * t)

t_fit = t
q_fit = q

qi_guess = q_fit[0]
D_guess = 0.001

params, covariance = curve_fit(
    exponential_arps,
    t_fit,
    q_fit,
    p0=[qi_guess, D_guess],
    bounds=([0, 0], [np.inf, 1]),
    maxfev=10000
)

qi_arps, D_arps = params

arps_fitted = exponential_arps(t_fit, qi_arps, D_arps)

last_date = df["TEST_DATE"].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_horizon,
    freq="D"
)

last_t = t[-1]
future_t = np.arange(last_t + 1, last_t + forecast_horizon + 1)
arps_forecast = exponential_arps(future_t, qi_arps, D_arps)

# =====================================================
# 8. Forecast comparison dataframe
# =====================================================
forecast_df = pd.DataFrame({
    "TEST_DATE": future_dates,
    "RF_FORECAST_OIL": rf_forecast,
    "ARPS_EXP_FORECAST_OIL": arps_forecast
})

forecast_df["DIFF_RF_MINUS_ARPS"] = (
    forecast_df["RF_FORECAST_OIL"] - forecast_df["ARPS_EXP_FORECAST_OIL"]
)

# =====================================================
# 9. Print result
# =====================================================
print("Random Forest test performance")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")

print("\nExponential Arps parameters")
print(f"qi : {qi_arps:.2f}")
print(f"D  : {D_arps:.6f} per day")
print(f"D  : {D_arps * 365 * 100:.2f}% nominal annual decline")

print("\nForecast comparison")
print(f"Last actual smooth oil      : {q[-1]:.2f}")
print(f"RF day-1 forecast           : {rf_forecast[0]:.2f}")
print(f"Arps day-1 forecast         : {arps_forecast[0]:.2f}")
print(f"RF day-{forecast_horizon} forecast        : {rf_forecast[-1]:.2f}")
print(f"Arps day-{forecast_horizon} forecast      : {arps_forecast[-1]:.2f}")

rf_decline_pct = (1 - rf_forecast[-1] / q[-1]) * 100
arps_decline_pct = (1 - arps_forecast[-1] / q[-1]) * 100

print(f"RF forecast decline         : {rf_decline_pct:.2f}%")
print(f"Arps forecast decline       : {arps_decline_pct:.2f}%")
print(f"RF - Arps at day-{forecast_horizon}       : {rf_forecast[-1] - arps_forecast[-1]:.2f}")

print("\nFirst 10 forecast rows")
print(forecast_df.head(10))

# =====================================================
# 10. Plot overlay
# =====================================================
test_dates = df["TEST_DATE"].iloc[target_indices[split:]]

plt.figure(figsize=(15, 7))

plt.plot(
    df["TEST_DATE"],
    df["OIL"],
    alpha=0.35,
    label="Actual Oil"
)

plt.plot(
    df["TEST_DATE"],
    df["OIL_SMOOTH"],
    linewidth=2,
    label="Smoothed Oil / Decline Trend"
)

plt.plot(
    df["TEST_DATE"],
    arps_fitted,
    linewidth=2,
    linestyle="-.",
    label="Exponential Arps Fit"
)

plt.plot(
    test_dates,
    test_pred_q,
    linewidth=2,
    label="Random Forest Test Prediction"
)

plt.plot(
    forecast_df["TEST_DATE"],
    forecast_df["RF_FORECAST_OIL"],
    linewidth=3,
    linestyle="--",
    label="Random Forest Recursive Forecast"
)

plt.plot(
    forecast_df["TEST_DATE"],
    forecast_df["ARPS_EXP_FORECAST_OIL"],
    linewidth=3,
    linestyle=":",
    label="Exponential Arps Forecast"
)

plt.axvline(
    df["TEST_DATE"].iloc[target_indices[split]],
    linestyle=":",
    label="Test Start"
)

plt.axvline(
    df["TEST_DATE"].iloc[-1],
    linestyle=":",
    label="Forecast Start"
)

plt.title("Random Forest Decline Forecast vs Exponential Arps")
plt.xlabel("Date")
plt.ylabel("Oil Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = "/home/rian/python_project/myvenv/dca_ml/Images/rf_vs_exponential_arps_overlay.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")


forecast_path = "/home/rian/python_project/myvenv/dca_ml/data/rf_vs_exponential_arps_forecast.csv"
forecast_df.to_csv(forecast_path, index=False)

print(f"\nSaved overlay plot to: {plot_path}")
print(f"Saved forecast comparison to: {forecast_path}")