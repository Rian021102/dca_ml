import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# 1. Load data
# =====================================================
path = "/home/rian/python_project/myvenv/dca_ml/data/test_data.csv"

df = pd.read_csv(path)
df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
df = df.sort_values("TEST_DATE").dropna(subset=["OIL"]).reset_index(drop=True)
df = df[df["OIL"] > 0].reset_index(drop=True)

df["OIL_SMOOTH"] = df["OIL"].rolling(window=7, min_periods=1).median()
q = df["OIL_SMOOTH"].values.astype(float)

# =====================================================
# 2. Feature engineering
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
# 3. Build target: decline rate  D = -(dq/dt) / q
# =====================================================
rows = []
targets_D = []
target_indices = []

minimum_history = 30

for i in range(minimum_history, len(q) - 1):
    rows.append(build_features(q, i))

    dt_days = (df["TEST_DATE"].iloc[i + 1] - df["TEST_DATE"].iloc[i]).days
    dt_days = max(dt_days, 1)

    # Arps decline rate / nominal decline:
    # D = -(dq/dt) / q
    dqdt = (q[i + 1] - q[i]) / dt_days
    D = -dqdt / q[i]

    targets_D.append(D)
    target_indices.append(i + 1)

X = pd.DataFrame(rows)
y_D = np.array(targets_D)
target_indices = np.array(target_indices)

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y_D[:split]
y_test = y_D[split:]

# =====================================================
# 4. Train Random Forest
# =====================================================
rf_D = RandomForestRegressor(
    n_estimators=120,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=1
)

rf_D.fit(X_train, y_train)

# =====================================================
# 5. Test prediction  --  EXACT exponential form
# =====================================================
# The underlying Arps ODE is:
#       dq/dt = -D * q
# whose exact solution over a step dt (with D held constant) is:
#       q(t+dt) = q(t) * exp(-D * dt)
#
# Previously we used the first-order Euler form  q * (1 - D*dt).
# The exponential form is the canonical Arps b=0 solution and
# is strictly positive for any D, so no q clipping is needed.

pred_D_test = rf_D.predict(X_test)

test_current_q = q[target_indices[split:] - 1]
test_actual_q = q[target_indices[split:]]

test_dt = []
for idx in target_indices[split:]:
    dt_days = (df["TEST_DATE"].iloc[idx] - df["TEST_DATE"].iloc[idx - 1]).days
    test_dt.append(max(dt_days, 1))

test_dt = np.array(test_dt)

# Exact exponential step instead of Euler approximation
test_pred_q = test_current_q * np.exp(-pred_D_test * test_dt)

mae = mean_absolute_error(test_actual_q, test_pred_q)
rmse = mean_squared_error(test_actual_q, test_pred_q) ** 0.5
r2 = r2_score(test_actual_q, test_pred_q)

# =====================================================
# 6. Recursive forecast  --  EXACT exponential form
# =====================================================
forecast_horizon = 1000
future_q = list(q.copy())

recent_D = []

for i in range(max(1, len(q) - 30), len(q) - 1):
    dt_days = (df["TEST_DATE"].iloc[i + 1] - df["TEST_DATE"].iloc[i]).days
    dt_days = max(dt_days, 1)

    dqdt = (q[i + 1] - q[i]) / dt_days
    D = -dqdt / q[i]
    recent_D.append(D)

recent_D = np.array(recent_D)
positive_recent_D = recent_D[recent_D > 0]

if len(positive_recent_D) > 0:
    terminal_D = np.median(positive_recent_D)
else:
    terminal_D = max(np.median(recent_D), 1e-5)

for step in range(forecast_horizon):
    idx = len(future_q) - 1

    X_future = pd.DataFrame([build_features(future_q, idx)])
    X_future = X_future[X.columns]

    predicted_D = float(rf_D.predict(X_future)[0])

    # Force decline-only: positive D
    predicted_D = max(predicted_D, 0.0)

    # Blend ML prediction with recent terminal D
    predicted_D = 0.7 * predicted_D + 0.3 * terminal_D

    # Physical bounds on D (per day)
    predicted_D = np.clip(predicted_D, 1e-5, 0.05)

    # Exact analytical Arps exponential step (dt = 1 day implicit)
    # q_next = q * exp(-D * dt)   --  strictly positive, no clipping needed
    next_q = future_q[-1] * np.exp(-predicted_D)

    future_q.append(next_q)

D_forecast = np.array(future_q[-forecast_horizon:])

future_dates = pd.date_range(
    start=df["TEST_DATE"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_horizon,
    freq="D"
)

forecast_df = pd.DataFrame({
    "TEST_DATE": future_dates,
    "RF_DECLINE_RATE_FORECAST_OIL": D_forecast
})

# =====================================================
# 7. Print result
# =====================================================
print("Random Forest using Arps decline rate target (exact exponential step)")
print("Target  : D = -(dq/dt) / q")
print("Stepping: q_next = q * exp(-D * dt)   [exact Arps b=0 solution]")
print()
print("Test performance")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")

print("\nTarget summary")
print(f"Average D in train : {np.mean(y_train):.6f} per day")
print(f"Median D in train  : {np.median(y_train):.6f} per day")
print(f"Recent terminal D  : {terminal_D:.6f} per day")

print("\nForecast summary")
print(f"Last actual smooth oil : {q[-1]:.2f}")
print(f"Day-1 forecast         : {D_forecast[0]:.2f}")
print(f"Day-{forecast_horizon} forecast       : {D_forecast[-1]:.2f}")
print(f"Forecast decline       : {(1 - D_forecast[-1] / q[-1]) * 100:.2f}%")

print("\nFirst 10 forecast rows")
print(forecast_df.head(10))

# =====================================================
# 8. Plot
# =====================================================
test_dates = df["TEST_DATE"].iloc[target_indices[split:]]

plt.figure(figsize=(15, 7))

plt.plot(df["TEST_DATE"], df["OIL"], alpha=0.35, label="Actual Oil")
plt.plot(df["TEST_DATE"], df["OIL_SMOOTH"], linewidth=2, label="Smoothed Oil / Decline Trend")
plt.plot(test_dates, test_pred_q, linewidth=2, label="RF Test Prediction (exact exp step)")
plt.plot(
    forecast_df["TEST_DATE"],
    forecast_df["RF_DECLINE_RATE_FORECAST_OIL"],
    linewidth=3,
    linestyle="--",
    label=f"{forecast_horizon}-Day RF Forecast (exact exp step)"
)

plt.axvline(df["TEST_DATE"].iloc[target_indices[split]], linestyle=":", label="Test Start")
plt.axvline(df["TEST_DATE"].iloc[-1], linestyle=":", label="Forecast Start")

plt.title("RF Forecast — Arps Decline Rate D with Exact Exponential Integration")
plt.xlabel("Date")
plt.ylabel("Oil Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = "/home/rian/python_project/myvenv/dca_ml/Images/rf_decline_rate_exact_forecast.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")

forecast_path = "/home/rian/python_project/myvenv/dca_ml/data/rf_decline_rate_exact_forecast.csv"
forecast_df.to_csv(forecast_path, index=False)

print(f"\nSaved plot to: {plot_path}")
print(f"Saved forecast to: {forecast_path}")
