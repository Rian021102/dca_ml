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


# =====================================================
# 2. Smooth oil rate
# =====================================================
df["OIL_SMOOTH"] = df["OIL"].rolling(window=7, min_periods=1).median()

q = df["OIL_SMOOTH"].values.astype(float)


# =====================================================
# 3. Feature engineering
# =====================================================
def build_features(q_series, idx):
    """
    Build historical features using only data up to idx.
    """
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
# 4. Build ML target: Arps Loss Ratio
# =====================================================
# Arps (1945) loss ratio:
#
#     LR = q / (-dq/dt) = 1/D
#
# Unit: days
#
# We only use decline intervals because LR is meaningful
# when dq/dt is negative.

rows = []
targets_lr = []
target_indices = []

minimum_history = 30
eps = 1e-8

for i in range(minimum_history, len(q) - 1):

    dt_days = (df["TEST_DATE"].iloc[i + 1] - df["TEST_DATE"].iloc[i]).days
    dt_days = max(dt_days, 1)

    dqdt = (q[i + 1] - q[i]) / dt_days

    if dqdt < -eps:
        LR = q[i] / (-dqdt)

        rows.append(build_features(q, i))
        targets_lr.append(LR)
        target_indices.append(i + 1)

X = pd.DataFrame(rows)
y_lr = np.array(targets_lr)
target_indices = np.array(target_indices)

print("Number of usable decline samples:", len(X))


# =====================================================
# 5. Time-series train/test split
# =====================================================
split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y_lr[:split]
y_test = y_lr[split:]


# =====================================================
# 6. Loss-ratio clipping range
# =====================================================
# LR can explode when dq/dt is close to zero.
# Use robust quantile-based clipping from training data.

lr_low = max(np.quantile(y_train, 0.05), 2.0)
lr_high = min(np.quantile(y_train, 0.95), 10000.0)

print(f"LR clipping range: {lr_low:.2f} to {lr_high:.2f} days")


# =====================================================
# 7. Train Random Forest
# =====================================================
rf_lr = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=1
)

rf_lr.fit(X_train, y_train)


# =====================================================
# 8. Test prediction  --  EXACT exponential form
# =====================================================
# Arps ODE in loss-ratio form:
#       dq/dt = -q / LR
# Exact solution over a step dt (with LR held constant):
#       q(t+dt) = q(t) * exp(-dt / LR)
#
# Previously we used the first-order Euler form  q * (1 - dt/LR).
# The exponential form is the canonical Arps b=0 solution and
# is strictly positive for any LR > 0, so no q clipping is needed.

pred_lr_test = rf_lr.predict(X_test)
pred_lr_test = np.clip(pred_lr_test, lr_low, lr_high)

test_current_q = q[target_indices[split:] - 1]
test_actual_q = q[target_indices[split:]]

test_dt = []

for idx in target_indices[split:]:
    dt_days = (df["TEST_DATE"].iloc[idx] - df["TEST_DATE"].iloc[idx - 1]).days
    dt_days = max(dt_days, 1)
    test_dt.append(dt_days)

test_dt = np.array(test_dt)

# Exact exponential step instead of Euler approximation
test_pred_q = test_current_q * np.exp(-test_dt / pred_lr_test)

mae = mean_absolute_error(test_actual_q, test_pred_q)
rmse = mean_squared_error(test_actual_q, test_pred_q) ** 0.5
r2 = r2_score(test_actual_q, test_pred_q)

print("\nRandom Forest using Arps Loss Ratio target (exact exponential step)")
print("Target  : LR = q / (-dq/dt) = 1/D")
print("Stepping: q_next = q * exp(-dt / LR)   [exact Arps b=0 solution]")
print()
print("Test performance")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.3f}")


# =====================================================
# 9. Recursive ML forecast  --  EXACT exponential form
# =====================================================
forecast_horizon = 1000

future_q = list(q.copy())

# Recent terminal LR
recent_lr = []

for i in range(max(1, len(q) - 30), len(q) - 1):

    dt_days = (df["TEST_DATE"].iloc[i + 1] - df["TEST_DATE"].iloc[i]).days
    dt_days = max(dt_days, 1)

    dqdt = (q[i + 1] - q[i]) / dt_days

    if dqdt < -eps:
        recent_lr.append(q[i] / (-dqdt))

if len(recent_lr) > 0:
    terminal_lr = float(np.median(recent_lr))
else:
    terminal_lr = float(np.median(y_train))

terminal_lr = float(np.clip(terminal_lr, lr_low, lr_high))

print("\nTarget summary")
print(f"Average LR in train : {np.mean(y_train):.2f} days")
print(f"Median LR in train  : {np.median(y_train):.2f} days")
print(f"Recent terminal LR  : {terminal_lr:.2f} days")


for step in range(forecast_horizon):

    idx = len(future_q) - 1

    X_future = pd.DataFrame([build_features(future_q, idx)])
    X_future = X_future[X.columns]

    predicted_lr = float(rf_lr.predict(X_future)[0])

    # Keep LR positive and physically reasonable
    predicted_lr = float(np.clip(predicted_lr, lr_low, lr_high))

    # Blend ML prediction with recent terminal LR
    predicted_lr = 0.7 * predicted_lr + 0.3 * terminal_lr

    # Exact analytical Arps exponential step (dt = 1 day implicit)
    # q_next = q * exp(-dt / LR)  --  strictly positive, no clipping needed
    next_q = future_q[-1] * np.exp(-1.0 / predicted_lr)

    future_q.append(next_q)

lr_forecast = np.array(future_q[-forecast_horizon:])


# =====================================================
# 10. Forecast dataframe
# =====================================================
future_dates = pd.date_range(
    start=df["TEST_DATE"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_horizon,
    freq="D"
)

forecast_df = pd.DataFrame({
    "TEST_DATE": future_dates,
    "RF_LOSS_RATIO_FORECAST_OIL": lr_forecast
})


print("\nForecast summary")
print(f"Last actual smooth oil : {q[-1]:.2f}")
print(f"Day-1 forecast         : {lr_forecast[0]:.2f}")
print(f"Day-{forecast_horizon} forecast       : {lr_forecast[-1]:.2f}")
print(f"Forecast decline       : {(1 - lr_forecast[-1] / q[-1]) * 100:.2f}%")

print("\nFirst 10 forecast rows")
print(forecast_df.head(10))


# =====================================================
# 11. Plot result
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
    test_dates,
    test_pred_q,
    linewidth=2,
    label="RF Test Prediction (exact exp step)"
)

plt.plot(
    forecast_df["TEST_DATE"],
    forecast_df["RF_LOSS_RATIO_FORECAST_OIL"],
    linewidth=3,
    linestyle="--",
    label=f"{forecast_horizon}-Day RF Forecast (exact exp step)"
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

plt.title("RF Forecast — Arps Loss Ratio LR with Exact Exponential Integration")
plt.xlabel("Date")
plt.ylabel("Oil Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = "/home/rian/python_project/myvenv/dca_ml/Images/rf_loss_ratio_exact_forecast.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")


# =====================================================
# 12. Save output
# =====================================================
forecast_path = "/home/rian/python_project/myvenv/dca_ml/data/rf_loss_ratio_exact_forecast.csv"
forecast_df.to_csv(forecast_path, index=False)

print("\nSaved files:")
print(plot_path)
print(forecast_path)


# =====================================================
# 13. Feature importance
# =====================================================
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_lr.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 feature importances")
print(feature_importance.head(15))
