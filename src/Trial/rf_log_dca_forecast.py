import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. Load and clean data
# =========================
DATA_PATH = "/home/rian/python_project/myvenv/dca_ml/data/selected_data_02.csv"   # change this path if needed
DATE_COL = "TEST_DATE"
RATE_COL = "OIL"
FORECAST_DAYS = 1000

df = pd.read_csv(DATA_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Keep only positive oil rates because log(q) cannot handle zero/negative values
df = df[df[RATE_COL].notna() & (df[RATE_COL] > 0)].copy()

# If there are duplicate dates, average them
daily = df.groupby(DATE_COL, as_index=False).agg({RATE_COL: "mean"})

# Create a complete daily date index.
# Missing days are interpolated so lag/rolling features remain continuous.
daily = daily.set_index(DATE_COL).asfreq("D")
daily[RATE_COL] = daily[RATE_COL].interpolate(method="time").ffill().bfill()
daily = daily.reset_index()

# Time index and log transform
daily["t"] = np.arange(len(daily))
daily["log_oil"] = np.log(daily[RATE_COL])


# =========================
# 2. Feature engineering
# =========================
def make_features(data, lags=(1, 2, 3, 7, 14), rolls=(3, 7, 14, 30)):
    """
    Creates DCA-like ML features from log(q):
    - lagged log(q)
    - rolling average of log(q)
    - rolling volatility
    - short-term decline/slope features
    - calendar features
    """
    out = data.copy()

    for lag in lags:
        out[f"lag_{lag}"] = out["log_oil"].shift(lag)

    for w in rolls:
        out[f"roll_mean_{w}"] = out["log_oil"].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out["log_oil"].shift(1).rolling(w).std()

    # Decline/shape indicators in log-rate space
    out["slope_3"] = out["log_oil"].shift(1) - out["log_oil"].shift(4)
    out["slope_7"] = out["log_oil"].shift(1) - out["log_oil"].shift(8)
    out["log_decline_ratio_1"] = out["log_oil"].shift(1) - out["log_oil"].shift(2)

    # Simple time/calendar features
    out["day"] = out[DATE_COL].dt.day
    out["month"] = out[DATE_COL].dt.month

    return out


model_df = make_features(daily).dropna().reset_index(drop=True)

feature_cols = [c for c in model_df.columns if c not in [DATE_COL, RATE_COL, "log_oil"]]
target_col = "log_oil"


# =========================
# 3. Train/test split
# =========================
split_idx = int(len(model_df) * 0.8)

train = model_df.iloc[:split_idx].copy()
test = model_df.iloc[split_idx:].copy()

X_train = train[feature_cols]
y_train = train[target_col]

X_test = test[feature_cols]
y_test = test[target_col]


# =========================
# 4. Train Random Forest on log(q)
# =========================
rf = RandomForestRegressor(
    n_estimators=700,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Predict log(q), then convert back to actual oil rate using exp()
train_pred_log = rf.predict(X_train)
test_pred_log = rf.predict(X_test)

train_pred = np.exp(train_pred_log)
test_pred = np.exp(test_pred_log)


# =========================
# 5. Evaluation
# =========================
print("Model performance on original oil-rate scale")
print("-" * 50)
print(f"Train MAE : {mean_absolute_error(train[RATE_COL], train_pred):.2f}")
print(f"Test MAE  : {mean_absolute_error(test[RATE_COL], test_pred):.2f}")
print(f"Train RMSE: {mean_squared_error(train[RATE_COL], train_pred) ** 0.5:.2f}")
print(f"Test RMSE : {mean_squared_error(test[RATE_COL], test_pred) ** 0.5:.2f}")
print(f"Train R2  : {r2_score(train[RATE_COL], train_pred):.3f}")
print(f"Test R2   : {r2_score(test[RATE_COL], test_pred):.3f}")

importance = (
    pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    })
    .sort_values("importance", ascending=False)
)

print("\nFeature importance")
print("-" * 50)
print(importance)


# =========================
# 6. Recursive future forecast
# =========================
def build_feature_row(history, next_date, feature_cols, lags=(1, 2, 3, 7, 14), rolls=(3, 7, 14, 30)):
    """
    Build one feature row for the next day using only historical/predicted log(q).
    This is what allows recursive forecasting:
    predicted day 1 becomes input for predicted day 2, and so on.
    """
    row = {}

    row["t"] = len(history)

    for lag in lags:
        row[f"lag_{lag}"] = history["log_oil"].iloc[-lag]

    for w in rolls:
        values = history["log_oil"].iloc[-w:]
        row[f"roll_mean_{w}"] = values.mean()
        row[f"roll_std_{w}"] = values.std()

    row["slope_3"] = history["log_oil"].iloc[-1] - history["log_oil"].iloc[-4]
    row["slope_7"] = history["log_oil"].iloc[-1] - history["log_oil"].iloc[-8]
    row["log_decline_ratio_1"] = history["log_oil"].iloc[-1] - history["log_oil"].iloc[-2]

    row["day"] = next_date.day
    row["month"] = next_date.month

    return pd.DataFrame([row])[feature_cols]


# Refit model using all available history before forecasting future
rf_full = RandomForestRegressor(
    n_estimators=700,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_full.fit(model_df[feature_cols], model_df[target_col])

history = daily[[DATE_COL, RATE_COL, "t", "log_oil"]].copy()
future_rows = []

for step in range(FORECAST_DAYS):
    next_date = history[DATE_COL].iloc[-1] + pd.Timedelta(days=1)

    X_next = build_feature_row(history, next_date, feature_cols)
    pred_log = rf_full.predict(X_next)[0]
    pred_oil = np.exp(pred_log)

    future_rows.append({
        DATE_COL: next_date,
        "OIL_FORECAST": pred_oil,
        "log_oil_forecast": pred_log
    })

    # Append prediction so the next forecast step uses it as history
    history = pd.concat([
        history,
        pd.DataFrame([{
            DATE_COL: next_date,
            RATE_COL: pred_oil,
            "t": len(history),
            "log_oil": pred_log
        }])
    ], ignore_index=True)

future = pd.DataFrame(future_rows)


# =========================
# 7. Plot result
# =========================
plt.figure(figsize=(14, 7))

plt.plot(daily[DATE_COL], daily[RATE_COL], label="Actual Oil", linewidth=1.6)
plt.plot(train[DATE_COL], train_pred, label="Train Prediction", linestyle="--", linewidth=1.2)
plt.plot(test[DATE_COL], test_pred, label="Test Prediction", linestyle="--", linewidth=1.2)
plt.plot(future[DATE_COL], future["OIL_FORECAST"], label=f"{FORECAST_DAYS}-Day RF Forecast", linewidth=2)

plt.axvline(test[DATE_COL].iloc[0], linestyle=":", label="Train/Test Split")
plt.axvline(future[DATE_COL].iloc[0], linestyle=":", label="Forecast Start")

plt.title("Random Forest DCA-style Forecast with log(q) Transform")
plt.xlabel("Date")
plt.ylabel("Oil Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/rian/python_project/myvenv/dca_ml/Images/rf_forecast_plot.png", dpi=300, bbox_inches="tight")
print("Plot saved to /home/rian/python_project/myvenv/dca_ml/Images/rf_forecast_plot.png")