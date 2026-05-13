import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# =========================
# 1. Load data
# =========================
path = Path("/home/rian/python_project/myvenv/dca_ml/data/selected_data.csv")
df = pd.read_csv(path)

df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
df = df.sort_values("TEST_DATE").reset_index(drop=True)

# Keep valid oil data
df = df.dropna(subset=["OIL"]).copy()
df = df[df["OIL"] >= 0].copy()

# =========================
# 2. Create temporal supervised data
#    Previous oil rates -> next oil rate
# =========================
def create_lagged_features(data, target_col="OIL", n_lags=5):
    df_lag = data.copy()
    
    for lag in range(1, n_lags + 1):
        df_lag[f"lag_{lag}"] = df_lag[target_col].shift(lag)
    
    df_lag["target_next"] = df_lag[target_col]
    df_lag = df_lag.dropna().reset_index(drop=True)
    
    feature_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
    X = df_lag[feature_cols]
    y = df_lag["target_next"]
    dates = df_lag["TEST_DATE"]
    
    return X, y, dates, df_lag, feature_cols

n_lags = 5
X, y, dates, df_lag, feature_cols = create_lagged_features(df, n_lags=n_lags)

# =========================
# 3. Time-based train/test split
# =========================
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]
dates_train = dates.iloc[:split_idx]
dates_test = dates.iloc[split_idx:]

# =========================
# 4. Train lightweight XGBoost
# =========================
model = XGBRegressor(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=2,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=1,
    random_state=42
)

model.fit(X_train, y_train)

# Predict train/test
y_pred_train = np.maximum(model.predict(X_train), 0)
y_pred_test = np.maximum(model.predict(X_test), 0)

# =========================
# 5. Recursive continuation forecast
# =========================
forecast_horizon = 100

last_date = df["TEST_DATE"].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_horizon,
    freq="D"
)

history = list(df["OIL"].iloc[-n_lags:].values)
future_forecast = []

for _ in range(forecast_horizon):
    # Because feature columns are lag_1, lag_2, ..., lag_5
    # lag_1 = most recent value
    input_features = pd.DataFrame(
        [history[-n_lags:][::-1]],
        columns=feature_cols
    )
    
    pred = model.predict(input_features)[0]
    pred = max(pred, 0)
    
    future_forecast.append(pred)
    history.append(pred)

forecast_df = pd.DataFrame({
    "TEST_DATE": future_dates,
    "Forecast_OIL": future_forecast
})

# =========================
# 6. Evaluation
# =========================
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2]
})

print("XGBoost temporal forecasting")
print("--------------------------------")
print(f"Lag features used: {n_lags}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print("\nTest metrics:")
print(metrics_df.to_string(index=False))

print("\nForecast preview:")
print(forecast_df.head(10).to_string(index=False))

# =========================
# 7. Plot
# =========================
plt.figure(figsize=(14, 7))

plt.plot(df["TEST_DATE"], df["OIL"], label="Actual Oil Rate")
plt.plot(dates_train, y_pred_train, label="Train Prediction")
plt.plot(dates_test, y_pred_test, label="Test Prediction")
plt.plot(forecast_df["TEST_DATE"], forecast_df["Forecast_OIL"], label="100-Day Forecast Continuation")

plt.xlabel("Date")
plt.ylabel("Oil Rate")
plt.title("XGBoost: Temporal Data Predicting Next Oil Rate")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plot_path = "/home/rian/python_project/myvenv/dca_ml/Images/xgboost_temporal_forecast.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved plot image to: {plot_path}")

