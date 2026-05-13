"""
XGBoost one-step-ahead model for daily oil production (well NO 15/9-F-12 H),
extended to a recursive multi-step forecast that continues past the last
observed date.

Setup:
  features at row t   = lagged OIL values + rolling stats computed from rows < t
  target at row t     = OIL(t)
At inference time we feed the model's own predictions back in for the next step.
"""
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- 1. Load ----------
df = pd.read_csv("/home/rian/python_project/myvenv/dca_ml/data/selected_data.csv")
df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
df = df.sort_values("TEST_DATE").reset_index(drop=True)
print(f"Loaded {len(df)} rows, {df['TEST_DATE'].min().date()} -> {df['TEST_DATE'].max().date()}")

# ---------- 2. Feature engineering ----------
LAGS = [1, 2, 3, 5, 7, 14, 21, 30]
ROLL_WINDOWS = [3, 7, 14, 30]

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    s = out["OIL"]
    for lag in LAGS:
        out[f"lag_{lag}"] = s.shift(lag)
    # rolling stats based on PAST values (shift(1) first so row t never sees y_t)
    past = s.shift(1)
    for w in ROLL_WINDOWS:
        out[f"rmean_{w}"] = past.rolling(w).mean()
        out[f"rstd_{w}"]  = past.rolling(w).std()
        out[f"rmin_{w}"]  = past.rolling(w).min()
        out[f"rmax_{w}"]  = past.rolling(w).max()
    out["dayofweek"] = out["TEST_DATE"].dt.dayofweek
    out["month"] = out["TEST_DATE"].dt.month
    out["day_idx"] = np.arange(len(out))
    return out

feat = build_features(df).dropna().reset_index(drop=True)
FEATURE_COLS = [c for c in feat.columns if c not in ("TEST_DATE", "WELL_NAME", "OIL")]
print(f"After dropping warm-up rows: {len(feat)} usable rows, {len(FEATURE_COLS)} features")

X = feat[FEATURE_COLS]
y = feat["OIL"]
dates = feat["TEST_DATE"]

# ---------- 3. Chronological split ----------
TEST = 90  # hold out the last 90 observations for evaluation
X_train, X_test = X.iloc[:-TEST], X.iloc[-TEST:]
y_train, y_test = y.iloc[:-TEST], y.iloc[-TEST:]
d_train, d_test = dates.iloc[:-TEST], dates.iloc[-TEST:]
print(f"Train: {len(X_train)}  Test: {len(X_test)} (last {TEST} obs)")

# ---------- 4. Train XGBoost ----------
model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=40,
    eval_metric="rmse",
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Best iter: {model.best_iteration}, best RMSE: {model.best_score:.2f}")

# ---------- 5. Evaluate ----------
y_pred_train = np.clip(model.predict(X_train), 0, None)
y_pred_test  = np.clip(model.predict(X_test),  0, None)

def metrics(yt, yp):
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    nz = yt > 0
    mape = (np.abs((yt[nz] - yp[nz]) / yt[nz])).mean() * 100 if nz.any() else np.nan
    return mae, rmse, mape

m_tr  = metrics(y_train.values, y_pred_train)
m_te  = metrics(y_test.values,  y_pred_test)
print(f"Train MAE={m_tr[0]:.1f} RMSE={m_tr[1]:.1f} MAPE={m_tr[2]:.2f}%")
print(f"Test  MAE={m_te[0]:.1f} RMSE={m_te[1]:.1f} MAPE={m_te[2]:.2f}%")

# ---------- 6. Recursive multi-step forecast ----------
HORIZON = 90  # forecast this many steps past the last observed date
history = df["OIL"].astype(float).tolist()
last_date = df["TEST_DATE"].iloc[-1]
future_dates, future_preds = [], []

for h in range(1, HORIZON + 1):
    row = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = history[-lag]
    arr = np.array(history, dtype=float)
    for w in ROLL_WINDOWS:
        tail = arr[-w:]
        row[f"rmean_{w}"] = float(tail.mean())
        row[f"rstd_{w}"]  = float(tail.std(ddof=1)) if w > 1 else 0.0
        row[f"rmin_{w}"]  = float(tail.min())
        row[f"rmax_{w}"]  = float(tail.max())
    next_date = last_date + pd.Timedelta(days=h)
    row["dayofweek"] = next_date.dayofweek
    row["month"]     = next_date.month
    row["day_idx"]   = len(df) + h - 1
    X_next = pd.DataFrame([row])[FEATURE_COLS]
    yhat = float(model.predict(X_next)[0])
    yhat = max(0.0, yhat)
    history.append(yhat)
    future_preds.append(yhat)
    future_dates.append(next_date)

# ---------- 6b. Plot actual/train/test/forecast ----------
plt.figure(figsize=(14, 6))
plt.plot(dates, y.values, label="Actual", color="black", linewidth=2)
plt.plot(d_train, y_pred_train, label="Train prediction", color="tab:blue", alpha=0.8)
plt.plot(d_test, y_pred_test, label="Test prediction", color="tab:orange", alpha=0.9)
plt.plot(future_dates, future_preds, label="Forecast", color="tab:green", linewidth=2)
plt.axvline(d_test.iloc[0], color="gray", linestyle="--", linewidth=1, label="Train/Test split")
plt.axvline(future_dates[0], color="tab:green", linestyle=":", linewidth=1.5, label="Forecast start")
plt.title("Oil Production: Actual vs Train/Test Predictions with Forecast")
plt.xlabel("Date")
plt.ylabel("OIL")
plt.legend()
plt.tight_layout()

plot_path = "/home/rian/python_project/myvenv/dca_ml/Images/actual_train_test_prediction.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved prediction plot to: {plot_path}")

# ---------- 7. Persist outputs ----------
out = {
    "actuals":   [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(df["TEST_DATE"], df["OIL"])],
    "test_pred": [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(d_test, y_pred_test)],
    "forecast":  [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in zip(future_dates, future_preds)],
    "metrics": {
        "train": {"mae": round(m_tr[0], 2), "rmse": round(m_tr[1], 2), "mape_pct": round(m_tr[2], 2)},
        "test":  {"mae": round(m_te[0], 2), "rmse": round(m_te[1], 2), "mape_pct": round(m_te[2], 2)},
        "best_iter": int(model.best_iteration),
    },
    "feature_importance": sorted(
        [{"feature": f, "gain": round(float(g), 4)}
         for f, g in zip(FEATURE_COLS, model.feature_importances_)],
        key=lambda r: -r["gain"],
    )[:12],
}
with open("/home/rian/python_project/myvenv/dca_ml/data/forecast_output.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))

outputs_dir = "/home/rian/python_project/myvenv/dca_ml/data/outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Save forecast as CSV too
pd.DataFrame({"DATE": future_dates, "OIL_FORECAST": future_preds}).to_csv(
    os.path.join(outputs_dir, "oil_forecast_90day.csv"), index=False
)

# Copy training script to outputs for the user
import shutil
shutil.copy(
    "/home/rian/python_project/myvenv/dca_ml/src/Trial/train_forecast.py",
    os.path.join(outputs_dir, "train_forecast.py"),
)

print("Top features by gain:")
for r in out["feature_importance"][:8]:
    print(f"  {r['feature']:12s}  {r['gain']:.4f}")
print(f"Forecast horizon: {HORIZON} days, ending {future_dates[-1].date()}")
print(f"Forecast endpoints: first={future_preds[0]:.1f}, last={future_preds[-1]:.1f}")
