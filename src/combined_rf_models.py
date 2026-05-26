import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit

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

def run_dcalike_model(df, q):
    """
    Implementation of the dcalike.py logic:
    Predicts log(q_{i+1}/q_i) using Random Forest.
    """
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
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Forecast
    forecast_horizon = 1000
    future_q_rf = list(q.copy())
    
    recent_log_declines = np.diff(np.log(np.clip(q[-30:], 1e-6, None)))
    negative_recent_declines = recent_log_declines[recent_log_declines < 0]
    terminal_decline = np.median(negative_recent_declines) if len(negative_recent_declines) > 0 else np.median(recent_log_declines)

    for step in range(forecast_horizon):
        idx = len(future_q_rf) - 1
        X_future = pd.DataFrame([build_features(future_q_rf, idx)])[X.columns]
        predicted_log_decline = float(rf.predict(X_future)[0])
        predicted_log_decline = min(predicted_log_decline, 0.0)
        if terminal_decline < 0:
            predicted_log_decline = 0.7 * predicted_log_decline + 0.3 * terminal_decline
        predicted_log_decline = np.clip(predicted_log_decline, -0.05, -1e-5)
        future_q_rf.append(future_q_rf[-1] * np.exp(predicted_log_decline))

    return np.array(future_q_rf[-forecast_horizon:])

def run_rf_decline_rate_model(df, q):
    """
    Implementation of rf_decline_rate_exact.py logic:
    Predicts nominal decline rate D = -(dq/dt)/q and uses exact exponential stepping.
    """
    rows = []
    targets_D = []
    target_indices = []
    minimum_history = 30

    for i in range(minimum_history, len(q) - 1):
        rows.append(build_features(q, i))
        dt_days = max((df["TEST_DATE"].iloc[i + 1] - df["TEST_DATE"].iloc[i]).days, 1)
        dqdt = (q[i + 1] - q[i]) / dt_days
        targets_D.append(-dqdt / q[i])
        target_indices.append(i + 1)

    X = pd.DataFrame(rows)
    y_D = np.array(targets_D)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_D[:split], y_D[split:]

    rf_D = RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=1)
    rf_D.fit(X_train, y_train)

    # Forecast
    forecast_horizon = 1000
    future_q = list(q.copy())
    
    recent_D = []
    for i in range(max(1, len(q) - 30), len(q) - 1):
        dt_days = max((df["TEST_DATE"].iloc[i + 1] - df["TEST_DATE"].iloc[i]).days, 1)
        recent_D.append(-(q[i+1] - q[i]) / (dt_days * q[i]))
    
    recent_D = np.array(recent_D)
    positive_recent_D = recent_D[recent_D > 0]
    terminal_D = np.median(positive_recent_D) if len(positive_recent_D) > 0 else max(np.median(recent_D), 1e-5)

    for step in range(forecast_horizon):
        idx = len(future_q) - 1
        X_future = pd.DataFrame([build_features(future_q, idx)])[X.columns]
        predicted_D = max(float(rf_D.predict(X_future)[0]), 0.0)
        predicted_D = 0.7 * predicted_D + 0.3 * terminal_D
        predicted_D = np.clip(predicted_D, 1e-5, 0.05)
        future_q.append(future_q[-1] * np.exp(-predicted_D))

    return np.array(future_q[-forecast_horizon:])

def run_rf_loss_ratio_model(df, q):
    """
    Implementation of rf_loss_ratio_exact.py logic:
    Predicts Loss Ratio LR = q / (-dq/dt) and uses exact exponential stepping.
    """
    rows = []
    targets_lr = []
    target_indices = []
    minimum_history = 30
    eps = 1e-8

    for i in range(minimum_history, len(q) - 1):
        dt_days = max((df["TEST_DATE"].iloc[i + 1] - df["TEST_DATE"].iloc[i]).days, 1)
        dqdt = (q[i + 1] - q[i]) / dt_days
        if dqdt < -eps:
            rows.append(build_features(q, i))
            targets_lr.append(q[i] / (-dqdt))
            target_indices.append(i + 1)

    X = pd.DataFrame(rows)
    y_lr = np.array(targets_lr)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_lr[:split], y_lr[split:]

    lr_low = max(np.quantile(y_train, 0.05), 2.0)
    lr_high = min(np.quantile(y_train, 0.95), 10000.0)

    rf_lr = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=1)
    rf_lr.fit(X_train, y_train)

    # Forecast (Simplified recursive implementation based on the provided logic)
    forecast_horizon = 1000
    future_q = list(q.copy())
    
    for step in range(forecast_horizon):
        idx = len(future_q) - 1
        X_future = pd.DataFrame([build_features(future_q, idx)])
        # Ensure columns match training X
        X_future = X_future[X.columns]
        pred_lr = float(rf_lr.predict(X_future)[0])
        pred_lr = np.clip(pred_lr, lr_low, lr_high)
        # q_next = q * exp(-dt / LR), dt = 1 day
        future_q.append(future_q[-1] * np.exp(-1.0 / pred_lr))

    return np.array(future_q[-forecast_horizon:])
