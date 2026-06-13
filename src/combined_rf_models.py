"""
combined_rf_models.py
Exposes three RF-based DCA forecast functions used by test.py:

  run_dcalike_model         – decline-ratio target  (dcamodel.py pipeline)
  run_rf_decline_rate_model – log-decline target     (newmodel.py pipeline)
  run_rf_loss_ratio_model   – rate-ratio target      (implemented here)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

import dcamodel as _dcamodel
import newmodel as _newmodel
from dcamodel import build_features          # re-exported for test.py


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pad_to(arr, horizon, fill=0.0):
    """Truncate or zero-pad *arr* to exactly *horizon* elements."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= horizon:
        return arr[:horizon]
    return np.concatenate([arr, np.full(horizon - len(arr), fill)])


# ---------------------------------------------------------------------------
# Loss-ratio dataset + model (not in dcamodel / newmodel)
# ---------------------------------------------------------------------------

def _create_loss_ratio_dataset(df, minimum_history=30):
    """Target: r_t = q_{t+1} / q_t  (multiplicative rate ratio, < 1 = decline)."""
    work = df.copy()
    work["OIL_NEXT"] = work["OIL_SMOOTH"].shift(-1)
    work["NEXT_TEST_DATE"] = work["TEST_DATE"].shift(-1)
    work["DELTA_T"] = (
        (work["NEXT_TEST_DATE"] - work["TEST_DATE"]).dt.total_seconds() / 86400.0
    )
    work.loc[work["DELTA_T"] <= 0, "DELTA_T"] = np.nan
    work["LOSS_RATIO"] = work["OIL_NEXT"] / work["OIL_SMOOTH"]
    work = work.dropna(
        subset=["LOSS_RATIO", "OIL_NEXT", "NEXT_TEST_DATE", "DELTA_T"]
    ).reset_index(drop=True)

    q = work["OIL_SMOOTH"].to_numpy(dtype=float)
    rows, targets, indices = [], [], []
    for i in range(minimum_history, len(work)):
        rows.append(build_features(q, i))
        targets.append(work["LOSS_RATIO"].iloc[i])
        indices.append(i)

    return (
        pd.DataFrame(rows),
        np.asarray(targets, dtype=float),
        np.asarray(indices, dtype=int),
        work,
    )


def _train_rf(X_train, y_train):
    if len(X_train) >= 20:
        tscv = TimeSeriesSplit(n_splits=min(5, len(X_train) - 1))
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=1),
            param_distributions={
                "n_estimators": [120, 200, 300],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.8],
            },
            n_iter=5,
            scoring="neg_mean_squared_error",
            cv=tscv,
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def _loss_ratio_recursive_forecast(
    model, model_df, feature_columns, stop_oil=0.1, max_steps=1000
):
    q_history = model_df["OIL_SMOOTH"].to_numpy(dtype=float).tolist()
    current_date = model_df["TEST_DATE"].iloc[-1]
    freq = pd.infer_freq(model_df["TEST_DATE"]) or "D"
    date_offset = pd.tseries.frequencies.to_offset(freq)

    forecasts = []
    for _ in range(max_steps):
        idx = len(q_history) - 1
        X_f = pd.DataFrame([build_features(q_history, idx)])[feature_columns]
        ratio = float(model.predict(X_f)[0])
        # Decline-only, clipped consistent with newmodel.py (-0.05 <= log_D <= -1e-5)
        ratio = float(np.clip(ratio, np.exp(-0.05), np.exp(-1e-5)))
        q_next = q_history[-1] * ratio
        forecasts.append(q_next)
        q_history.append(q_next)
        current_date = current_date + date_offset
        if q_next <= stop_oil:
            break

    return np.asarray(forecasts, dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dcalike_model(df, q, forecast_horizon=1000):
    """DCA-like RF model – decline-ratio target (dcamodel.py pipeline)."""
    result = _dcamodel.run_dca_pipeline(
        df, stop_oil=0.1, max_forecast_steps=forecast_horizon
    )
    fc = result["forecast_df"]["FORECAST_OIL"].values
    return _pad_to(fc, forecast_horizon)


def run_rf_decline_rate_model(df, q, forecast_horizon=1000):
    """RF decline-rate model – log-decline target (newmodel.py pipeline)."""
    result = _newmodel.run_dca_pipeline(
        df, stop_oil=0.1, max_forecast_steps=forecast_horizon
    )
    fc = result["forecast_df"]["FORECAST_OIL"].values
    return _pad_to(fc, forecast_horizon)


def run_rf_loss_ratio_model(df, q, forecast_horizon=1000):
    """RF loss-ratio model – multiplicative rate-ratio target."""
    X, y, indices, model_df = _create_loss_ratio_dataset(df, minimum_history=30)
    if len(X) == 0:
        return np.zeros(forecast_horizon)

    train_size = int(len(model_df) * 0.8)
    train_mask = indices < train_size
    X_train, y_train = X.loc[train_mask], y[train_mask]

    model = _train_rf(X_train, y_train)
    fc = _loss_ratio_recursive_forecast(
        model, model_df, X.columns, stop_oil=0.1, max_steps=forecast_horizon
    )
    return _pad_to(fc, forecast_horizon)
