import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def build_features(q_series, idx):
    """Build DCA-like ML features from oil-rate history up to index idx only.

    This follows the dcalike.py feature idea: current rate, log current rate,
    rolling statistics, log-slope, and oil-rate ratio statistics.
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

def create_dataset(df, minimum_history=30):
    """Create supervised ML data using the same target convention as dcalike.py.

    Target:
        log_decline_t = ln(q_{t+1} / q_t) / delta_t

    For daily data this is equivalent to the original dcalike.py target:
        ln(q_{t+1} / q_t)

    During recursive forecasting, the prediction is constrained to be <= 0,
    then converted back to oil rate using:
        q_{t+1} = q_t * exp(predicted_log_decline * delta_t)

    Features are built only from historical smoothed oil up to time t.
    """
    work = df.copy()
    work["OIL_NEXT"] = work["OIL_SMOOTH"].shift(-1)
    work["NEXT_TEST_DATE"] = work["TEST_DATE"].shift(-1)
    work["DELTA_T"] = (work["NEXT_TEST_DATE"] - work["TEST_DATE"]).dt.total_seconds() / 86400.0
    work.loc[work["DELTA_T"] <= 0, "DELTA_T"] = np.nan

    # dcalike.py target convention: negative values mean decline, positive values mean increase.
    work["LOG_DECLINE"] = np.log(work["OIL_NEXT"] / work["OIL_SMOOTH"]) / work["DELTA_T"]
    work = work.dropna(subset=["LOG_DECLINE", "OIL_NEXT", "NEXT_TEST_DATE", "DELTA_T"]).reset_index(drop=True)

    q = work["OIL_SMOOTH"].to_numpy(dtype=float)

    rows, targets, target_indices = [], [], []
    for i in range(minimum_history, len(work)):
        rows.append(build_features(q, i))
        targets.append(work["LOG_DECLINE"].iloc[i])
        target_indices.append(i)

    X = pd.DataFrame(rows)
    y = np.asarray(targets, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)

    return X, y, target_indices, work

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_current_figure(path, dpi=200):
    _ensure_parent_dir(path)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def split_train_test_by_time(X_all, y_all, target_indices, total_rows, train_fraction=0.8):
    train_size = int(total_rows * train_fraction)
    train_mask = target_indices < train_size
    test_mask = ~train_mask

    return {
        "train_size": train_size,
        "X_train": X_all.loc[train_mask],
        "y_train": y_all[train_mask],
        "X_test": X_all.loc[test_mask],
        "y_test": y_all[test_mask],
        "test_target_indices": target_indices[test_mask],
    }

def train_decline_model(X_train, y_train):
    if len(X_train) >= 20:
        n_splits = min(5, len(X_train) - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        base_model = RandomForestRegressor(random_state=42, n_jobs=1)
        param_distributions = {
            "n_estimators": [120, 200, 300],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.8],
        }

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=5,
            scoring="neg_mean_squared_error",
            cv=tscv,
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        return {
            "model": search.best_estimator_,
            "best_cv_rmse": np.sqrt(-search.best_score_),
            "best_params": search.best_params_,
            "used_cv_search": True,
        }

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "best_cv_rmse": None,
        "best_params": None,
        "used_cv_search": False,
    }


def evaluate_oil_predictions(actual_oil, predicted_oil):
    mse = mean_squared_error(actual_oil, predicted_oil)
    return {
        "mae": mean_absolute_error(actual_oil, predicted_oil),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(actual_oil, predicted_oil),
    }


def _tree_estimators(model):
    """Return tree estimators from a plain RF or a fitted search wrapper."""
    return getattr(model, "estimators_", [])


def recursive_forecast(
    model,
    model_df,
    feature_columns,
    stop_oil=1.0,
    max_forecast_steps=100000,
    confidence_percentiles=(5, 95),
):
    """dcalike.py-style recursive forecast, stopped by an oil-rate threshold.

    This is the same core logic as dcalike.py:
    1. Build features from the current recursive oil history.
    2. Predict log decline.
    3. Force decline-only behavior with min(prediction, 0.0).
    4. Blend with the recent terminal decline.
    5. Clip the decline to avoid flat forecasts or unrealistic collapse.
    6. Convert predicted log decline back to oil rate with exp().

    Difference from dcalike.py:
    - dcalike.py uses a fixed forecast_horizon.
    - This function keeps forecasting until FORECAST_OIL <= stop_oil,
      with max_forecast_steps only used as a safety cap.
    """
    future_oil_predictions = []
    future_dates = []
    future_log_declines = []
    future_oil_lower = []
    future_oil_upper = []

    q_history = model_df["OIL_SMOOTH"].to_numpy(dtype=float).tolist()
    current_date = model_df["TEST_DATE"].iloc[-1]

    recent_log_declines = np.diff(np.log(np.clip(q_history[-30:], 1e-6, None)))
    negative_recent_declines = recent_log_declines[recent_log_declines < 0]

    if len(negative_recent_declines) > 0:
        terminal_decline = float(np.median(negative_recent_declines))
    elif len(recent_log_declines) > 0:
        terminal_decline = float(np.median(recent_log_declines))
    else:
        terminal_decline = -1e-5

    freq = pd.infer_freq(model_df["TEST_DATE"]) or "D"
    date_offset = pd.tseries.frequencies.to_offset(freq)

    # Stop immediately if the last historical rate is already at or below threshold.
    if q_history[-1] <= stop_oil:
        return pd.DataFrame(
            columns=[
                "TEST_DATE",
                "PRED_LOG_DECLINE",
                "FORECAST_OIL",
                "FORECAST_OIL_LOWER",
                "FORECAST_OIL_UPPER",
            ]
        )

    for _ in range(max_forecast_steps):
        idx = len(q_history) - 1

        X_future = pd.DataFrame([build_features(q_history, idx)])
        X_future = X_future[feature_columns]

        predicted_log_decline = float(model.predict(X_future)[0])

        # Decline-only constraint copied from dcalike.py.
        predicted_log_decline = min(predicted_log_decline, 0.0)

        # Blend ML decline with recent terminal decline copied from dcalike.py.
        if terminal_decline < 0:
            predicted_log_decline = 0.7 * predicted_log_decline + 0.3 * terminal_decline

        # Avoid unrealistic collapse or flat forecast copied from dcalike.py.
        predicted_log_decline = float(np.clip(predicted_log_decline, -0.05, -1e-5))

        next_date = current_date + date_offset
        step_t = (next_date - current_date) / np.timedelta64(1, "D")
        if step_t <= 0:
            step_t = 1.0

        next_q = q_history[-1] * np.exp(predicted_log_decline * step_t)

        estimators = _tree_estimators(model)
        if estimators:
            tree_pred = np.array([est.predict(X_future.to_numpy())[0] for est in estimators], dtype=float)
            tree_pred = np.minimum(tree_pred, 0.0)
            if terminal_decline < 0:
                tree_pred = 0.7 * tree_pred + 0.3 * terminal_decline
            tree_pred = np.clip(tree_pred, -0.05, -1e-5)
            q_next_dist = q_history[-1] * np.exp(tree_pred * step_t)
            lo = float(np.percentile(q_next_dist, confidence_percentiles[0]))
            hi = float(np.percentile(q_next_dist, confidence_percentiles[1]))
        else:
            lo = hi = float(next_q)

        future_log_declines.append(predicted_log_decline)
        future_oil_predictions.append(float(next_q))
        future_oil_lower.append(float(lo))
        future_oil_upper.append(float(hi))
        future_dates.append(next_date)

        q_history.append(float(next_q))
        current_date = next_date

        # Threshold-based stopping condition replacing dcalike.py's fixed horizon.
        if next_q <= stop_oil:
            break

    return pd.DataFrame(
        {
            "TEST_DATE": future_dates,
            "PRED_LOG_DECLINE": future_log_declines,
            "FORECAST_OIL": future_oil_predictions,
            "FORECAST_OIL_LOWER": future_oil_lower,
            "FORECAST_OIL_UPPER": future_oil_upper,
        }
    )


def run_dca_pipeline(
    df,
    minimum_history=30,
    train_fraction=0.8,
    stop_oil=1.0,
    max_forecast_steps=100000,
    confidence_percentiles=(5, 95),
):
    """Run end-to-end DCA-like ML forecasting without saving CSV/PNG files.

    Forecasting is threshold-driven through stop_oil. max_forecast_steps is
    only a safety cap to prevent an infinite loop if the threshold is very low.
    """
    X_all, y_all, target_indices, model_df = create_dataset(df, minimum_history=minimum_history)
    if len(X_all) == 0:
        raise ValueError("Not enough rows after preprocessing to build training samples.")

    split = split_train_test_by_time(
        X_all,
        y_all,
        target_indices,
        total_rows=len(model_df),
        train_fraction=train_fraction,
    )
    train_size = split["train_size"]
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    test_target_indices = split["test_target_indices"]

    train_result = train_decline_model(X_train, y_train)
    model = train_result["model"]

    test_dates = model_df["NEXT_TEST_DATE"].iloc[test_target_indices].to_numpy()
    oil_at_t = model_df["OIL_SMOOTH"].iloc[test_target_indices].to_numpy()
    test_delta_t = model_df["DELTA_T"].iloc[test_target_indices].to_numpy()
    actual_test_oil = model_df["OIL_NEXT"].iloc[test_target_indices].to_numpy()

    if len(X_test) > 0:
        predicted_test_log_decline = model.predict(X_test)
        predicted_test_log_decline = np.minimum(predicted_test_log_decline, 0.0)
        predicted_test_oil = oil_at_t * np.exp(predicted_test_log_decline * test_delta_t)

        tree_pred = np.vstack([est.predict(X_test.to_numpy()) for est in model.estimators_]).T
        tree_pred = np.minimum(tree_pred, 0.0)
        test_oil_dist = oil_at_t[:, None] * np.exp(tree_pred * test_delta_t[:, None])
        predicted_test_oil_lower = np.percentile(test_oil_dist, confidence_percentiles[0], axis=1)
        predicted_test_oil_upper = np.percentile(test_oil_dist, confidence_percentiles[1], axis=1)

        test_metrics = evaluate_oil_predictions(actual_test_oil, predicted_test_oil)
    else:
        predicted_test_log_decline = np.array([])
        predicted_test_oil = np.array([])
        predicted_test_oil_lower = np.array([])
        predicted_test_oil_upper = np.array([])
        test_metrics = {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "r2": np.nan}

    forecast_df = recursive_forecast(
        model=model,
        model_df=model_df,
        feature_columns=X_all.columns,
        stop_oil=stop_oil,
        max_forecast_steps=max_forecast_steps,
        confidence_percentiles=confidence_percentiles,
    )

    test_result_df = pd.DataFrame(
        {
            "TEST_DATE": test_dates,
            "ACTUAL_TEST_OIL": actual_test_oil,
            "PREDICTED_TEST_OIL": predicted_test_oil,
            "PREDICTED_TEST_OIL_LOWER": predicted_test_oil_lower,
            "PREDICTED_TEST_OIL_UPPER": predicted_test_oil_upper,
            "PRED_LOG_DECLINE": predicted_test_log_decline,
        }
    )

    return {
        "model": model,
        "model_df": model_df,
        "X_all": X_all,
        "y_all": y_all,
        "train_size": train_size,
        "train_result": train_result,
        "test_result_df": test_result_df,
        "test_metrics": test_metrics,
        "forecast_df": forecast_df,
    }

