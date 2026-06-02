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
    """Create supervised ML data where the target is DCA decline ratio.

    Target:
        D_t = ln(q_t / q_{t+1}) / delta_t

    Forecast equation:
        q_{t+1} = q_t * exp(-D_pred * delta_t)

    Features are built only from historical smoothed oil up to time t.
    """
    work = df.copy()
    work["OIL_NEXT"] = work["OIL_SMOOTH"].shift(-1)
    work["NEXT_TEST_DATE"] = work["TEST_DATE"].shift(-1)
    work["DELTA_T"] = (work["NEXT_TEST_DATE"] - work["TEST_DATE"]).dt.total_seconds() / 86400.0
    work.loc[work["DELTA_T"] <= 0, "DELTA_T"] = np.nan

    # Positive D means decline. Negative D means the smoothed rate increased;
    # remove those points so the learned model remains DCA-like decline-only.
    work["DEC_RATIO"] = np.log(work["OIL_SMOOTH"] / work["OIL_NEXT"]) / work["DELTA_T"]
    work.loc[work["DEC_RATIO"] < 0, "DEC_RATIO"] = np.nan
    work = work.dropna(subset=["DEC_RATIO", "OIL_NEXT", "NEXT_TEST_DATE", "DELTA_T"]).reset_index(drop=True)

    q = work["OIL_SMOOTH"].to_numpy(dtype=float)

    rows, targets, target_indices = [], [], []
    for i in range(minimum_history, len(work)):
        rows.append(build_features(q, i))
        targets.append(work["DEC_RATIO"].iloc[i])
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

def recursive_forecast(
    model,
    model_df,
    feature_columns,
    stop_oil=1.0,
    max_forecast_steps=1000,
    confidence_percentiles=(5, 95),
):
    future_oil_predictions = []
    future_dates = []
    future_decline_ratios = []
    future_oil_lower = []
    future_oil_upper = []

    q_history = model_df["OIL_SMOOTH"].to_numpy(dtype=float).tolist()
    q_t = q_history[-1]

    freq = pd.infer_freq(model_df["TEST_DATE"]) or "D"
    date_offset = pd.tseries.frequencies.to_offset(freq)
    current_date = model_df["TEST_DATE"].iloc[-1]

    for _ in range(max_forecast_steps):
        idx = len(q_history) - 1
        X_future = pd.DataFrame([build_features(q_history, idx)])
        X_future = X_future[feature_columns]

        pred_dec_ratio = float(model.predict(X_future)[0])
        pred_dec_ratio = max(pred_dec_ratio, 0.0)

        tree_pred = np.array([est.predict(X_future)[0] for est in model.estimators_], dtype=float)
        tree_pred = np.clip(tree_pred, 0.0, None)

        next_date = current_date + date_offset
        step_t = (next_date - current_date) / np.timedelta64(1, "D")
        if step_t <= 0:
            step_t = 1.0

        q_next = q_t * np.exp(-pred_dec_ratio * step_t)
        q_next = max(q_next, stop_oil)

        q_next_dist = q_t * np.exp(-tree_pred * step_t)
        lo = float(np.percentile(q_next_dist, confidence_percentiles[0]))
        hi = float(np.percentile(q_next_dist, confidence_percentiles[1]))
        lo = max(lo, float(stop_oil))
        hi = max(hi, lo)

        future_decline_ratios.append(pred_dec_ratio)
        future_oil_predictions.append(q_next)
        future_oil_lower.append(lo)
        future_oil_upper.append(hi)
        future_dates.append(next_date)

        q_history.append(q_next)
        q_t = q_next
        current_date = next_date

        if q_t <= stop_oil:
            break

    return pd.DataFrame(
        {
            "TEST_DATE": future_dates,
            "PRED_DEC_RATIO": future_decline_ratios,
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
    max_forecast_steps=1000,
    confidence_percentiles=(5, 95),
):
    """Run end-to-end DCA-like ML forecasting without saving CSV/PNG files."""
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
        predicted_test_dec_ratio = model.predict(X_test)
        predicted_test_dec_ratio = np.clip(predicted_test_dec_ratio, 0.0, None)
        predicted_test_oil = oil_at_t * np.exp(-predicted_test_dec_ratio * test_delta_t)

        tree_pred = np.vstack([est.predict(X_test) for est in model.estimators_]).T
        tree_pred = np.clip(tree_pred, 0.0, None)
        test_oil_dist = oil_at_t[:, None] * np.exp(-tree_pred * test_delta_t[:, None])
        predicted_test_oil_lower = np.percentile(test_oil_dist, confidence_percentiles[0], axis=1)
        predicted_test_oil_upper = np.percentile(test_oil_dist, confidence_percentiles[1], axis=1)

        test_metrics = evaluate_oil_predictions(actual_test_oil, predicted_test_oil)
    else:
        predicted_test_dec_ratio = np.array([])
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
            "PRED_DEC_RATIO": predicted_test_dec_ratio,
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