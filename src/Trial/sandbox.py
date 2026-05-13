import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load(path):
    df = pd.read_excel(path)
    df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
    return df


def prepare_data(df, well_name):
    df = df[df["WELL_NAME"] == well_name].copy()
    df = df[df["TEST_DATE"] >= "2023-01-01"].copy()

    df = (
        df.groupby("TEST_DATE", as_index=False)["OIL"]
        .sum()
        .sort_values("TEST_DATE")
        .reset_index(drop=True)
    )

    df["q"] = df["OIL"].replace(0, np.nan)
    df["t"] = (df["TEST_DATE"] - df["TEST_DATE"].min()).dt.days

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["t", "q"]).copy()

    return df


def forecast_future(rf, df, forecast_steps=100):
    day_diffs = df["TEST_DATE"].diff().dropna().dt.days
    step_days = int(day_diffs.median()) if not day_diffs.empty else 1

    if step_days <= 0:
        step_days = 1

    last_date = df["TEST_DATE"].iloc[-1]
    last_t = df["t"].iloc[-1]

    future_dates = []
    future_q = []

    for i in range(1, forecast_steps + 1):
        next_t = last_t + step_days * i
        next_date = last_date + pd.Timedelta(days=step_days * i)

        X_next = pd.DataFrame({
            "t": [next_t]
        })

        q_next = rf.predict(X_next)[0]
        q_next = max(q_next, 0)

        future_dates.append(next_date)
        future_q.append(q_next)

    forecast_df = pd.DataFrame({
        "TEST_DATE": future_dates,
        "forecast_q": future_q
    })

    return forecast_df


def main():
    path = "/home/rian/python_project/myvenv/dca_ml/data/sel_wells.xlsx"
    well_name = "MAHONI-13BP1"

    output_dir = Path("/home/rian/python_project/myvenv/dca_ml/Images")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load(path)
    df = prepare_data(df, well_name)

    if len(df) < 5:
        raise ValueError("Not enough data to train the model.")

    # ======================================================
    # Friend's simple sandbox logic:
    # predictor: t
    # target: q(t)
    # ======================================================
    X = df[["t"]]
    y = df["q"]
    dates = df["TEST_DATE"]

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    )

    rf.fit(X, y)

    y_fitted = rf.predict(X)

    mse = mean_squared_error(y, y_fitted)
    mae = mean_absolute_error(y, y_fitted)
    r2 = r2_score(y, y_fitted)

    print("Simple RF DCA-like model")
    print("Predictor: t")
    print("Target   : q(t)")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 : {r2:.4f}")

    forecast_steps = 100
    forecast_df = forecast_future(rf, df, forecast_steps=forecast_steps)

    forecast_file = output_dir / "simple_rf_t_q_forecast_values.csv"
    forecast_df.to_csv(forecast_file, index=False)

    plt.figure(figsize=(12, 6))

    plt.scatter(
        dates,
        y,
        label="Actual",
        s=24
    )

    plt.plot(
        dates,
        y_fitted,
        label="RF fitted q(t)",
        linewidth=2
    )

    plt.plot(
        forecast_df["TEST_DATE"],
        forecast_df["forecast_q"],
        label="RF forecast",
        linestyle="--",
        linewidth=2
    )

    plt.title("Simple Random Forest DCA-like Model: q = f(t)")
    plt.xlabel("Date")
    plt.ylabel("Oil Rate")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = output_dir / "simple_rf_t_q_forecast.png"
    plt.savefig(output_file, dpi=150)

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()

    plt.close()

    print(f"Plot saved to: {output_file}")
    print(f"Forecast values saved to: {forecast_file}")

    imp = pd.DataFrame({
        "feature": ["t"],
        "importance": rf.feature_importances_
    })

    print("\nFeature Importance:")
    print(imp)


if __name__ == "__main__":
    main()