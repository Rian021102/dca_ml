import pandas as pd
from combined_rf_models import run_dcalike_model, run_rf_decline_rate_model, run_rf_loss_ratio_model
from newmodel import run_dca_pipeline
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, AutoDateLocator


def main():
    # Load data
    path = "/home/rian/python_project/myvenv/dca_ml/data/test_data.csv"
    df = pd.read_csv(path)
    df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
    df = df.sort_values("TEST_DATE").dropna(subset=["OIL"]).reset_index(drop=True)
    df = df[df["OIL"] > 0].reset_index(drop=True)

    # Time index in days from first date
    df["t"] = (df["TEST_DATE"] - df["TEST_DATE"].iloc[0]).dt.days

    # Smooth oil rate
    df["OIL_SMOOTH"] = df["OIL"].rolling(window=7, min_periods=1).median()
    q = df["OIL_SMOOTH"].values.astype(float)

    # Run the three RF models
    print("Running DCALike model...")
    forecast_dcalike = run_dcalike_model(df, q)

    print("Running RF Decline Rate model...")
    forecast_decline_rate = run_rf_decline_rate_model(df, q)

    print("Running RF Loss Ratio model...")
    forecast_loss_ratio = run_rf_loss_ratio_model(df, q)

    print("Running NewModel pipeline...")
    newmodel_result = run_dca_pipeline(df)
    newmodel_forecast_df = newmodel_result.get("forecast_df", pd.DataFrame())

    forecast_horizon = 1000

    # Clip forecasts below threshold
    threshold = 0.1

    def clip_to_zero(forecast):
        clipped = []
        for val in forecast:
            if val < threshold:
                clipped.append(0)
            else:
                clipped.append(val)
        return np.array(clipped)

    f_dcalike = clip_to_zero(forecast_dcalike)
    f_decline = clip_to_zero(forecast_decline_rate)
    f_loss = clip_to_zero(forecast_loss_ratio)
    f_newmodel = (
        clip_to_zero(newmodel_forecast_df["FORECAST_OIL"].to_numpy(dtype=float))
        if not newmodel_forecast_df.empty and "FORECAST_OIL" in newmodel_forecast_df.columns
        else np.array([])
    )

    # Print Cumulative Sum (divided by 1000)
    hist_sum = np.sum(q)

    print("\n--- Cumulative Production (k-units) ---")
    print(f"Historical: {hist_sum/1000:.2f}")
    print(f"DCALike Forecast: {(hist_sum + np.sum(f_dcalike))/1000:.2f}")
    print(f"RF Decline Rate Forecast: {(hist_sum + np.sum(f_decline))/1000:.2f}")
    print(f"RF Loss Ratio Forecast: {(hist_sum + np.sum(f_loss))/1000:.2f}")
    print(f"NewModel Forecast: {(hist_sum + np.sum(f_newmodel))/1000:.2f}")

    # Plotting results with actual dates
    plt.figure(figsize=(14, 7))

    # X-axis as dates
    dates = df["TEST_DATE"]
    plt.plot(dates, q, label="Historical (Smoothed)", color="black", linewidth=2)

    # Forecast dates
    last_date = df["TEST_DATE"].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")

    plt.plot(forecast_dates, f_dcalike, label="DCALike Forecast", linestyle="--")
    plt.plot(forecast_dates, f_decline, label="RF Decline Rate Forecast", linestyle="--")
    plt.plot(forecast_dates, f_loss, label="RF Loss Ratio Forecast", linestyle="--")

    if not newmodel_forecast_df.empty and {"TEST_DATE", "FORECAST_OIL"}.issubset(newmodel_forecast_df.columns):
        plt.plot(
            pd.to_datetime(newmodel_forecast_df["TEST_DATE"]),
            f_newmodel,
            label="NewModel Forecast",
            linestyle="--",
        )

    plt.title("Comparison of DCA Forecasts with Actual Dates")
    plt.xlabel("Date")
    plt.ylabel("Oil Rate")
    plt.legend()
    plt.grid(True)

    # Format X axis to show dates clearly
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(AutoDateLocator())
    plt.gcf().autofmt_xdate()

    plt.savefig("forecast_comparison.png")
    print("\nPlot saved as forecast_comparison.png")


if __name__ == "__main__":
    main()
