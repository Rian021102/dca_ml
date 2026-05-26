import pandas as pd
from combined_rf_models import build_features,run_dcalike_model,run_rf_decline_rate_model,run_rf_loss_ratio_model
from exponent import exponential
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Exponential Model
    print("Running Exponential model...")
    # The exponential function in exponent.py expects df with 't' and 'OIL'
    # We use the smoothed oil for consistency
    df_exp = df.copy()
    df_exp["OIL"] = df["OIL_SMOOTH"]
    di = exponential(df_exp)
    
    # Generate exponential forecast
    qi = q[0]
    forecast_horizon = 1000
    t_start = df["t"].iloc[-1]
    t_forecast = np.arange(t_start + 1, t_start + forecast_horizon + 1)
    forecast_exp = qi * np.exp(-di * t_forecast)

    # 1. Forecast until OIL is equal to zero (or very small)
    # Since these are exponential/decay models, they never hit exactly 0.
    # We'll clip them at a threshold (e.g., 0.1 bbl/d)
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
    f_exp = clip_to_zero(forecast_exp)

    # 2. Print Cumulative Sum (divided by 1000)
    # Total = Sum(Historical) + Sum(Forecast)
    hist_sum = np.sum(q)
    
    print("\n--- Cumulative Production (k-units) ---")
    print(f"Historical: {hist_sum/1000:.2f}")
    print(f"DCALike Forecast: {(hist_sum + np.sum(f_dcalike))/1000:.2f}")
    print(f"RF Decline Rate Forecast: {(hist_sum + np.sum(f_decline))/1000:.2f}")
    print(f"RF Loss Ratio Forecast: {(hist_sum + np.sum(f_loss))/1000:.2f}")
    print(f"Exponential Forecast: {(hist_sum + np.sum(f_exp))/1000:.2f}")

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
    plt.plot(forecast_dates, f_exp, label="Exponential Forecast", linestyle="--", color="red")
    
    plt.title("Comparison of DCA Forecasts with Actual Dates")
    plt.xlabel("Date")
    plt.ylabel("Oil Rate")
    plt.legend()
    plt.grid(True)
    
    # Format X axis to show dates clearly
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(AutoDateLocator())
    plt.gcf().autofmt_xdate()

    plt.savefig("forecast_comparison.png")
    print("\nPlot saved as forecast_comparison.png")

if __name__ == "__main__":
    main()