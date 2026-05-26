import pandas as pd
from dca_ml.src.combined_rf_models import build_features,run_dcalike_model,run_rf_decline_rate_model,run_rf_loss_ratio_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    # Load data
    path = "/home/rian/python_project/myvenv/dca_ml/data/test_data.csv"
    df = pd.read_csv(path)
    df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
    df = df.sort_values("TEST_DATE").dropna(subset=["OIL"]).reset_index(drop=True)
    df = df[df["OIL"] > 0].reset_index(drop=True)

    # Smooth oil rate
    df["OIL_SMOOTH"] = df["OIL"].rolling(window=7, min_periods=1).median()
    q = df["OIL_SMOOTH"].values.astype(float)

    # Run the three models
    print("Running DCALike model...")
    forecast_dcalike = run_dcalike_model(df, q)

    print("Running RF Decline Rate model...")
    forecast_decline_rate = run_rf_decline_rate_model(df, q)

    print("Running RF Loss Ratio model...")
    forecast_loss_ratio = run_rf_loss_ratio_model(df, q)

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(q, label="Historical (Smoothed)", color="black", linewidth=2)
    
    # Create x-axis for forecast
    forecast_idx = np.arange(len(q), len(q) + 1000)
    
    plt.plot(forecast_idx, forecast_dcalike, label="DCALike Forecast", linestyle="--")
    plt.plot(forecast_idx, forecast_decline_rate, label="RF Decline Rate Forecast", linestyle="--")
    plt.plot(forecast_idx, forecast_loss_ratio, label="RF Loss Ratio Forecast", linestyle="--")
    
    plt.title("Comparison of RF-based DCA Forecasts")
    plt.xlabel("Days")
    plt.ylabel("Oil Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("forecast_comparison.png")
    print("Plot saved as forecast_comparison.png")

if __name__ == "__main__":
    main()