import streamlit as st
from src.combined_rf_models import build_features, run_dcalike_model, run_rf_decline_rate_model, run_rf_loss_ratio_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("DCA Forecasting Using Machine Learning Models")
    #upload file either csv or excel
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
        df = df.sort_values("TEST_DATE").dropna(subset=["OIL"]).reset_index(drop=True)
        df = df[df["OIL"] > 0].reset_index(drop=True)

        # Smooth oil rate
        df["OIL_SMOOTH"] = df["OIL"].rolling(window=7, min_periods=1).median()
        q = df["OIL_SMOOTH"].values.astype(float)

        st.write("Data loaded successfully. Running models...")

        # Run the three models
        forecast_dcalike = run_dcalike_model(df, q)
        forecast_decline_rate = run_rf_decline_rate_model(df, q)
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
        
        st.pyplot(plt)
if __name__ == "__main__":
    main()