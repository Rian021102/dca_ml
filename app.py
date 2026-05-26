import streamlit as st
from src.combined_rf_models import build_features, run_dcalike_model, run_rf_decline_rate_model, run_rf_loss_ratio_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter, AutoDateLocator

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
        df["t"] = (df["TEST_DATE"] - df["TEST_DATE"].iloc[0]).dt.days
        q = df["OIL_SMOOTH"].values.astype(float)

        stop_threshold = st.number_input(
            "Forecast stop threshold (OIL)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Forecast stops when predicted OIL is less than or equal to this value.",
        )

        st.write("Data loaded successfully. Running models...")

        # Run the three RF models
        forecast_dcalike = run_dcalike_model(df, q, oil_zero_threshold=stop_threshold)
        forecast_decline_rate = run_rf_decline_rate_model(df, q, oil_zero_threshold=stop_threshold)
        forecast_loss_ratio = run_rf_loss_ratio_model(df, q, oil_zero_threshold=stop_threshold)

        # Clip to zero
        def clip_to_zero(forecast):
            return np.where(forecast <= stop_threshold, 0, forecast)

        f_dcalike = clip_to_zero(forecast_dcalike)
        f_decline = clip_to_zero(forecast_decline_rate)
        f_loss = clip_to_zero(forecast_loss_ratio)

        # Cumulative Sums
        hist_sum = np.sum(q)
        st.subheader("Cumulative Production (k-units)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Historical", f"{hist_sum/1000:.2f}")
        col2.metric("DCALike", f"{(hist_sum + np.sum(f_dcalike))/1000:.2f}")
        col3.metric("RF Decline", f"{(hist_sum + np.sum(f_decline))/1000:.2f}")
        col4.metric("RF Loss", f"{(hist_sum + np.sum(f_loss))/1000:.2f}")

        # Plotting results
        plt.figure(figsize=(12, 6))
        dates = df["TEST_DATE"]
        plt.plot(dates, q, label="Historical (Smoothed)", color="black", linewidth=2)
        
        last_date = df["TEST_DATE"].iloc[-1]

        forecast_dates_dcalike = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=len(f_dcalike), freq="D"
        )
        forecast_dates_decline = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=len(f_decline), freq="D"
        )
        forecast_dates_loss = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=len(f_loss), freq="D"
        )

        plt.plot(forecast_dates_dcalike, f_dcalike, label="DCALike Forecast", linestyle="--")
        plt.plot(forecast_dates_decline, f_decline, label="RF Decline Rate Forecast", linestyle="--")
        plt.plot(forecast_dates_loss, f_loss, label="RF Loss Ratio Forecast", linestyle="--")
        
        plt.title("Comparison of RF-based DCA Forecasts with Actual Dates")
        plt.xlabel("Date")
        plt.ylabel("Oil Rate")
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        st.pyplot(plt)
if __name__ == "__main__":
    main()