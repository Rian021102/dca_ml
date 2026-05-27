import streamlit as st
from src.combined_rf_models import run_dcalike_model, run_rf_decline_rate_model, run_rf_loss_ratio_model
from src.exponent import exponential
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

        df["t"] = (df["TEST_DATE"] - df["TEST_DATE"].iloc[0]).dt.days
        q = df["OIL"].values.astype(float)
        q_smooth_vis = pd.Series(q).rolling(window=7, min_periods=1).median().values

        qi_idx = st.select_slider(
            "Qi point slider (Exponential method only)",
            options=list(range(len(df))),
            value=0,
            format_func=lambda i: f"{df.loc[i, 'TEST_DATE'].date()} | OIL={float(df.loc[i, 'OIL']):.4f}",
            help="Move this slider to change only the Exponential method qi point.",
        )
        qi_date = df.loc[qi_idx, "TEST_DATE"]
        qi = float(df.loc[qi_idx, "OIL"])
        st.caption(f"Exponential qi is taken from: {qi_date.date()} (OIL={qi:.4f})")

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

        # Exponential decline forecast using slider-selected qi date
        forecast_horizon = 1000
        exp_fit_df = df.loc[qi_idx:, ["TEST_DATE", "OIL"]].copy().reset_index(drop=True)
        exp_fit_df["t"] = (exp_fit_df["TEST_DATE"] - qi_date).dt.days.astype(float)

        f_exp = np.array([])
        di = np.nan
        if len(exp_fit_df) >= 2:
            try:
                di = exponential(exp_fit_df, qi=qi)
                exp_values = []
                last_date = df["TEST_DATE"].iloc[-1]
                for step in range(1, forecast_horizon + 1):
                    t_future = (last_date + pd.Timedelta(days=step) - qi_date).days
                    next_q = qi * np.exp(-di * t_future)
                    if next_q <= stop_threshold:
                        exp_values.append(0.0)
                        break
                    exp_values.append(next_q)
                f_exp = np.array(exp_values, dtype=float)
            except Exception as exc:
                st.warning(f"Exponential decline fit failed: {exc}")

        # Clip to zero
        def clip_to_zero(forecast):
            return np.where(forecast <= stop_threshold, 0, forecast)

        f_dcalike = clip_to_zero(forecast_dcalike)
        f_decline = clip_to_zero(forecast_decline_rate)
        f_loss = clip_to_zero(forecast_loss_ratio)
        f_exp = clip_to_zero(f_exp)

        # Cumulative Sums
        hist_sum = np.sum(q)
        st.subheader("Cumulative Production (k-units)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Historical", f"{hist_sum/1000:.2f}")
        col2.metric("DCALike", f"{(hist_sum + np.sum(f_dcalike))/1000:.2f}")
        col3.metric("RF Decline", f"{(hist_sum + np.sum(f_decline))/1000:.2f}")
        col4.metric("RF Loss", f"{(hist_sum + np.sum(f_loss))/1000:.2f}")
        col5.metric("Exponential", f"{(hist_sum + np.sum(f_exp))/1000:.2f}")

        if np.isfinite(di):
            st.write(f"Exponential decline parameter Di: {di:.6f}")

        # Plotting results (interactive)
        dates = df["TEST_DATE"]
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
        forecast_dates_exp = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=len(f_exp), freq="D"
        )
        y_max = max(
            np.max(q) if len(q) else 0,
            np.max(q_smooth_vis) if len(q_smooth_vis) else 0,
            np.max(f_dcalike) if len(f_dcalike) else 0,
            np.max(f_decline) if len(f_decline) else 0,
            np.max(f_loss) if len(f_loss) else 0,
            np.max(f_exp) if len(f_exp) else 0,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=q,
                mode="lines+markers",
                name="Historical (Actual)",
                line=dict(color="rgba(60, 60, 60, 0.7)", width=1.5),
                marker=dict(size=4, color="rgba(60, 60, 60, 0.7)"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=q_smooth_vis,
                mode="lines",
                name="Historical (Smoothed)",
                line=dict(color="royalblue", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_dates_dcalike,
                y=f_dcalike,
                mode="lines",
                name="DCALike Forecast",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_dates_decline,
                y=f_decline,
                mode="lines",
                name="RF Decline Rate Forecast",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_dates_loss,
                y=f_loss,
                mode="lines",
                name="RF Loss Ratio Forecast",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_dates_exp,
                y=f_exp,
                mode="lines",
                name="Exponential Forecast",
                line=dict(dash="dash"),
            )
        )

        # Draw qi marker as a standard scatter line to avoid Timestamp arithmetic in add_vline.
        fig.add_trace(
            go.Scatter(
                x=[qi_date, qi_date],
                y=[0, y_max],
                mode="lines",
                name="Qi date (Exponential)",
                line=dict(color="red", width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[qi_date],
                y=[qi],
                mode="markers",
                name="Qi point",
                marker=dict(color="red", size=9, symbol="diamond"),
            )
        )

        fig.update_layout(
            title="Comparison of RF-based DCA Forecasts with Actual Dates",
            xaxis_title="Date",
            yaxis_title="Oil Rate",
            hovermode="x unified",
            legend_title_text="Series",
            template="plotly_white",
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()