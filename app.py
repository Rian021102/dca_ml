import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
from src.datautils import load_data
from src.dcamodel import run_dca_pipeline
from src.exponent import exponential_with_uncertainty
import plotly.graph_objects as go


@st.cache_data(show_spinner=False)
def _load_data_from_upload(file_bytes, file_name):
    buf = io.BytesIO(file_bytes)
    buf.name = file_name
    return load_data(buf)


def _prepare_exponential_fit(df, start_date, end_date):
    mask = (df["TEST_DATE"] >= pd.to_datetime(start_date)) & (df["TEST_DATE"] <= pd.to_datetime(end_date))
    exp_input = df.loc[mask].copy().reset_index(drop=True)
    if len(exp_input) < 3:
        raise ValueError("Need at least 3 filtered rows for exponential fit.")

    exp_df = exp_input[["TEST_DATE", "OIL_SMOOTH"]].rename(columns={"OIL_SMOOTH": "OIL"})
    exp_df["t"] = (
        (exp_df["TEST_DATE"] - exp_df["TEST_DATE"].iloc[0]).dt.total_seconds() / 86400.0
    )

    qi = float(exp_df["OIL"].iloc[0])
    di, di_lo, di_hi = exponential_with_uncertainty(exp_df[["t", "OIL"]], qi=qi)
    di = max(float(di), 0.0)
    di_lo = max(float(di_lo), 0.0)
    di_hi = max(float(di_hi), di_lo)

    t_vals = exp_df["t"].to_numpy(dtype=float)
    exp_df["EXP_FIT_OIL"] = qi * np.exp(-di * t_vals)
    exp_df["EXP_FIT_OIL_LOWER"] = qi * np.exp(-di_hi * t_vals)
    exp_df["EXP_FIT_OIL_UPPER"] = qi * np.exp(-di_lo * t_vals)
    return exp_df, qi, di, di_lo, di_hi


def _build_exponential_forecast(model_df, di, di_lower, di_upper, stop_oil=1.0, max_forecast_steps=1000):
    if len(model_df) == 0:
        return pd.DataFrame(columns=["TEST_DATE", "EXP_FORECAST_OIL", "EXP_FORECAST_OIL_LOWER", "EXP_FORECAST_OIL_UPPER"])

    q_t = float(model_df["OIL_SMOOTH"].iloc[-1])
    current_date = pd.to_datetime(model_df["TEST_DATE"].iloc[-1])
    freq = pd.infer_freq(model_df["TEST_DATE"]) or "D"
    date_offset = pd.tseries.frequencies.to_offset(freq)

    out_dates = []
    out_values = []
    out_lower = []
    out_upper = []

    for _ in range(max_forecast_steps):
        next_date = current_date + date_offset
        step_t = (next_date - current_date) / np.timedelta64(1, "D")
        if step_t <= 0:
            step_t = 1.0

        q_next = q_t * np.exp(-di * step_t)
        q_next = max(float(q_next), float(stop_oil))

        q_next_lo = q_t * np.exp(-di_upper * step_t)
        q_next_hi = q_t * np.exp(-di_lower * step_t)
        q_next_lo = max(float(q_next_lo), float(stop_oil))
        q_next_hi = max(float(q_next_hi), q_next_lo)

        out_dates.append(next_date)
        out_values.append(q_next)
        out_lower.append(q_next_lo)
        out_upper.append(q_next_hi)

        current_date = next_date
        q_t = q_next
        if q_t <= stop_oil:
            break

    return pd.DataFrame(
        {
            "TEST_DATE": out_dates,
            "EXP_FORECAST_OIL": out_values,
            "EXP_FORECAST_OIL_LOWER": out_lower,
            "EXP_FORECAST_OIL_UPPER": out_upper,
        }
    )


def _build_cumulative_frame(full_df, ml_forecast_df, exp_forecast_df):
    actual_base = full_df[["TEST_DATE", "OIL_SMOOTH"]].copy()
    actual_base["CUM_ACTUAL"] = actual_base["OIL_SMOOTH"].cumsum()

    last_actual_cum = float(actual_base["CUM_ACTUAL"].iloc[-1]) if len(actual_base) > 0 else 0.0

    ml_cum = ml_forecast_df[["TEST_DATE", "FORECAST_OIL", "FORECAST_OIL_LOWER", "FORECAST_OIL_UPPER"]].copy()
    if len(ml_cum) > 0:
        ml_cum["CUM_ML"] = last_actual_cum + ml_cum["FORECAST_OIL"].cumsum()
        ml_cum["CUM_ML_LOWER"] = last_actual_cum + ml_cum["FORECAST_OIL_LOWER"].cumsum()
        ml_cum["CUM_ML_UPPER"] = last_actual_cum + ml_cum["FORECAST_OIL_UPPER"].cumsum()

        # Extend ML cumulative to start at the first actual date.
        ml_hist = actual_base[["TEST_DATE", "CUM_ACTUAL"]].copy()
        ml_hist = ml_hist.rename(columns={"CUM_ACTUAL": "CUM_ML"})
        ml_hist["CUM_ML_LOWER"] = ml_hist["CUM_ML"]
        ml_hist["CUM_ML_UPPER"] = ml_hist["CUM_ML"]
        ml_cum = pd.concat(
            [ml_hist[["TEST_DATE", "CUM_ML", "CUM_ML_LOWER", "CUM_ML_UPPER"]], ml_cum[["TEST_DATE", "CUM_ML", "CUM_ML_LOWER", "CUM_ML_UPPER"]]],
            ignore_index=True,
        ).drop_duplicates(subset=["TEST_DATE"], keep="first").sort_values("TEST_DATE")

    exp_cum = exp_forecast_df[["TEST_DATE", "EXP_FORECAST_OIL", "EXP_FORECAST_OIL_LOWER", "EXP_FORECAST_OIL_UPPER"]].copy()
    if len(exp_cum) > 0:
        exp_cum["CUM_EXP"] = last_actual_cum + exp_cum["EXP_FORECAST_OIL"].cumsum()
        exp_cum["CUM_EXP_LOWER"] = last_actual_cum + exp_cum["EXP_FORECAST_OIL_LOWER"].cumsum()
        exp_cum["CUM_EXP_UPPER"] = last_actual_cum + exp_cum["EXP_FORECAST_OIL_UPPER"].cumsum()

        # Extend exponential cumulative to start at the first actual date.
        exp_hist = actual_base[["TEST_DATE", "CUM_ACTUAL"]].copy()
        exp_hist = exp_hist.rename(columns={"CUM_ACTUAL": "CUM_EXP"})
        exp_hist["CUM_EXP_LOWER"] = exp_hist["CUM_EXP"]
        exp_hist["CUM_EXP_UPPER"] = exp_hist["CUM_EXP"]
        exp_cum = pd.concat(
            [exp_hist[["TEST_DATE", "CUM_EXP", "CUM_EXP_LOWER", "CUM_EXP_UPPER"]], exp_cum[["TEST_DATE", "CUM_EXP", "CUM_EXP_LOWER", "CUM_EXP_UPPER"]]],
            ignore_index=True,
        ).drop_duplicates(subset=["TEST_DATE"], keep="first").sort_values("TEST_DATE")

    return actual_base, ml_cum, exp_cum


def _df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def _run_dca_cached(file_bytes, file_name, minimum_history, train_fraction, stop_oil, max_forecast_steps):
    buf = io.BytesIO(file_bytes)
    buf.name = file_name
    base_df = load_data(buf)
    return run_dca_pipeline(
        df=base_df,
        minimum_history=minimum_history,
        train_fraction=train_fraction,
        stop_oil=stop_oil,
        max_forecast_steps=max_forecast_steps,
    )


def _save_results_to_directory(save_dir, prefix, forecast_df, test_df, metrics_dict, exp_df=None, exp_forecast_df=None):
    os.makedirs(save_dir, exist_ok=True)

    forecast_path = os.path.join(save_dir, f"{prefix}_forecast.csv")
    test_path = os.path.join(save_dir, f"{prefix}_test_results.csv")
    metrics_path = os.path.join(save_dir, f"{prefix}_metrics.csv")

    forecast_df.to_csv(forecast_path, index=False)
    test_df.to_csv(test_path, index=False)
    pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)

    saved_paths = [forecast_path, test_path, metrics_path]
    if exp_df is not None and len(exp_df) > 0:
        exp_path = os.path.join(save_dir, f"{prefix}_exponential_fit.csv")
        exp_df.to_csv(exp_path, index=False)
        saved_paths.append(exp_path)
    if exp_forecast_df is not None and len(exp_forecast_df) > 0:
        exp_forecast_path = os.path.join(save_dir, f"{prefix}_exponential_forecast.csv")
        exp_forecast_df.to_csv(exp_forecast_path, index=False)
        saved_paths.append(exp_forecast_path)

    return saved_paths


def main():
    st.title("DCA Forecasting Using Machine Learning Models")
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV or Excel)",
        type=["csv", "xlsx", "xls", "xlsm", "xlsb", "ods"],
    )
    if uploaded_file is not None:
        try:
            df = _load_data_from_upload(uploaded_file.getvalue(), uploaded_file.name)
        except Exception as exc:
            st.error(f"Error loading data: {exc}")
            return

        st.success("Data loaded successfully")
        st.dataframe(df.head())

        st.subheader("Input Data")

        date_options = sorted(pd.to_datetime(df["TEST_DATE"]).dt.date.unique().tolist())
        if len(date_options) < 3:
            st.error("Need at least 3 unique dates for exponent filter range.")
            return

        exp_start_date, exp_end_date = st.select_slider(
            "Data slider for exponent.py only (date range)",
            options=date_options,
            value=(date_options[0], date_options[-1]),
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["TEST_DATE"],
                y=df["OIL"],
                mode="markers",
                name="Actual OIL",
                marker={"size": 6, "opacity": 0.7},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["TEST_DATE"],
                y=df["OIL_SMOOTH"],
                mode="lines",
                name="Smoothed OIL",
                line={"width": 2},
            )
        )
        fig.add_vline(
            x=exp_start_date,
            line_width=2,
            line_dash="dash",
            line_color="firebrick",
        )
        fig.add_vline(
            x=exp_end_date,
            line_width=2,
            line_dash="dot",
            line_color="darkorange",
        )
        fig.update_layout(
            title="Actual vs Smoothed OIL (Exponent Range Markers)",
            xaxis_title="Date",
            yaxis_title="Oil Production",
            template="plotly_white",
            legend_title="Series",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Exponent filter range: {exp_start_date} to {exp_end_date}. "
            "This slider only affects exponent.py fitting, not DCA model training."
        )

        exp_df = pd.DataFrame()
        exp_forecast_df = pd.DataFrame()
        di = 0.0
        di_lo = 0.0
        di_hi = 0.0
        try:
            exp_df, qi, di, di_lo, di_hi = _prepare_exponential_fit(df, exp_start_date, exp_end_date)
            st.caption(
                f"Exponential fit using smoothed OIL in range {exp_start_date} to {exp_end_date} "
                f"(qi={qi:.4f}, Di={di:.6f}/day, CI: {di_lo:.6f}-{di_hi:.6f})"
            )

            st.subheader("Exponential Decline Plot")
            fig_exp = go.Figure()
            fig_exp.add_trace(
                go.Scatter(
                    x=exp_df["TEST_DATE"],
                    y=exp_df["OIL"],
                    mode="markers",
                    name="Filtered Smoothed OIL",
                    marker={"size": 6, "opacity": 0.7},
                )
            )
            fig_exp.add_trace(
                go.Scatter(
                    x=exp_df["TEST_DATE"],
                    y=exp_df["EXP_FIT_OIL"],
                    mode="lines",
                    name="Exponential Fit",
                    line={"width": 3},
                )
            )
            fig_exp.add_trace(
                go.Scatter(
                    x=exp_df["TEST_DATE"],
                    y=exp_df["EXP_FIT_OIL_UPPER"],
                    mode="lines",
                    name="Exp Fit Upper",
                    line={"width": 0},
                    showlegend=False,
                )
            )
            fig_exp.add_trace(
                go.Scatter(
                    x=exp_df["TEST_DATE"],
                    y=exp_df["EXP_FIT_OIL_LOWER"],
                    mode="lines",
                    name="Exp Fit CI",
                    fill="tonexty",
                    fillcolor="rgba(255, 102, 0, 0.60)",
                    line={"width": 0},
                )
            )
            fig_exp.update_layout(
                title="Exponential Decline Fit on Filtered Smoothed Data",
                xaxis_title="Date",
                yaxis_title="OIL",
                template="plotly_white",
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        except Exception as exc:
            st.warning(f"Exponential fit not available for current filter: {exc}")

        st.subheader("DCA Model Run")
        minimum_history = st.slider("Minimum history for features", min_value=10, max_value=60, value=30, step=1)
        train_fraction = st.slider("Train fraction", min_value=0.6, max_value=0.95, value=0.8, step=0.05)
        stop_oil = st.number_input("Forecast stop oil", min_value=0.1, value=1.0, step=0.1)
        max_forecast_steps = st.number_input("Max forecast steps", min_value=10, value=1000, step=10)

        if st.button("Run DCA Forecast"):
            try:
                with st.spinner("Running DCA pipeline..."):
                    result = _run_dca_cached(
                        file_bytes=uploaded_file.getvalue(),
                        file_name=uploaded_file.name,
                        minimum_history=int(minimum_history),
                        train_fraction=float(train_fraction),
                        stop_oil=float(stop_oil),
                        max_forecast_steps=int(max_forecast_steps),
                    )
                    st.session_state["dca_result"] = result
            except Exception as exc:
                st.error(f"Error running DCA model: {exc}")
                return

        result = st.session_state.get("dca_result")
        if not result:
            return

        metrics = result["test_metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test MAE", f"{metrics['mae']:.4f}" if metrics["mae"] == metrics["mae"] else "N/A")
        col2.metric("Test MSE", f"{metrics['mse']:.4f}" if metrics["mse"] == metrics["mse"] else "N/A")
        col3.metric("Test RMSE", f"{metrics['rmse']:.4f}" if metrics["rmse"] == metrics["rmse"] else "N/A")
        col4.metric("Test R2", f"{metrics['r2']:.4f}" if metrics["r2"] == metrics["r2"] else "N/A")

        model_df = result["model_df"]
        test_df = result["test_result_df"]
        forecast_df = result["forecast_df"]
        if di > 0:
            exp_forecast_df = _build_exponential_forecast(
                model_df=model_df,
                di=di,
                di_lower=di_lo,
                di_upper=di_hi,
                stop_oil=float(stop_oil),
                max_forecast_steps=int(max_forecast_steps),
            )

        actual_cum_df, ml_cum_df, exp_cum_df = _build_cumulative_frame(df, forecast_df, exp_forecast_df)

        ml_total_cum = float(ml_cum_df["CUM_ML"].iloc[-1]) if len(ml_cum_df) > 0 else np.nan
        exp_total_cum = float(exp_cum_df["CUM_EXP"].iloc[-1]) if len(exp_cum_df) > 0 else np.nan
        cum1, cum2 = st.columns(2)
        cum1.metric("Total Cumulative ML", f"{ml_total_cum:.2f}" if ml_total_cum == ml_total_cum else "N/A")
        cum2.metric("Total Cumulative Exponential", f"{exp_total_cum:.2f}" if exp_total_cum == exp_total_cum else "N/A")

        st.subheader("Result Plot 1: Test Actual vs Predicted")
        fig_test = go.Figure()
        fig_test.add_trace(
            go.Scatter(
                x=test_df["TEST_DATE"],
                y=test_df["ACTUAL_TEST_OIL"],
                mode="lines+markers",
                name="Actual Test OIL",
            )
        )
        fig_test.add_trace(
            go.Scatter(
                x=test_df["TEST_DATE"],
                y=test_df["PREDICTED_TEST_OIL"],
                mode="lines",
                name="Predicted Test OIL",
            )
        )
        fig_test.update_layout(
            title="Test Period: Actual vs Predicted OIL",
            xaxis_title="Date",
            yaxis_title="Smoothed OIL",
            template="plotly_white",
        )
        st.plotly_chart(fig_test, use_container_width=True)

        st.subheader("Result Plot 2: Historical + Forecast")
        c_toggle1, c_toggle2, c_toggle3, c_toggle4, c_toggle5 = st.columns(5)
        show_raw_oil = c_toggle1.checkbox("Raw OIL", value=True)
        show_smooth_oil = c_toggle2.checkbox("Smoothed OIL", value=True)
        show_dca_forecast = c_toggle3.checkbox("DCA Forecast", value=True)
        show_exp_fit = c_toggle4.checkbox("Exp Fit", value=True)
        show_exp_forecast = c_toggle5.checkbox("Exp Forecast", value=True)
        c_toggle6, c_toggle7, c_toggle8 = st.columns(3)
        show_cum_ml = c_toggle6.checkbox("Cum ML (y2)", value=True)
        show_cum_exp = c_toggle7.checkbox("Cum Exp (y2)", value=True)
        show_cum_ci = c_toggle8.checkbox("Cum CI Bands", value=True)

        fig_forecast = go.Figure()
        if show_raw_oil:
            fig_forecast.add_trace(
                go.Scatter(
                    x=model_df["TEST_DATE"],
                    y=model_df["OIL"],
                    mode="markers",
                    name="Raw OIL",
                    marker={"size": 5, "opacity": 0.5},
                )
            )
        if show_smooth_oil:
            fig_forecast.add_trace(
                go.Scatter(
                    x=model_df["TEST_DATE"],
                    y=model_df["OIL_SMOOTH"],
                    mode="lines",
                    name="Smoothed OIL",
                    line={"width": 2},
                )
            )
        if show_dca_forecast:
            fig_forecast.add_trace(
                go.Scatter(
                    x=forecast_df["TEST_DATE"],
                    y=forecast_df["FORECAST_OIL"],
                    mode="lines",
                    name="Forecast OIL",
                    line={"width": 3, "dash": "dash"},
                )
            )
            if len(forecast_df) > 0:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_df["TEST_DATE"],
                        y=forecast_df["FORECAST_OIL_UPPER"],
                        mode="lines",
                        line={"width": 0},
                        name="DCA Upper",
                        showlegend=False,
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_df["TEST_DATE"],
                        y=forecast_df["FORECAST_OIL_LOWER"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(0, 89, 255, 0.60)",
                        name="DCA CI",
                    )
                )
        if show_exp_fit and len(exp_df) > 0:
            fig_forecast.add_trace(
                go.Scatter(
                    x=exp_df["TEST_DATE"],
                    y=exp_df["EXP_FIT_OIL"],
                    mode="lines",
                    name="Exponential Fit (Filtered)",
                    line={"width": 3, "dash": "dot"},
                )
            )
        if show_exp_forecast and len(exp_forecast_df) > 0:
            fig_forecast.add_trace(
                go.Scatter(
                    x=exp_forecast_df["TEST_DATE"],
                    y=exp_forecast_df["EXP_FORECAST_OIL"],
                    mode="lines",
                    name="Exponential Forecast",
                    line={"width": 3, "dash": "dashdot"},
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=exp_forecast_df["TEST_DATE"],
                    y=exp_forecast_df["EXP_FORECAST_OIL_UPPER"],
                    mode="lines",
                    line={"width": 0},
                    name="Exp Fc Upper",
                    showlegend=False,
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=exp_forecast_df["TEST_DATE"],
                    y=exp_forecast_df["EXP_FORECAST_OIL_LOWER"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(255, 64, 0, 0.62)",
                    name="Exp Forecast CI",
                )
            )

        if show_cum_ml and len(ml_cum_df) > 0:
            fig_forecast.add_trace(
                go.Scatter(
                    x=ml_cum_df["TEST_DATE"],
                    y=ml_cum_df["CUM_ML"],
                    mode="lines",
                    name="Cumulative ML",
                    line={"width": 2, "dash": "longdash"},
                    yaxis="y2",
                )
            )
            if show_cum_ci:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=ml_cum_df["TEST_DATE"],
                        y=ml_cum_df["CUM_ML_UPPER"],
                        mode="lines",
                        line={"width": 0},
                        name="Cum ML Upper",
                        showlegend=False,
                        yaxis="y2",
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=ml_cum_df["TEST_DATE"],
                        y=ml_cum_df["CUM_ML_LOWER"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(0, 224, 255, 0.60)",
                        name="Cum ML CI",
                        yaxis="y2",
                    )
                )
        if show_cum_exp and len(exp_cum_df) > 0:
            fig_forecast.add_trace(
                go.Scatter(
                    x=exp_cum_df["TEST_DATE"],
                    y=exp_cum_df["CUM_EXP"],
                    mode="lines",
                    name="Cumulative Exp",
                    line={"width": 2, "dash": "longdashdot"},
                    yaxis="y2",
                )
            )
            if show_cum_ci:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=exp_cum_df["TEST_DATE"],
                        y=exp_cum_df["CUM_EXP_UPPER"],
                        mode="lines",
                        line={"width": 0},
                        name="Cum Exp Upper",
                        showlegend=False,
                        yaxis="y2",
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=exp_cum_df["TEST_DATE"],
                        y=exp_cum_df["CUM_EXP_LOWER"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(255, 179, 0, 0.60)",
                        name="Cum Exp CI",
                        yaxis="y2",
                    )
                )
        fig_forecast.update_layout(
            title="Historical, Recursive Forecast, Exponential Fit, and Exponential Forecast",
            xaxis_title="Date",
            yaxis_title="OIL",
            yaxis2={"title": "Cumulative OIL", "overlaying": "y", "side": "right", "showgrid": False},
            template="plotly_white",
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("Result Plot 3: Predicted vs Actual Scatter")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=test_df["PREDICTED_TEST_OIL"],
                y=test_df["ACTUAL_TEST_OIL"],
                mode="markers",
                name="Test points",
                marker={"size": 7, "opacity": 0.75},
            )
        )
        if len(test_df) > 0:
            min_v = float(min(test_df["PREDICTED_TEST_OIL"].min(), test_df["ACTUAL_TEST_OIL"].min()))
            max_v = float(max(test_df["PREDICTED_TEST_OIL"].max(), test_df["ACTUAL_TEST_OIL"].max()))
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_v, max_v],
                    y=[min_v, max_v],
                    mode="lines",
                    name="Ideal y=x",
                    line={"dash": "dash"},
                )
            )
        fig_scatter.update_layout(
            title="Predicted vs Actual Test OIL",
            xaxis_title="Predicted OIL",
            yaxis_title="Actual OIL",
            template="plotly_white",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Forecast Table")
        st.dataframe(forecast_df.head(20))
        if len(exp_forecast_df) > 0:
            st.subheader("Exponential Forecast Table")
            st.dataframe(exp_forecast_df.head(20))

        st.subheader("Cumulative Table")
        cum_table = pd.DataFrame()
        if len(ml_cum_df) > 0:
            cum_table = ml_cum_df[["TEST_DATE", "CUM_ML", "CUM_ML_LOWER", "CUM_ML_UPPER"]].copy()
        if len(exp_cum_df) > 0:
            exp_cols = exp_cum_df[["TEST_DATE", "CUM_EXP", "CUM_EXP_LOWER", "CUM_EXP_UPPER"]].copy()
            if len(cum_table) == 0:
                cum_table = exp_cols
            else:
                cum_table = cum_table.merge(exp_cols, on="TEST_DATE", how="outer").sort_values("TEST_DATE")
        if len(cum_table) > 0:
            st.dataframe(cum_table.head(30))

        st.subheader("Downloads")
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download Forecast CSV",
            data=_df_to_csv_bytes(forecast_df),
            file_name="forecast_output.csv",
            mime="text/csv",
        )
        c2.download_button(
            "Download Test Result CSV",
            data=_df_to_csv_bytes(test_df),
            file_name="test_result_output.csv",
            mime="text/csv",
        )

        st.subheader("Save Results to Folder")
        default_dir = os.path.join(os.getcwd(), "saved_results")
        save_dir = st.text_input("Target directory", value=default_dir)
        default_prefix = f"dca_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_prefix = st.text_input("Filename prefix", value=default_prefix)
        include_exp_fit = st.checkbox("Include exponential fit and forecast CSV", value=True)

        if st.button("Save Results to Directory"):
            try:
                exp_to_save = exp_df if include_exp_fit and "exp_df" in locals() else None
                saved = _save_results_to_directory(
                    save_dir=save_dir,
                    prefix=save_prefix,
                    forecast_df=forecast_df,
                    test_df=test_df,
                    metrics_dict=metrics,
                    exp_df=exp_to_save,
                    exp_forecast_df=exp_forecast_df if include_exp_fit else None,
                )
                st.success("Saved files:")
                for p in saved:
                    st.write(p)
            except Exception as exc:
                st.error(f"Failed to save results: {exc}")

       
if __name__ == "__main__":
    main()