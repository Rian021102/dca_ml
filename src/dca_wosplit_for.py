import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import TimeSeriesSplit


def load(path):
    '''
    Load the production data from an Excel file and preprocess the date column.
     - Convert 'DATEPRD' to datetime format for time series analysis.
     - Ensure the data is sorted by date for proper time series modeling.
     - Handle any missing or zero values in 'BORE_OIL_VOL' to avoid issues with log transformations and feature engineering later on.
     - Return the cleaned DataFrame ready for feature engineering.
    '''
    # Load
    df = pd.read_excel(path)
    df['DATEPRD'] = pd.to_datetime(df['DATEPRD'])
    return df

def features(df):
    '''
    Create features for the DCA-like model:
     - 't': Time index in days since the first production date.
    '''
    # use DATEPRD as time index
    df['t'] = (df['DATEPRD'] - df['DATEPRD'].min()).dt.days
    df['1/logq']=1/np.log(df['BORE_OIL_VOL'].replace(0, np.nan))
    #rolling mean
    df['q_rolling_mean'] = df['BORE_OIL_VOL'].rolling(window=3, min_periods=1).mean()
    #dq/dt
    df['dq_dt'] = df['BORE_OIL_VOL'].diff() / df['t'].diff()
    #logq
    df['logq'] = np.log(df['BORE_OIL_VOL'].replace(0, np.nan))
    #lagged features only from previous time steps to avoid leakage
    df['q_lag1'] = df['BORE_OIL_VOL'].shift(1)
    df['q_lag2'] = df['BORE_OIL_VOL'].shift(2)
    df['q_lag3'] = df['BORE_OIL_VOL'].shift(3)
    # Replace zeros to avoid log/ratio issues
    df['q'] = df['BORE_OIL_VOL'].replace(0, np.nan)
    return df


def clean_training_data(df):
    '''
    Clean the training data for the DCA-like model:
     - Keep only rows with finite numeric values needed by the model.
     - Drop rows with NaN values in the 't' or 'q' columns.
    '''
    # Keep only rows with finite numeric values needed by the model
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['t', 'q']).copy()
    return clean_df


def forecast_future(rf, df, feature_cols, forecast_steps=30):
    '''
    Forecast future q values iteratively using lag-based features.
    '''
    if forecast_steps <= 0:
        return pd.DataFrame(columns=['DATEPRD', 'forecast_q'])

    # Use a robust step size from historical dates.
    day_diffs = df['DATEPRD'].diff().dropna().dt.days
    step_days = int(day_diffs.median()) if not day_diffs.empty else 1
    if step_days <= 0:
        step_days = 1

    last_date = df['DATEPRD'].iloc[-1]
    last_t = int(df['t'].iloc[-1])
    q_lag1 = float(df['q'].iloc[-1])
    q_lag2 = float(df['q'].iloc[-2])
    q_lag3 = float(df['q'].iloc[-3])

    future_dates = []
    future_preds = []

    for step in range(1, forecast_steps + 1):
        next_t = last_t + step_days
        q_roll = np.mean([q_lag1, q_lag2, q_lag3])
        dq_dt = (q_lag1 - q_lag2) / step_days

        # Approximate log-based terms from latest known/predicted rate.
        safe_q = max(q_lag1, 1e-6)
        logq = np.log(safe_q)
        inv_logq = 1.0 / logq if abs(logq) > 1e-12 else np.nan

        x_next = pd.DataFrame([
            {
                't': next_t,
                '1/logq': inv_logq,
                'q_rolling_mean': q_roll,
                'dq_dt': dq_dt,
                'logq': logq,
                'q_lag1': q_lag1,
                'q_lag2': q_lag2,
                'q_lag3': q_lag3,
            }
        ])[feature_cols]

        # Keep feature vector finite for inference.
        x_next = x_next.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        q_next = float(rf.predict(x_next)[0])
        q_next = max(q_next, 0.0)

        next_date = last_date + pd.Timedelta(days=step_days * step)
        future_dates.append(next_date)
        future_preds.append(q_next)

        # Roll lags for next step.
        q_lag3 = q_lag2
        q_lag2 = q_lag1
        q_lag1 = q_next
        last_t = next_t

    return pd.DataFrame({'DATEPRD': future_dates, 'forecast_q': future_preds})

def main():
    path='/home/rian/python_project/myvenv/dca_ml/data/Volve production data.xlsx'
    df = load(path)
    well_name='NO 15/9-F-12 H'
    df = df[df['WELL_BORE_CODE'] == well_name].copy()
    # Aggregate by date
    df = df.groupby('DATEPRD', as_index=False)['BORE_OIL_VOL'].sum()
    df = df.sort_values('DATEPRD').reset_index(drop=True)
    df = features(df)
    df = clean_training_data(df)

    if len(df) < 5:
        raise ValueError('Not enough valid non-NaN rows after cleaning to train the model.')
    
    feature_cols = ['t', '1/logq', 'q_rolling_mean',
                    'dq_dt','logq',
                    'q_lag1', 'q_lag2', 'q_lag3']

    X = df[feature_cols]
    y = df['q']
    dates = df['DATEPRD']
    # Train with walk-forward cross-validation on the full history
    n_splits = min(5, len(X) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        fold_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = fold_model.predict(X_fold_val)

        fold_mse = mean_squared_error(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)

        cv_mse_scores.append(fold_mse)
        cv_mae_scores.append(fold_mae)
        cv_r2_scores.append(fold_r2)
        print(f'Fold {fold} -> MSE: {fold_mse:.2f}, MAE: {fold_mae:.2f}, R2: {fold_r2:.4f}')

    print(
        'Walk-forward CV avg -> '
        f'MSE: {np.mean(cv_mse_scores):.2f}, '
        f'MAE: {np.mean(cv_mae_scores):.2f}, '
        f'R2: {np.mean(cv_r2_scores):.4f}'
    )

    # Final model fit on the full available history
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X, y)
    y_fitted = rf.predict(X)

    # In-sample fit metrics (no train/test split requested)
    mse = mean_squared_error(y, y_fitted)
    mae = mean_absolute_error(y, y_fitted)
    r2 = r2_score(y, y_fitted)
    print(f'In-sample fit -> MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}')

    forecast_steps = 30
    forecast_df = forecast_future(rf, df, feature_cols, forecast_steps=forecast_steps)
    print(f'Forecast generated for {forecast_steps} future steps.')
    print(forecast_df.head(10))

    # Plot
    plt.figure(figsize=(12,6))
    plt.scatter(dates, y, label='Actual', s=24, color='tab:blue')
    plt.plot(dates, y_fitted, label='Fitted', linewidth=2, color='tab:green')
    if not forecast_df.empty:
        plt.plot(
            forecast_df['DATEPRD'],
            forecast_df['forecast_q'],
            label='Forecast',
            linewidth=2,
            linestyle='--',
            color='tab:red'
        )
    plt.title('DCA-like Random Forest (Full Training + Forecast)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_dir = Path('/home/rian/python_project/myvenv/dca_ml/Images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'dca04_actual_fitted_forecast.png'
    plt.savefig(output_file, dpi=150)

    forecast_file = output_dir / 'dca04_forecast_values.csv'
    forecast_df.to_csv(forecast_file, index=False)

    # Show only when running with an interactive backend.
    if 'agg' not in matplotlib.get_backend().lower():
        plt.show()
    plt.close()
    print(f'Plot saved to: {output_file}')
    print(f'Forecast values saved to: {forecast_file}')

    # Feature importance
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(imp)

if __name__ == "__main__":
    main()