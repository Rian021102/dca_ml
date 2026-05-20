import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import TimeSeriesSplit

def load(path):
    df = pd.read_excel(path)
    # Adjust column name if necessary to match your Excel
    df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'])
    return df

def features(df):
    # Ensure min_date is consistent for 't' calculation
    min_date = df['TEST_DATE'].min()
    df['t'] = (df['TEST_DATE'] - min_date).dt.days
    df['logq'] = np.log(df['OIL'].replace(0, np.nan))
    df['1/logq'] = 1 / df['logq']
    df['q_rolling_mean'] = df['OIL'].rolling(window=3, min_periods=1).mean()
    df['dq_dt'] = df['OIL'].diff() / df['t'].diff()
    df['q_lag1'] = df['OIL'].shift(1)
    df['q_lag2'] = df['OIL'].shift(2)
    df['q_lag3'] = df['OIL'].shift(3)
    df['q'] = df['OIL'].replace(0, np.nan)
    return df

def clean_training_data(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=['t', 'q', 'q_lag3']).copy()

def forecast_future(model, last_row_df, n_days, feature_cols, min_date):
    '''
    Recursive forecasting: predict the next day, update lags, and repeat.
    '''
    future_predictions = []
    current_data = last_row_df.copy()
    
    # We need a small history to calculate rolling means and lags during recursion
    # For simplicity, we track the last few 'q' values
    history_q = [current_data['q_lag2'].iloc[0], current_data['q_lag1'].iloc[0], current_data['q'].iloc[0]]
    last_date = current_data['TEST_DATE'].iloc[0]
    last_t = current_data['t'].iloc[0]

    for _ in range(n_days):
        # 1. Increment Time
        last_date += pd.Timedelta(days=1)
        last_t += 1
        
        # 2. Update Features based on previous step
        q_prev = history_q[-1]
        dq_dt = (q_prev - history_q[-2]) / 1 # dt is 1 day
        logq = np.log(q_prev if q_prev > 0 else 1e-5)
        
        row = pd.DataFrame({
            't': [last_t],
            '1/logq': [1 / logq],
            'q_rolling_mean': [np.mean(history_q[-3:])],
            'dq_dt': [dq_dt],
            'logq': [logq],
            'q_lag1': [history_q[-1]],
            'q_lag2': [history_q[-2]],
            'q_lag3': [history_q[-3]]
        })
        
        # 3. Predict
        q_pred = model.predict(row[feature_cols])[0]
        # Ensure physical reality (oil can't be negative)
        q_pred = max(0, q_pred)
        
        # 4. Store and Update history
        future_predictions.append({'TEST_DATE': last_date, 'q_pred': q_pred})
        history_q.append(q_pred)
        
    return pd.DataFrame(future_predictions)

def main():
    path = 'P:/project/pythonpro/myvenv/dca_ml/data/sel_wells.xlsx'
    df_raw = load(path)
    well_name = 'ATTAKA B-8RD1'
    
    df = df_raw[df_raw['WELL_NAME'] == well_name].copy()
    df = df.groupby('TEST_DATE', as_index=False)['OIL'].sum()
    df = df.sort_values('TEST_DATE').reset_index(drop=True)
    
    min_date = df['TEST_DATE'].min()
    df = features(df)
    df_clean = clean_training_data(df)

    feature_cols = ['t', '1/logq', 'q_rolling_mean', 'dq_dt', 'logq', 'q_lag1', 'q_lag2', 'q_lag3']
    X = df_clean[feature_cols]
    y = df_clean['q']
    
    # Train on everything up to the split for validation, then refit on all for forecasting
    split = int(len(df_clean) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    
    # Forecasting 30 days into the future
    # We use the absolute last row of our cleaned data to start the forecast
    last_row = df_clean.tail(1)
    forecast_steps = 1000
    df_forecast = forecast_future(rf, last_row, forecast_steps, feature_cols, min_date)

    # Plotting
    plt.figure(figsize=(12,6))
    plt.scatter(df_clean['TEST_DATE'][:split], y_train, label='Train Actual', s=20)
    plt.scatter(df_clean['TEST_DATE'][split:], y_test, label='Test Actual', s=20)
    
    # Plot the Forecast
    plt.plot(df_forecast['TEST_DATE'], df_forecast['q_pred'], 
             label='Future Forecast', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"DCA Forecast: {well_name}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save Logic
    output_dir = Path('P:/project/pythonpro/myvenv/dca_ml/Images')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'dca_forecast.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()