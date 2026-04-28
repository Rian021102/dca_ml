import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 


def load(path):
    # Load
    df = pd.read_excel(path)
    df['DATEPRD'] = pd.to_datetime(df['DATEPRD'])
    return df

def features(df):
    # use DATEPRD as time index
    df['t'] = (df['DATEPRD'] - df['DATEPRD'].min()).dt.days
    df['1/logq']=1/np.log(df['BORE_OIL_VOL'].replace(0, np.nan))
    #rolling mean
    df['q_rolling_mean'] = df['BORE_OIL_VOL'].rolling(window=3, min_periods=1).mean()
    #dq/dt
    df['dq_dt'] = df['BORE_OIL_VOL'].diff() / df['t'].diff()
    #logq
    df['logq'] = np.log(df['BORE_OIL_VOL'].replace(0, np.nan))
    # Replace zeros to avoid log/ratio issues
    df['q'] = df['BORE_OIL_VOL'].replace(0, np.nan)
    return df


def clean_training_data(df):
    # Keep only rows with finite numeric values needed by the model
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['t', 'q']).copy()
    return clean_df

def main():
    path='/home/rian/python_project/myvenv/dca_ml/data/Volve production data.xlsx'
    df = load(path)
    well_name='NO 15/9-F-14 H'
    df = df[df['WELL_BORE_CODE'] == well_name].copy()
    # Aggregate by date
    df = df.groupby('DATEPRD', as_index=False)['BORE_OIL_VOL'].sum()
    df = df.sort_values('DATEPRD').reset_index(drop=True)
    df = features(df)
    df = clean_training_data(df)

    if len(df) < 5:
        raise ValueError('Not enough valid non-NaN rows after cleaning to train the model.')
    
    feature_cols = ['t', '1/logq', 'q_rolling_mean',
                    'dq_dt','logq']

    # X = df[feature_cols]
    # y = df['q']
    dates = df['DATEPRD']
    # Split
    split = int(len(df) * 0.8)
    X_train, X_test = df.iloc[:split], df.iloc[split:]
    y_train, y_test = df['q'].iloc[:split], df['q'].iloc[split:]
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]
   

    # Train
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    # Predict
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    y_pred = y_pred_test
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}')
    # Plot
    plt.figure(figsize=(12,6))
    plt.scatter(dates[:split], y_train, label='Train Actual', s=28, color='tab:blue')
    plt.scatter(dates[split:], y_test, label='Test Actual', s=28, color='tab:orange')
    plt.plot(dates[split:], y_pred_test, label='Test Pred', linewidth=2, color='tab:green')
    plt.title("DCA-like Random Forest (Leakage Fixed)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_dir = Path('/home/rian/python_project/myvenv/dca_ml/Images')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'dca03_actual_vs_predicted.png'
    plt.savefig(output_file, dpi=150)

    # Show only when running with an interactive backend.
    if 'agg' not in matplotlib.get_backend().lower():
        plt.show()
    plt.close()
    print(f'Plot saved to: {output_file}')

    # Feature importance
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(imp)

if __name__ == "__main__":
    main()