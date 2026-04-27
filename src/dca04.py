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
    # Split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    # Train with walk-forward cross-validation on the training window
    tscv = TimeSeriesSplit(n_splits=5)
    cv_mse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

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

    # Final model fit on the full train window
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
    output_file = output_dir / 'dca04_actual_vs_predicted.png'
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