import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data function
def load(path):
    # Load
    df = pd.read_excel(path)
    col_checker = ['DATEPRD', 'BORE_OIL_VOL', 'WELL_BORE_CODE']
    #if any col_checkers in df.columns:
    for col in col_checker:
        if col in df.columns:
         #replace DATEPRD with TEST_DATE and BORE_OIL_VOL with OIL
            if col == 'DATEPRD':
                df.rename(columns={col: 'TEST_DATE'}, inplace=True)
                df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'])
            elif col == 'BORE_OIL_VOL':
                df.rename(columns={col: 'OIL'}, inplace=True)
            elif col=='WELL_BORE_CODE':
                df.rename(columns={col:'WELL_NAME'},inplace=True)
    if 'TEST_DATE' in df.columns:
        df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'], errors='coerce')
    # #replace 0 as NaN in OIL column
    if 'OIL' in df.columns:
        df['OIL'] = df['OIL'].replace(0, np.nan)
        #drop rows with NaN in OIL column
        df = df.dropna(subset=['OIL'])
    return df

def create_window_features(df, target_col="OIL", window_size=10):
    """
    Example:
    [q(t-5), q(t-4), q(t-3), q(t-2), q(t-1)] -> q(t)
    """

    values = df[target_col].values

    X, y = [], []

    for i in range(window_size, len(values)):
        past_window = values[i-window_size:i]
        target = values[i]

        X.append(past_window)
        y.append(target)

    feature_cols = [f"{target_col}_lag_{i}" for i in range(window_size, 0, -1)]

    X = pd.DataFrame(X, columns=feature_cols)
    y = pd.Series(y, name=target_col)

    return X, y


def main():
    path='/home/rian/python_project/myvenv/dca_ml/data/Volve production data.xlsx'
    df = load(path)
    df=df[['TEST_DATE','OIL','WELL_NAME']]
    well_name='NO 15/9-F-14 H'
    df=df[df['WELL_NAME']==well_name].copy()
    #filter df to 2015-07-01 onward
    # df = df[df['TEST_DATE'] >= '2015-07-01'].copy()

    target_col = "OIL"
    window_size = 5

    df = df.sort_values("TEST_DATE").reset_index(drop=True)

    X, y = create_window_features(
        df=df,
        target_col=target_col,
        window_size=window_size
    )

    # time-series split, not random split
    split = int(len(X) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # =========================
    # Train Random Forest
    # =========================
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )

    rf.fit(X_train, y_train)

    # =========================
    # Predict
    # =========================
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
    print("Test RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("Test R2   :", r2_score(y_test, y_pred_test))


    # =========================
    # Plot
    # =========================
    dates = df["TEST_DATE"].iloc[window_size:].reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y.values, label="Actual")
    plt.plot(dates.iloc[:split], y_pred_train, label="Train Prediction")
    plt.plot(dates.iloc[split:], y_pred_test, label="Test Prediction")

    plt.title("Random Forest with LSTM-like Sliding Window Features")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    images_dir = Path('/home/rian/python_project/myvenv/dca_ml/Images')
    images_dir.mkdir(parents=True, exist_ok=True)
    safe_well_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in well_name)
    output_path = images_dir / f'{safe_well_name}_rf_window_actual_vs_predicted.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'Saved plot: {output_path}')
    
    

if __name__ == "__main__":
    main()