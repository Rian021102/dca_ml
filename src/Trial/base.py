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
    return df

# Make features functions
def make_features_01(df):
    #only use TEST_DATE as time index and OIL as target variable
    df['t'] = (df['TEST_DATE'] - df['TEST_DATE'].min()).dt.days
    return df

def make_features_02(df):
    # use TEST_DATE as time index
    df['t'] = (df['TEST_DATE'] - df['TEST_DATE'].min()).dt.days
    # Create lag features for the target variable
    df['rate_1']=df['OIL'].shift(1)
    df['rate_2']=df['OIL'].shift(2)
    df['rate_3']=df['OIL'].shift(3)
    #replace 0 with NaN and Dropna for all new features
    df[['OIL','rate_1','rate_2','rate_3']] = df[['OIL','rate_1','rate_2','rate_3']].replace(0, np.nan)
    df = df.dropna(subset=['OIL', 'rate_1', 'rate_2', 'rate_3'])
    return df

def make_features_03(df):
     # use TEST_DATE as time index
    df['t'] = (df['TEST_DATE'] - df['TEST_DATE'].min()).dt.days
    df['1/logq']=1/np.log(df['OIL'].replace(0, np.nan))
    #rolling mean
    df['q_rolling_mean'] = df['OIL'].rolling(window=3, min_periods=1).mean()
    #dq/dt
    df['dq_dt'] = df['OIL'].diff() / df['t'].diff()
    #logq
    df['logq'] = np.log(df['OIL'].replace(0, np.nan))
    # Replace zeros to avoid log/ratio issues
    df['q'] = df['OIL'].replace(0, np.nan)
    return df

# End of feature engineering functions

def main():
    # path='/home/rian/python_project/myvenv/dca_ml/data/sel_wells.xlsx'
    path='/home/rian/python_project/myvenv/dca_ml/data/Volve production data.xlsx'

    df=load(path)
    df=df[['TEST_DATE','OIL','WELL_NAME']]
    well_name='ATTAKA B-8RD1'
    df=df[df['WELL_NAME']==well_name].copy()
    #crete features
    df=make_features_03(df)

    split = int(len(df) * 0.8)
    train_dates = df['TEST_DATE'].iloc[:split]
    test_dates = df['TEST_DATE'].iloc[split:]
    X_train, X_test = df.iloc[:split], df.iloc[split:]
    y_train, y_test = df['OIL'].iloc[:split], df['OIL'].iloc[split:]

    #select features to use for training
    feature_cols = ['t']
    # feature_cols = ['rate_1', 'rate_2', 'rate_3']
    # feature_cols = ['t', '1/logq', 'q_rolling_mean',
    #                 'dq_dt','logq']
    

    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    #train model
    model=RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train,y_train)
    #predict
    y_pred=model.predict(X_test)
    
    #evaluate
    mse=mean_squared_error(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}')

    #save image of train, test actual, and test predicted values
    plt.figure(figsize=(10,6))
    plt.plot(train_dates, y_train, label='Train (Actual)', marker='o', alpha=0.7)
    plt.plot(test_dates, y_test, label='Test (Actual)', marker='o')
    plt.plot(test_dates, y_pred, label='Test (Predicted)', marker='x', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Oil Production')
    plt.title(f'Actual vs Predicted Oil Production for {well_name}')
    plt.legend()
    plt.grid()
    images_dir = Path('/home/rian/python_project/myvenv/dca_ml/Images')
    images_dir.mkdir(parents=True, exist_ok=True)
    safe_well_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in well_name)
    output_path = images_dir / f'{safe_well_name}_actual_vs_predicted.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'Saved plot: {output_path}')

if __name__ == "__main__":
    main()
