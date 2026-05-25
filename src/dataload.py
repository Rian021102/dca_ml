import pandas as pd
import numpy as np

def load_data(path):
    #if csv file
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    #if excel file
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"])
    df = df.sort_values("TEST_DATE").dropna(subset=["OIL"]).reset_index(drop=True)
    return df


