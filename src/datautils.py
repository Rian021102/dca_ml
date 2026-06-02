import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path=None):
    if path is None:
        raise ValueError("Please provide a file path to a CSV or Excel file.")

    source = path
    source_name = None

    # Accept path-like inputs and Streamlit UploadedFile/file-like inputs.
    if isinstance(source, (str, os.PathLike)):
        source_name = os.fspath(source)
        if not os.path.exists(source_name):
            raise FileNotFoundError(f"File not found: {source_name}")
    else:
        source_name = getattr(source, "name", None)
        if hasattr(source, "seek"):
            source.seek(0)

    # Load CSV or Excel based on extension (preferred) or parser fallback.
    ext = os.path.splitext(source_name)[1].lower() if source_name else ""
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(source)
    elif ext in [".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"]:
        df = pd.read_excel(source)
    else:
        try:
            df = pd.read_csv(source)
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)
            try:
                df = pd.read_excel(source)
            except Exception as exc:
                raise ValueError(
                    "Unsupported file type. Use CSV (.csv) or Excel (.xls/.xlsx/.xlsm/.xlsb/.ods)."
                ) from exc

    # Support common schema variants across DCA datasets.
    date_col = "TEST_DATE" if "TEST_DATE" in df.columns else ("DATEPRD" if "DATEPRD" in df.columns else None)
    oil_col = "OIL" if "OIL" in df.columns else ("BORE_OIL_VOL" if "BORE_OIL_VOL" in df.columns else None)
    if date_col is None or oil_col is None:
        raise KeyError("Expected date/oil columns not found. Need TEST_DATE/DATEPRD and OIL/BORE_OIL_VOL.")

    if date_col != "TEST_DATE":
        df["TEST_DATE"] = df[date_col]
    if oil_col != "OIL":
        df["OIL"] = df[oil_col]

    # Parse mixed date formats (e.g., 18-Nov-09 and 2009-11-18).
    df["TEST_DATE"] = pd.to_datetime(df["TEST_DATE"], format="mixed", dayfirst=True, errors="coerce")
    df["OIL"] = pd.to_numeric(df["OIL"], errors="coerce")

    df = df.dropna(subset=["TEST_DATE", "OIL"]).sort_values("TEST_DATE").reset_index(drop=True)
    df = df[df["OIL"] > 0].reset_index(drop=True)

    # Smoothing is used to make the ML target/forecast behave like DCA trend decline.
    df["OIL_SMOOTH"] = df["OIL"].rolling(window=7, min_periods=1).median()
    df["OIL_SMOOTH"] = df["OIL_SMOOTH"].clip(lower=1e-6)

    return df

def visualize_data(df, save_path=None):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="TEST_DATE", y="OIL", data=df, label="Raw OIL", alpha=0.45)
    sns.lineplot(x="TEST_DATE", y="OIL_SMOOTH", data=df, label="Smoothed OIL / DCA trend")
    plt.title("Oil Production Over Time")
    plt.xlabel("Date")
    plt.ylabel("Oil Production")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return save_path