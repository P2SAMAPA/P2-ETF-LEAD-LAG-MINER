"""
Fetch and prepare data from Hugging Face dataset.
"""
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config


def load_master_data() -> pd.DataFrame:
    """Download master_data.parquet from HF and return DataFrame."""
    print(f"Downloading {config.HF_INPUT_FILE} from {config.HF_INPUT_DATASET}...")
    file_path = hf_hub_download(
        repo_id=config.HF_INPUT_DATASET,
        filename=config.HF_INPUT_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
    )
    df = pd.read_parquet(file_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp index to datetime, sort, and compute log returns.
    Expects the index to be UNIX milliseconds (values ~1.2e12).
    """
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame index dtype:", df.index.dtype)
    print("DataFrame index sample (first 5):", df.index[:5].tolist() if len(df) > 0 else "empty")

    # Strategy 1: Index is already datetime
    if pd.api.types.is_datetime64_any_dtype(df.index):
        print("Index is already datetime. Using as is.")
        df = df.sort_index()
        return compute_returns(df)

    # Strategy 2: Index is numeric – convert based on magnitude
    if pd.api.types.is_numeric_dtype(df.index):
        sample_val = df.index[0] if len(df) > 0 else 0
        if sample_val > 1e12:  # Nanoseconds
            unit = "ns"
        elif sample_val > 1e10:  # Milliseconds
            unit = "ms"
        elif sample_val > 1e9:   # Seconds
            unit = "s"
        else:
            unit = None

        if unit is not None:
            print(f"Converting numeric index to datetime using unit='{unit}'.")
            df.index = pd.to_datetime(df.index, unit=unit)
            df = df.sort_index()
            return compute_returns(df)

    # Strategy 3: Look for a timestamp column if index isn't the timestamp
    possible_time_cols = ["__index_level_0__", "date", "Date", "timestamp", "time", "index"]
    time_col = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col is not None:
        print(f"Found timestamp column: {time_col}")
        # Convert to datetime
        if pd.api.types.is_numeric_dtype(df[time_col]):
            sample_val = df[time_col].iloc[0]
            if sample_val > 1e12:
                unit = "ns"
            elif sample_val > 1e10:
                unit = "ms"
            elif sample_val > 1e9:
                unit = "s"
            else:
                unit = None
            if unit:
                df["date"] = pd.to_datetime(df[time_col], unit=unit)
            else:
                df["date"] = pd.to_datetime(df[time_col])
        else:
            df["date"] = pd.to_datetime(df[time_col])
        df = df.set_index("date")
        if time_col != "date":
            df = df.drop(columns=[time_col])
        df = df.sort_index()
        return compute_returns(df)

    # Strategy 4: Try parsing each column as datetime
    for col in df.columns:
        try:
            converted = pd.to_datetime(df[col])
            if converted.notna().all():
                print(f"Column '{col}' can be parsed as datetime. Using it.")
                df["date"] = converted
                df = df.set_index("date")
                df = df.drop(columns=[col])
                df = df.sort_index()
                return compute_returns(df)
        except:
            continue

    # If all strategies fail
    raise KeyError(
        f"Unable to locate date information. Columns: {df.columns.tolist()}, "
        f"Index dtype: {df.index.dtype}, Index sample: {df.index[:5].tolist()}"
    )


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns for price columns (ETFs)."""
    # Identify price columns as those not in MACRO_COLS
    price_cols = [col for col in df.columns if col not in config.MACRO_COLS]
    for col in price_cols:
        df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))
    return df


def get_universe_returns(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Return DataFrame of returns for given universe ('fi' or 'equity')."""
    if universe == "fi":
        tickers = config.FI_COMMODITY_TICKERS
    elif universe == "equity":
        tickers = config.EQUITY_TICKERS
    else:
        raise ValueError("universe must be 'fi' or 'equity'")

    ret_cols = [f"{t}_ret" for t in tickers if f"{t}_ret" in df.columns]
    return df[ret_cols].dropna()


def get_date_range_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Slice DataFrame between start_date and end_date inclusive."""
    return df.loc[start_date:end_date]


def get_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of macro columns."""
    available_macro = [col for col in config.MACRO_COLS if col in df.columns]
    return df[available_macro].dropna()
