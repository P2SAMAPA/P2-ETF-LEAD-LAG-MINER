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
    Convert timestamp to datetime, set index, sort, and compute log returns.
    Handles various possible timestamp column names.
    """
    # Try to identify the timestamp column
    possible_time_cols = ["__index_level_0__", "date", "timestamp", "time", "index"]
    time_col = None

    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        # If no timestamp column found, check if the index itself is a timestamp
        if df.index.dtype.kind in "iuf" and df.index.min() > 1e9:
            # Index looks like UNIX seconds
            df.index = pd.to_datetime(df.index, unit="s")
        else:
            raise KeyError("No timestamp column found and index is not UNIX seconds.")
    else:
        # Convert the timestamp column to datetime and set as index
        if df[time_col].dtype.kind in "iuf":
            # Looks like UNIX timestamp
            df["date"] = pd.to_datetime(df[time_col], unit="s")
        else:
            df["date"] = pd.to_datetime(df[time_col])
        df = df.set_index("date")
        # Drop the original timestamp column if it's not needed
        if time_col != "date":
            df = df.drop(columns=[time_col])

    df = df.sort_index()

    # Compute daily log returns for all price columns (ETFs)
    # Identify price columns as those that are not macro and not the date index
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
