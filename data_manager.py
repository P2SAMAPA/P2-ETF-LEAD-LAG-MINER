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
    """Convert UNIX timestamp to datetime, set index, sort, and compute log returns."""
    # Convert UNIX seconds to datetime
    df["date"] = pd.to_datetime(df["__index_level_0__"], unit="s")
    df = df.set_index("date").sort_index()

    # Drop the index column
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])

    # Compute daily log returns for all price columns (ETFs)
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
