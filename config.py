"""
Configuration for Lead-Lag-Miner.
"""
import os
from datetime import datetime

# Hugging Face configuration
HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-lead-lag-miner-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME"]
BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"

# Macro columns (available in dataset)
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Lags for analysis (in days)
LAGS = [1, 3, 5, 10]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Minimum required data points for training
MIN_TRAIN_DAYS = 252 * 2  # 2 years

# Shrinking window start years
SHRINKING_START_YEARS = list(range(2008, 2025))  # 2008 through 2024

# Selection weights for shrinking window aggregation (updated: 60/10/10/20)
WEIGHT_RETURN = 0.6
WEIGHT_SHARPE = 0.1
WEIGHT_HITRATE = 0.1
WEIGHT_MAXDD = 0.2

# Annualization factor (trading days)
TRADING_DAYS_PER_YEAR = 252
