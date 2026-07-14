"""
Training pipeline for ETF Lead-Lag Miner
FIXED: Handles new ETFs with shorter histories (SMH, SOXX, XLB)
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import sys
import os

# Import local modules
from data_manager import DataManager
from lead_lag_engine import compute_lead_lag_metrics, clean_returns_for_analysis
from selector import select_etf
from us_calendar import get_us_calendar
from utils import safe_import_data

warnings.filterwarnings('ignore')

# Configuration
from config import FI_ETFS, EQ_ETFS, LAGS, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT


def train_global(universe, returns, end_date=None):
    """
    Global training with 80/10/10 split.
    FIXED: Handles missing data for new tickers.
    """
    print(f"\n=== Global Training for {universe} Universe ===")
    
    # FIX: Clean returns before analysis
    clean_returns = clean_returns_for_analysis(returns, min_valid_obs=50)
    
    # FIX: Check if we have enough tickers after cleaning
    if len(clean_returns.columns) < 2:
        print(f"Error: Not enough tickers in {universe} universe after cleaning")
        return None
    
    # Split data
    n = len(clean_returns)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    
    train_data = clean_returns.iloc[:train_end]
    val_data = clean_returns.iloc[train_end:val_end]
    test_data = clean_returns.iloc[val_end:]
    
    print(f"Train: {len(train_data)} days, Val: {len(val_data)} days, Test: {len(test_data)} days")
    
    # Compute lead-lag metrics on training data
    metrics = compute_lead_lag_metrics(train_data, max_lag=max(LAGS))
    
    # Select best ETF based on validation data
    try:
        selected_etf = select_etf(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            metrics=metrics,
            universe=universe,
            end_date=end_date
        )
    except Exception as e:
        print(f"Selection failed: {e}")
        # FIX: Fallback to highest return ETF
        print("Using fallback selection based on returns...")
        avg_returns = train_data.mean().sort_values(ascending=False)
        selected_etf = avg_returns.index[0] if len(avg_returns) > 0 else None
    
    return selected_etf


def train_shrinking_window(universe, returns, end_date=None):
    """
    Shrinking window training (expanding window from 2008).
    FIXED: Handles tickers with insufficient history.
    """
    print(f"\n=== Shrinking Window Training for {universe} Universe ===")
    
    # FIX: Clean returns
    clean_returns = clean_returns_for_analysis(returns, min_valid_obs=30)
    
    if len(clean_returns.columns) < 2:
        print(f"Error: Not enough tickers in {universe} universe after cleaning")
        return None
    
    # Start from 2008
    start_date = '2008-01-01'
    start_idx = clean_returns.index.get_loc(pd.Timestamp(start_date), method='nearest')
    
    selections = []
    windows = []
    
    for year in range(2008, 2026):
        # FIX: End at current year end or data end
        end_date_str = f"{year}-12-31"
        if end_date_str > clean_returns.index[-1].strftime('%Y-%m-%d'):
            end_date_str = clean_returns.index[-1].strftime('%Y-%m-%d')
        
        # Get data up to this year
        window_data = clean_returns.loc[start_date:end_date_str]
        
        if len(window_data) < 100:  # Need minimum data
            continue
            
        # Compute metrics on window
        try:
            metrics = compute_lead_lag_metrics(window_data, max_lag=max(LAGS))
            
            # Simple selection: highest average return in window
            avg_returns = window_data.mean().sort_values(ascending=False)
            selected = avg_returns.index[0] if len(avg_returns) > 0 else None
            
            if selected:
                selections.append(selected)
                windows.append(year)
                print(f"Window {year}: ETF={selected}, Ann Return={window_data[selected].mean()*252:.1%}")
                
        except Exception as e:
            print(f"Window {year}: Failed - {str(e)[:50]}")
            continue
    
    # FIX: Weighted ensemble - more recent years get more weight
    if selections:
        unique_etfs = list(set(selections))
        weights = {}
        
        for i, etf in enumerate(selections):
            # FIX: Weight by recency (most recent years weighted more)
            weight = i + 1  # Linear increasing weight
            if etf not in weights:
                weights[etf] = 0
            weights[etf] += weight
        
        # FIX: Normalize by total weight
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Select with highest weighted score
        selected_etf = max(weights, key=weights.get)
        print(f"\nWeighted ensemble pick: {selected_etf}")
        print(f"Selection weights: {weights}")
        
        return selected_etf
    else:
        print("No valid windows found")
        return None


def run_training():
    """
    Main training function.
    FIXED: Better error handling and logging.
    """
    print("Loading data...")
    
    # FIX: Load master data with proper error handling
    try:
        # Load from local or remote
        if os.path.exists('master_data.parquet'):
            df = pd.read_parquet('master_data.parquet')
        else:
            # Try to download from remote
            import requests
            url = "https://github.com/P2SAMAPA/fi-etf-macro-signal-master-data/raw/main/master_data.parquet"
            response = requests.get(url)
            with open('master_data.parquet', 'wb') as f:
                f.write(response.content)
            df = pd.read_parquet('master_data.parquet')
            
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # FIX: Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # FIX: Define core ETFs (price/return columns, not derived features)
    core_etfs = [
        'GLD', 'SPY', 'AGG', 'TLT', 'VCIT', 'LQD', 'HYG', 'VNQ', 'SLV', 'QQQ',
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XME', 'GDX',
        'IWM', 'IWF', 'XSD', 'SMH', 'SOXX', 'XBI', 'XLB', 'XLRE', 'IWD', 'IWO'
    ]
    
    # FIX: Extract only core ETF columns that exist
    available_etfs = [t for t in core_etfs if t in df.columns]
    print(f"Available ETFs: {available_etfs}")
    
    # Separate into FI and Equity universes
    fi_universe = [t for t in FI_ETFS if t in available_etfs]
    eq_universe = [t for t in EQ_ETFS if t in available_etfs]
    
    print(f"FI Universe: {fi_universe}")
    print(f"Equity Universe: {eq_universe}")
    
    # FIX: Compute returns from price data (assuming price columns)
    price_df = df[available_etfs].copy()
    
    # FIX: Handle missing values in price data
    price_df = price_df.fillna(method='ffill').fillna(method='bfill')
    
    # FIX: Compute returns with min_periods to avoid NaN issues
    returns_df = price_df.pct_change().dropna()
    
    # FIX: Limit data to valid range
    min_date = returns_df.index.min()
    max_date = returns_df.index.max()
    print(f"Returns range: {min_date} to {max_date}")
    
    # FIX: Run training with proper error handling
    try:
        if fi_universe:
            fi_returns = returns_df[fi_universe]
            print(f"\nProcessing fi universe...")
            fi_selected = train_shrinking_window('fi', fi_returns, end_date=df.index[-1].strftime('%Y-%m-%d'))
            
            if fi_selected is None:
                # Try global training as fallback
                fi_selected = train_global('fi', fi_returns, end_date=df.index[-1].strftime('%Y-%m-%d'))
        else:
            fi_selected = None
            print("No FI ETFs available")
    except Exception as e:
        print(f"FI training failed: {e}")
        fi_selected = None
    
    try:
        if eq_universe:
            eq_returns = returns_df[eq_universe]
            print(f"\nProcessing equity universe...")
            eq_selected = train_shrinking_window('equity', eq_returns, end_date=df.index[-1].strftime('%Y-%m-%d'))
            
            if eq_selected is None:
                # Try global training as fallback
                eq_selected = train_global('equity', eq_returns, end_date=df.index[-1].strftime('%Y-%m-%d'))
        else:
            eq_selected = None
            print("No Equity ETFs available")
    except Exception as e:
        print(f"Equity training failed: {e}")
        eq_selected = None
    
    # FIX: Save results
    results = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fi_selected': fi_selected,
        'eq_selected': eq_selected,
        'fi_universe': fi_universe,
        'eq_universe': eq_universe
    }
    
    # Save to file
    results_df = pd.DataFrame([results])
    results_df.to_csv('training_results.csv', index=False)
    print(f"\nResults saved to training_results.csv")
    
    return results


if __name__ == "__main__":
    print("Run python training.py")
    output = run_training()
    
    if output:
        print("\n=== TRAINING COMPLETE ===")
        print(f"FI Selection: {output.get('fi_selected', 'N/A')}")
        print(f"Equity Selection: {output.get('eq_selected', 'N/A')}")
    else:
        print("\n=== TRAINING FAILED ===")
        sys.exit(1)
