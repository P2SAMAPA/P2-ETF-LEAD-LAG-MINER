"""
Lead-Lag Engine for ETF Temporal Asymmetry Detection
Fixed to handle tickers with different data lengths (like SMH, SOXX, XLB)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


def cross_correlation_matrix(returns_df, max_lag=5, min_obs=30):
    """
    Compute cross-correlation at various lags between all ETF pairs.
    FIXED: Handles NaN values and different series lengths gracefully.
    
    Args:
        returns_df: DataFrame of returns (rows=dates, columns=tickers)
        max_lag: Maximum lag to test
        min_obs: Minimum observations required for correlation calculation
        
    Returns:
        corr_matrix: DataFrame of max correlations
        corr_lag_matrix: DataFrame of lags at which max correlation occurs
    """
    tickers = returns_df.columns
    n = len(tickers)
    
    corr_matrix = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
    corr_lag_matrix = pd.DataFrame(np.zeros((n, n)), index=tickers, columns=tickers)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # FIX: Get clean series, drop NaNs
            series_i = returns_df.iloc[:, i].dropna()
            series_j = returns_df.iloc[:, j].dropna()
            
            # FIX: Align indices to ensure same dates
            common_idx = series_i.index.intersection(series_j.index)
            if len(common_idx) < min_obs:
                # Not enough overlapping data - skip
                corr_matrix.iloc[i, j] = np.nan
                corr_lag_matrix.iloc[i, j] = np.nan
                continue
                
            series_i = series_i.loc[common_idx]
            series_j = series_j.loc[common_idx]
            
            max_corr = -np.inf
            best_lag = 0
            
            # Check positive lags (i leads j)
            for lag in range(1, max_lag + 1):
                if len(series_i) <= lag:
                    break
                    
                # FIX: Ensure both segments have same length
                len_i = len(series_i) - lag
                len_j = len(series_j) - lag
                
                # FIX: Use min length to avoid size mismatch
                min_len = min(len_i, len_j)
                if min_len < min_obs:
                    break
                    
                seg_i = series_i.iloc[:min_len]
                seg_j = series_j.iloc[lag:lag+min_len]
                
                try:
                    # FIX: Use pearsonr which handles edge cases better
                    corr, _ = pearsonr(seg_i, seg_j)
                except (ValueError, RuntimeError):
                    corr = np.nan
                    
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
            
            # Check negative lags (j leads i)
            for lag in range(1, max_lag + 1):
                if len(series_j) <= lag:
                    break
                    
                len_i = len(series_i) - lag
                len_j = len(series_j) - lag
                
                min_len = min(len_i, len_j)
                if min_len < min_obs:
                    break
                    
                seg_i = series_i.iloc[lag:lag+min_len]
                seg_j = series_j.iloc[:min_len]
                
                try:
                    corr, _ = pearsonr(seg_i, seg_j)
                except (ValueError, RuntimeError):
                    corr = np.nan
                    
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = -lag
            
            corr_matrix.iloc[i, j] = max_corr if max_corr != -np.inf else np.nan
            corr_lag_matrix.iloc[i, j] = best_lag
    
    return corr_matrix, corr_lag_matrix


def granger_causality_matrix(returns_df, max_lag=5, significance=0.05):
    """
    Compute Granger causality p-values between all ETF pairs.
    FIXED: Handles tickers with insufficient data.
    """
    tickers = returns_df.columns
    n = len(tickers)
    
    p_matrix = pd.DataFrame(np.ones((n, n)), index=tickers, columns=tickers)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # FIX: Get clean data with overlapping dates
            series_i = returns_df.iloc[:, i].dropna()
            series_j = returns_df.iloc[:, j].dropna()
            
            common_idx = series_i.index.intersection(series_j.index)
            if len(common_idx) < 30:  # Minimum for Granger test
                continue
                
            # FIX: Ensure enough data for max_lag
            if len(common_idx) < 3 * max_lag + 1:
                # Use smaller max_lag if data is limited
                adjusted_lag = max(1, len(common_idx) // 10)
            else:
                adjusted_lag = max_lag
                
            try:
                data = pd.DataFrame({
                    'i': series_i.loc[common_idx],
                    'j': series_j.loc[common_idx]
                }).dropna()
                
                # Run Granger test with adjusted lag
                test_result = grangercausalitytests(data[['i', 'j']], 
                                                    maxlag=adjusted_lag, 
                                                    verbose=False)
                
                # Extract best p-value (min across lags)
                best_p = 1.0
                for lag in range(1, adjusted_lag + 1):
                    p_val = test_result[lag][0]['ssr_ftest'][1]
                    best_p = min(best_p, p_val)
                
                p_matrix.iloc[i, j] = best_p
                
            except (ValueError, IndexError, KeyError):
                # FIX: Skip problematic pairs
                continue
    
    return p_matrix


def var_impulse_response(returns_df, n_lags=3, n_periods=10):
    """
    Compute VAR impulse response functions.
    FIXED: Handles missing data and short histories.
    """
    # FIX: Drop any columns with all NaN
    clean_df = returns_df.dropna(axis=1, how='all')
    
    # FIX: Fill remaining NaNs with median to keep VAR stable
    clean_df = clean_df.fillna(clean_df.median())
    
    # FIX: Check if we have enough data
    if len(clean_df) < n_lags * 3:
        print(f"Warning: Not enough data for VAR (need {n_lags*3}, have {len(clean_df)})")
        return None, None
    
    try:
        model = VAR(clean_df)
        results = model.fit(maxlags=n_lags, ic='aic', verbose=False)
        irf = results.irf(periods=n_periods)
        
        # FIX: Handle potential errors in irf
        try:
            irf_matrix = irf.irfs
        except AttributeError:
            irf_matrix = None
            
        return results, irf_matrix
        
    except (ValueError, np.linalg.LinAlgError, Exception) as e:
        print(f"VAR estimation failed: {e}")
        return None, None


def transfer_entropy_matrix(returns_df, max_lag=3, k=3):
    """
    Compute transfer entropy between ETF pairs.
    FIXED: Simplified version that handles different lengths.
    """
    # FIX: Use mutual_info_regression as proxy for transfer entropy
    # (more robust with different data lengths)
    tickers = returns_df.columns
    n = len(tickers)
    
    te_matrix = pd.DataFrame(np.zeros((n, n)), index=tickers, columns=tickers)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # FIX: Get overlapping data
            series_i = returns_df.iloc[:, i].dropna()
            series_j = returns_df.iloc[:, j].dropna()
            
            common_idx = series_i.index.intersection(series_j.index)
            if len(common_idx) < 30:
                continue
                
            # FIX: Use lagged mutual information as proxy
            try:
                X = series_i.loc[common_idx].values.reshape(-1, 1)
                y = series_j.loc[common_idx].shift(-1).dropna().values
                
                if len(X) != len(y):
                    min_len = min(len(X), len(y))
                    X = X[:min_len]
                    y = y[:min_len]
                
                # Normalize for better entropy estimation
                X_norm = (X - X.mean()) / (X.std() + 1e-8)
                y_norm = (y - y.mean()) / (y.std() + 1e-8)
                
                mi = mutual_info_regression(X_norm, y_norm, random_state=42)[0]
                te_matrix.iloc[i, j] = max(0, mi)  # Mutual info is non-negative
                
            except Exception:
                continue
    
    return te_matrix


def compute_lead_lag_metrics(returns_df, max_lag=5):
    """
    Compute all lead-lag metrics with proper error handling.
    """
    print("Computing cross-correlation matrix...")
    corr_mat, lag_mat = cross_correlation_matrix(returns_df, max_lag=max_lag)
    
    print("Computing Granger causality...")
    gc_mat = granger_causality_matrix(returns_df, max_lag=max_lag)
    
    print("Computing VAR impulse response...")
    var_results, irf_mat = var_impulse_response(returns_df)
    
    print("Computing transfer entropy...")
    te_mat = transfer_entropy_matrix(returns_df, max_lag=max_lag)
    
    return {
        'correlation': corr_mat,
        'lag': lag_mat,
        'granger': gc_mat,
        'var': var_results,
        'irf': irf_mat,
        'transfer_entropy': te_mat
    }


# FIX: Add a helper function to clean returns before analysis
def clean_returns_for_analysis(returns_df, min_valid_obs=50):
    """
    Remove tickers with insufficient data and align dates.
    """
    # Drop columns with too few non-NaN values
    valid_cols = returns_df.columns[returns_df.count() >= min_valid_obs]
    clean_df = returns_df[valid_cols].copy()
    
    # Drop rows where all values are NaN
    clean_df = clean_df.dropna(how='all')
    
    # Forward fill missing values (carry last observation forward)
    clean_df = clean_df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Cleaned data: {len(clean_df)} rows, {len(clean_df.columns)} tickers")
    print(f"Removed tickers: {set(returns_df.columns) - set(clean_df.columns)}")
    
    return clean_df
