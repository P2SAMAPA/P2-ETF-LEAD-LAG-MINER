"""
Core lead-lag analysis methods.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings("ignore")
import config

# Try to import pyinform, fallback to custom implementation if not available
try:
    from pyinform.transferentropy import transfer_entropy
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    print("Warning: pyinform not available. Using fallback transfer entropy implementation.")


def cross_correlation_matrix(returns: pd.DataFrame, max_lag: int = 10) -> tuple:
    """
    Compute maximum absolute cross-correlation and corresponding lag for all pairs.
    Returns:
        corr_matrix: DataFrame of max correlation values
        lag_matrix: DataFrame of lag (positive means row leads column)
    FIXED: Handles different series lengths and NaN values.
    """
    n = len(returns.columns)
    tickers = [col.replace("_ret", "") for col in returns.columns]
    corr_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    lag_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=int)

    for i, col_i in enumerate(returns.columns):
        for j, col_j in enumerate(returns.columns):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
                lag_matrix.iloc[i, j] = 0
                continue

            # FIX: Get clean series, drop NaNs
            series_i = returns[col_i].dropna().values
            series_j = returns[col_j].dropna().values
            
            # FIX: Check for minimum length
            if len(series_i) < 50 or len(series_j) < 50:
                corr_matrix.iloc[i, j] = np.nan
                lag_matrix.iloc[i, j] = 0
                continue

            max_corr = 0.0
            best_lag = 0

            # FIX: Align lengths for each lag calculation
            # Check i leading j (i at t-lag, j at t)
            for lag in range(1, max_lag + 1):
                if len(series_i) <= lag or len(series_j) <= lag:
                    continue
                # FIX: Use min length to avoid size mismatch
                min_len = min(len(series_i) - lag, len(series_j) - lag)
                if min_len < 10:
                    continue
                seg_i = series_i[:min_len]
                seg_j = series_j[lag:lag+min_len]
                try:
                    corr = np.corrcoef(seg_i, seg_j)[0, 1]
                    if not np.isnan(corr) and abs(corr) > abs(max_corr):
                        max_corr = corr
                        best_lag = lag
                except:
                    continue

            # Check j leading i (j at t-lag, i at t)
            for lag in range(1, max_lag + 1):
                if len(series_j) <= lag or len(series_i) <= lag:
                    continue
                min_len = min(len(series_j) - lag, len(series_i) - lag)
                if min_len < 10:
                    continue
                seg_j = series_j[:min_len]
                seg_i = series_i[lag:lag+min_len]
                try:
                    corr = np.corrcoef(seg_j, seg_i)[0, 1]
                    if not np.isnan(corr) and abs(corr) > abs(max_corr):
                        max_corr = corr
                        best_lag = -lag  # negative lag indicates column leads row
                except:
                    continue

            corr_matrix.iloc[i, j] = max_corr
            lag_matrix.iloc[i, j] = best_lag

    return corr_matrix, lag_matrix


def granger_causality_matrix(returns: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """
    Test Granger causality for all pairs at each lag.
    Returns DataFrame of min p-value across lags for each pair (row causes column).
    FIXED: Handles tickers with insufficient data.
    """
    n = len(returns.columns)
    tickers = [col.replace("_ret", "") for col in returns.columns]
    pval_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    for i, col_i in enumerate(returns.columns):
        for j, col_j in enumerate(returns.columns):
            if i == j:
                pval_matrix.iloc[i, j] = 1.0
                continue

            # FIX: Get overlapping data only
            data = returns[[col_j, col_i]].dropna()
            if len(data) < 50:
                pval_matrix.iloc[i, j] = np.nan
                continue

            # FIX: Adjust max_lag based on data length
            adjusted_lag = min(max_lag, len(data) // 10)
            if adjusted_lag < 1:
                pval_matrix.iloc[i, j] = np.nan
                continue

            try:
                gc_res = grangercausalitytests(data, maxlag=adjusted_lag, verbose=False)
                min_p = 1.0
                for lag in range(1, adjusted_lag + 1):
                    pval = gc_res[lag][0]["ssr_ftest"][1]
                    if pval < min_p:
                        min_p = pval
                pval_matrix.iloc[i, j] = min_p
            except:
                pval_matrix.iloc[i, j] = np.nan

    return pval_matrix


def var_impulse_response_leadlag(returns: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """
    Fit VAR and compute orthogonalized impulse response peak lag for each pair.
    Returns DataFrame of lag (positive means row shock affects column).
    FIXED: Handles missing data and short histories.
    """
    n = len(returns.columns)
    tickers = [col.replace("_ret", "") for col in returns.columns]
    irf_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=int)

    # FIX: Drop any columns with all NaN and fill remaining
    clean_returns = returns.dropna(axis=1, how='all')
    if clean_returns.empty:
        irf_matrix[:] = 0
        return irf_matrix
    
    clean_returns = clean_returns.fillna(clean_returns.median())
    
    # FIX: Check if we have enough data
    if len(clean_returns) < 50:
        irf_matrix[:] = 0
        return irf_matrix

    try:
        model = VAR(clean_returns)
        results = model.fit(maxlags=min(max_lag, len(clean_returns)//10, 5), ic="aic")
        irf = results.irf(periods=min(max_lag, len(clean_returns)//10))
        orth_irf = irf.orth_irfs

        for i, shock_var in enumerate(tickers):
            for j, resp_var in enumerate(tickers):
                if i == j:
                    irf_matrix.iloc[i, j] = 0
                    continue
                # FIX: Check dimensions
                if i < orth_irf.shape[2] and j < orth_irf.shape[1]:
                    response = orth_irf[:, j, i]
                    peak_lag = np.argmax(np.abs(response))
                    irf_matrix.iloc[i, j] = peak_lag
                else:
                    irf_matrix.iloc[i, j] = 0
    except Exception as e:
        print(f"VAR IRF failed: {e}")
        irf_matrix[:] = 0

    return irf_matrix


def transfer_entropy_fallback(source, target, lag=1):
    """
    Fallback transfer entropy approximation when pyinform is not available.
    Uses correlation-based approximation.
    """
    if len(source) != len(target):
        min_len = min(len(source), len(target))
        source = source[:min_len]
        target = target[:min_len]
    
    if len(source) < 50:
        return 0.0
    
    # Compute correlation at the specified lag as proxy
    try:
        corr = np.corrcoef(source[:-lag], target[lag:])[0, 1]
        if np.isnan(corr):
            return 0.0
        # Convert correlation to approximate mutual information
        # Using -0.5 * log(1 - r^2) which is the Gaussian mutual information
        r_squared = corr ** 2
        if r_squared >= 1.0:
            return 0.0
        return -0.5 * np.log(1 - r_squared)
    except:
        return 0.0


def transfer_entropy_matrix(returns: pd.DataFrame, lag: int = 1, n_shuffles: int = 100) -> pd.DataFrame:
    """
    Compute Effective Transfer Entropy (ETE) for all pairs at given lag.
    Returns DataFrame of TE values (row -> column).
    FIXED: Handles missing pyinform and data issues.
    """
    n = len(returns.columns)
    tickers = [col.replace("_ret", "") for col in returns.columns]
    te_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    for i, source_col in enumerate(returns.columns):
        source = returns[source_col].dropna().values
        for j, target_col in enumerate(returns.columns):
            if i == j:
                te_matrix.iloc[i, j] = 0.0
                continue
            target = returns[target_col].dropna().values
            
            # Align lengths
            min_len = min(len(source), len(target))
            if min_len < 50:
                te_matrix.iloc[i, j] = np.nan
                continue
            src_aligned = source[:min_len]
            tgt_aligned = target[:min_len]

            try:
                if PYINFORM_AVAILABLE:
                    te = transfer_entropy(src_aligned, tgt_aligned, lag)
                    # Effective TE via shuffling
                    shuffled_tes = []
                    for _ in range(min(n_shuffles, 30)):  # Fewer shuffles for speed
                        shuffled = np.random.permutation(src_aligned)
                        shuffled_tes.append(transfer_entropy(shuffled, tgt_aligned, lag))
                    ete = te - np.mean(shuffled_tes)
                    te_matrix.iloc[i, j] = max(ete, 0.0)
                else:
                    # Use fallback implementation
                    te_matrix.iloc[i, j] = transfer_entropy_fallback(src_aligned, tgt_aligned, lag)
            except Exception as e:
                # Fallback on error
                te_matrix.iloc[i, j] = transfer_entropy_fallback(src_aligned, tgt_aligned, lag)

    return te_matrix


def lead_lag_consensus(corr_lag: pd.DataFrame, gc_pval: pd.DataFrame,
                       irf_lag: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple methods into a consensus lead-lag score.
    Returns DataFrame with score (higher means row leads column).
    FIXED: Properly iterates over DataFrame cells instead of using Series.
    """
    tickers = corr_lag.index
    score = pd.DataFrame(0.0, index=tickers, columns=tickers)

    # Cross-correlation: sign of lag indicates direction
    for i in tickers:
        for j in tickers:
            if i == j:
                continue
            lag_val = corr_lag.loc[i, j]
            # FIX: Check if it's a scalar (not a Series)
            if isinstance(lag_val, pd.Series):
                lag_val = lag_val.iloc[0] if len(lag_val) > 0 else np.nan
            if not np.isnan(lag_val) and lag_val > 0:
                score.loc[i, j] += 1.0
            elif not np.isnan(lag_val) and lag_val < 0:
                score.loc[j, i] += 1.0

    # Granger causality: lower p-value => stronger evidence
    for i in tickers:
        for j in tickers:
            if i == j:
                continue
            p = gc_pval.loc[i, j]
            if isinstance(p, pd.Series):
                p = p.iloc[0] if len(p) > 0 else np.nan
            if not np.isnan(p) and p < 0.05:
                score.loc[i, j] += 1.0

    # VAR IRF: shorter lag => stronger immediate impact
    for i in tickers:
        for j in tickers:
            if i == j:
                continue
            lag_val = irf_lag.loc[i, j]
            if isinstance(lag_val, pd.Series):
                lag_val = lag_val.iloc[0] if len(lag_val) > 0 else np.nan
            if not np.isnan(lag_val) and lag_val > 0:
                score.loc[i, j] += 1.0 / (lag_val + 1)
            elif not np.isnan(lag_val) and lag_val < 0:
                score.loc[j, i] += 1.0 / (abs(lag_val) + 1)

    # Transfer Entropy: higher TE => stronger information flow
    te_max = te.values.max()
    if not np.isnan(te_max) and te_max > 0:
        for i in tickers:
            for j in tickers:
                if i == j:
                    continue
                te_val = te.loc[i, j]
                if isinstance(te_val, pd.Series):
                    te_val = te_val.iloc[0] if len(te_val) > 0 else np.nan
                if not np.isnan(te_val):
                    score.loc[i, j] += te_val / te_max

    return score
