"""
Core lead-lag analysis methods.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from pyinform.transferentropy import transfer_entropy
import warnings

warnings.filterwarnings("ignore")
import config


def cross_correlation_matrix(returns: pd.DataFrame, max_lag: int = 10) -> tuple:
    """
    Compute maximum absolute cross-correlation and corresponding lag for all pairs.
    Returns:
        corr_matrix: DataFrame of max correlation values
        lag_matrix: DataFrame of lag (positive means row leads column)
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

            series_i = returns[col_i].values
            series_j = returns[col_j].values
            max_corr = 0.0
            best_lag = 0

            # Check i leading j (i at t-lag, j at t)
            for lag in range(1, max_lag + 1):
                if len(series_i) <= lag:
                    continue
                corr = np.corrcoef(series_i[:-lag], series_j[lag:])[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag

            # Check j leading i (j at t-lag, i at t)
            for lag in range(1, max_lag + 1):
                if len(series_j) <= lag:
                    continue
                corr = np.corrcoef(series_j[:-lag], series_i[lag:])[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = -lag  # negative lag indicates column leads row

            corr_matrix.iloc[i, j] = max_corr
            lag_matrix.iloc[i, j] = best_lag

    return corr_matrix, lag_matrix


def granger_causality_matrix(returns: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """
    Test Granger causality for all pairs at each lag.
    Returns DataFrame of min p-value across lags for each pair (row causes column).
    """
    n = len(returns.columns)
    tickers = [col.replace("_ret", "") for col in returns.columns]
    pval_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    for i, col_i in enumerate(returns.columns):
        for j, col_j in enumerate(returns.columns):
            if i == j:
                pval_matrix.iloc[i, j] = 1.0
                continue

            data = returns[[col_j, col_i]].dropna()
            if len(data) < 50:
                pval_matrix.iloc[i, j] = np.nan
                continue

            try:
                gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                min_p = 1.0
                for lag in range(1, max_lag + 1):
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
    """
    n = len(returns.columns)
    tickers = [col.replace("_ret", "") for col in returns.columns]
    irf_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=int)

    try:
        model = VAR(returns)
        results = model.fit(maxlags=min(max_lag, 5), ic="aic")
        irf = results.irf(periods=max_lag)
        orth_irf = irf.orth_irfs

        for i, shock_var in enumerate(tickers):
            for j, resp_var in enumerate(tickers):
                if i == j:
                    irf_matrix.iloc[i, j] = 0
                    continue
                response = orth_irf[:, j, i]  # response of j to shock in i
                peak_lag = np.argmax(np.abs(response))
                irf_matrix.iloc[i, j] = peak_lag
    except Exception as e:
        print(f"VAR IRF failed: {e}")
        irf_matrix[:] = 0

    return irf_matrix


def transfer_entropy_matrix(returns: pd.DataFrame, lag: int = 1, n_shuffles: int = 100) -> pd.DataFrame:
    """
    Compute Effective Transfer Entropy (ETE) for all pairs at given lag.
    Returns DataFrame of TE values (row -> column).
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
                te = transfer_entropy(src_aligned, tgt_aligned, lag)
                # Effective TE via shuffling
                shuffled_tes = []
                for _ in range(n_shuffles):
                    np.random.shuffle(src_aligned)
                    shuffled_tes.append(transfer_entropy(src_aligned, tgt_aligned, lag))
                ete = te - np.mean(shuffled_tes)
                te_matrix.iloc[i, j] = max(ete, 0.0)
            except:
                te_matrix.iloc[i, j] = 0.0

    return te_matrix


def lead_lag_consensus(corr_lag: pd.DataFrame, gc_pval: pd.DataFrame,
                       irf_lag: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple methods into a consensus lead-lag score.
    Returns DataFrame with score (higher means row leads column).
    """
    tickers = corr_lag.index
    score = pd.DataFrame(0.0, index=tickers, columns=tickers)

    # Cross-correlation: sign of lag indicates direction
    for i in tickers:
        for j in tickers:
            if i == j:
                continue
            lag_val = corr_lag.loc[i, j]
            if lag_val > 0:
                score.loc[i, j] += 1.0
            elif lag_val < 0:
                score.loc[j, i] += 1.0

    # Granger causality: lower p-value => stronger evidence
    for i in tickers:
        for j in tickers:
            if i == j:
                continue
            p = gc_pval.loc[i, j]
            if not np.isnan(p) and p < 0.05:
                score.loc[i, j] += 1.0

    # VAR IRF: shorter lag => stronger immediate impact
    for i in tickers:
        for j in tickers:
            if i == j:
                continue
            lag_val = irf_lag.loc[i, j]
            if lag_val > 0:
                score.loc[i, j] += 1.0 / (lag_val + 1)
            elif lag_val < 0:
                score.loc[j, i] += 1.0 / (abs(lag_val) + 1)

    # Transfer Entropy: higher TE => stronger information flow
    te_max = te.values.max()
    if te_max > 0:
        for i in tickers:
            for j in tickers:
                if i == j:
                    continue
                score.loc[i, j] += te.loc[i, j] / te_max

    return score
