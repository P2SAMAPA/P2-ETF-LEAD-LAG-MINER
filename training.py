"""
Global and Shrinking Window training orchestration.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from lead_lag_engine import (
    cross_correlation_matrix,
    granger_causality_matrix,
    var_impulse_response_leadlag,
    transfer_entropy_matrix,
    lead_lag_consensus,
)
from selector import select_top_etf
from us_calendar import next_trading_day
from push_results import push_daily_result


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    """Compute performance metrics for a given ETF ticker."""
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}

    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    hit_rate = (ret_series > 0).mean()

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
    }


def train_global(universe: str, returns: pd.DataFrame, end_date: str) -> dict:
    """Train using global 80/10/10 split ending at end_date."""
    total_days = len(returns)
    train_end = int(total_days * config.TRAIN_RATIO)
    val_end = train_end + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end]
    val_ret = returns.iloc[train_end:val_end]
    test_ret = returns.iloc[val_end:]

    # Compute lead-lag metrics on training data
    corr, corr_lag = cross_correlation_matrix(train_ret, max_lag=max(config.LAGS))
    gc_pval = granger_causality_matrix(train_ret, max_lag=max(config.LAGS))
    irf_lag = var_impulse_response_leadlag(train_ret, max_lag=max(config.LAGS))
    te = transfer_entropy_matrix(train_ret, lag=1)

    consensus = lead_lag_consensus(corr_lag, gc_pval, irf_lag, te)

    # Select top ETF based on validation set performance
    tickers = [col.replace("_ret", "") for col in returns.columns]
    top_etf = select_top_etf(consensus, val_ret, tickers)

    # Predicted return: average forward return of selected ETF on validation set (annualized)
    col = f"{top_etf}_ret"
    if col in val_ret.columns:
        pred_return = val_ret[col].mean() * config.TRADING_DAYS_PER_YEAR
    else:
        pred_return = None

    # Evaluate on test set
    metrics = evaluate_etf(top_etf, test_ret)

    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_shrinking_window(universe: str, returns: pd.DataFrame) -> dict:
    """Run shrinking window training starting from each year in config."""
    results = []
    tickers = [col.replace("_ret", "") for col in returns.columns]

    for start_year in config.SHRINKING_START_YEARS:
        start_date = f"{start_year}-01-01"
        mask = returns.index >= start_date
        if mask.sum() < config.MIN_TRAIN_DAYS:
            continue
        window_returns = returns.loc[mask]

        total_days = len(window_returns)
        train_end = int(total_days * config.TRAIN_RATIO)
        val_end = train_end + int(total_days * config.VAL_RATIO)

        train_ret = window_returns.iloc[:train_end]
        val_ret = window_returns.iloc[train_end:val_end]
        test_ret = window_returns.iloc[val_end:]

        if len(val_ret) < 20 or len(test_ret) < 20:
            continue

        # Recompute lead-lag metrics using only this window's training data
        corr, corr_lag = cross_correlation_matrix(train_ret, max_lag=max(config.LAGS))
        gc_pval = granger_causality_matrix(train_ret, max_lag=max(config.LAGS))
        irf_lag = var_impulse_response_leadlag(train_ret, max_lag=max(config.LAGS))
        te = transfer_entropy_matrix(train_ret, lag=1)

        consensus = lead_lag_consensus(corr_lag, gc_pval, irf_lag, te)

        top_etf = select_top_etf(consensus, val_ret, tickers)
        metrics = evaluate_etf(top_etf, test_ret)

        # Predicted return for this window (from validation)
        col = f"{top_etf}_ret"
        val_pred_return = val_ret[col].mean() * config.TRADING_DAYS_PER_YEAR if col in val_ret.columns else None

        results.append({
            "window_start": start_date,
            "train_end": train_ret.index[-1].strftime("%Y-%m-%d"),
            "val_end": val_ret.index[-1].strftime("%Y-%m-%d"),
            "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
            "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
            "ticker": top_etf,
            "val_pred_return": val_pred_return,
            "metrics": metrics,
        })

        print(f"Window {start_year}: ETF={top_etf}, Ann Return={metrics.get('ann_return',0)*100:.1f}%")

    if not results:
        return {"ticker": None, "windows": [], "pred_return": None}

    weighted_pick = aggregate_windows(results)

    # Compute predicted return for the weighted pick: average val_pred_return across windows where it was chosen
    pick_returns = [w["val_pred_return"] for w in results if w["ticker"] == weighted_pick and w["val_pred_return"] is not None]
    pred_return = np.mean(pick_returns) if pick_returns else None

    print(f"  Weighted ensemble pick: {weighted_pick}")
    return {
        "ticker": weighted_pick,
        "pred_return": pred_return,
        "windows": results,
    }


def aggregate_windows(windows: list) -> str:
    scores = {}
    for w in windows:
        ticker = w["ticker"]
        ret = w["metrics"].get("ann_return", 0.0)
        sharpe = w["metrics"].get("sharpe", 0.0)
        max_dd = w["metrics"].get("max_dd", -1.0)
        hit_rate = w["metrics"].get("hit_rate", 0.0)

        if ret <= 0:
            weight = 0.0
        else:
            dd_score = 1.0 / (1.0 + abs(max_dd))
            weight = (
                config.WEIGHT_RETURN * ret
                + config.WEIGHT_SHARPE * sharpe
                + config.WEIGHT_HITRATE * hit_rate
                + config.WEIGHT_MAXDD * dd_score
            )
        scores[ticker] = scores.get(ticker, 0.0) + weight

    if not scores:
        return windows[-1]["ticker"] if windows else None
    return max(scores, key=scores.get)


def run_training() -> dict:
    """Main training pipeline for both universes."""
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)

    results = {}
    for universe in ["fi", "equity"]:
        print(f"Processing {universe} universe...")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            print(f"Warning: No returns data for {universe}")
            continue

        # Global training
        global_res = train_global(universe, returns, end_date=df.index[-1].strftime("%Y-%m-%d"))

        # Shrinking window training
        shrinking_res = train_shrinking_window(universe, returns)

        results[universe] = {
            "global": global_res,
            "shrinking": shrinking_res,
        }

    return results


if __name__ == "__main__":
    output = run_training()
    # Push to HF dataset if token exists
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set. Results not pushed.")
        print(json.dumps(output, indent=2, default=str))
