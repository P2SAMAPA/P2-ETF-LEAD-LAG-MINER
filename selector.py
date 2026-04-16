"""
ETF selection logic for lead-lag consensus.
"""
import pandas as pd
import numpy as np
import config


def select_top_etf(consensus: pd.DataFrame, val_returns: pd.DataFrame, tickers: list) -> str:
    """
    Given lead-lag consensus matrix and validation returns,
    select the ETF predicted to have highest next-day return.
    Simplified: choose ETF with highest average incoming lead score.
    """
    incoming_scores = consensus.sum(axis=0)  # sum of columns: how much others lead this ETF
    if incoming_scores.empty:
        return tickers[0] if tickers else None
    best_ticker = incoming_scores.idxmax()
    return best_ticker
