# P2 ETF Lead-Lag Miner

Quantitative engine for detecting temporal asymmetry (lead-lag relationships) among ETFs using cross-correlation, Granger causality, VAR impulse response, and Transfer Entropy.

## Features
- Two separate universes: FI/Commodities and Equity
- Global training (80/10/10 split) and Shrinking Window training
- Weighted ETF selection for next trading day
- Clean Streamlit UI matching professional design

## Setup
```bash
pip install -r requirements.txt
