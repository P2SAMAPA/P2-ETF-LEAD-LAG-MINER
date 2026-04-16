"""
Streamlit UI for Lead-Lag-Miner.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import config
from push_results import load_latest_result
from us_calendar import next_trading_day, is_trading_day

st.set_page_config(page_title="Lead-Lag Miner", layout="wide")

# Custom CSS for light shading and clean style
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .ticker-large {
        font-size: 4rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .conviction {
        font-size: 1.5rem;
        color: #2e7d32;
        margin: 0;
    }
    .meta-text {
        color: #6c757d;
        font-size: 0.9rem;
    }
    .metric-table {
        width: 100%;
        border-collapse: collapse;
    }
    .metric-table th {
        text-align: left;
        padding: 0.3rem 0;
        font-weight: 600;
        border-bottom: 1px solid #dee2e6;
    }
    .metric-table td {
        padding: 0.3rem 0;
        border-bottom: 1px solid #f1f3f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">Lead-Lag Miner ETF Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Cross-Correlation · Granger Causality · VAR IRF · Transfer Entropy</div>',
    unsafe_allow_html=True,
)

# Tabs
tab_fi, tab_eq = st.tabs(["Option A — Fixed Income / Commodities", "Option B — Equity Sectors"])

# Load latest results
results = load_latest_result()

# Helper to display a card
def display_card(universe: str, mode: str, data: dict):
    if not data:
        st.info("Waiting for training output...")
        return

    ticker = data.get("ticker")
    if not ticker:
        st.info("Waiting for training output...")
        return

    # Next trading day
    next_day = next_trading_day(datetime.utcnow())
    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f'<div class="ticker-large">{ticker}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conviction">100.0% conviction</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="meta-text">Signal for {next_day.strftime("%Y-%m-%d")} · Generated {gen_time}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="meta-text"><strong>Source:</strong> {mode}</div>', unsafe_allow_html=True)

        with col2:
            # Show 2nd/3rd place if available
            st.markdown('<div class="meta-text">2nd: — 0.0%</div>', unsafe_allow_html=True)
            st.markdown('<div class="meta-text">3rd: — 0.0%</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Metrics table
        metrics = data.get("metrics", {})
        if metrics:
            if mode == "Global":
                st.markdown(f'**FIXED SPLIT (80/10/10)**  ')
                st.markdown(f'Test: {data.get("test_start", "")} → {data.get("test_end", "")}')
            else:
                # Show latest window info
                windows = data.get("windows", [])
                if windows:
                    last_win = windows[-1]
                    st.markdown(f'**SHRINKING WINDOW**  ')
                    st.markdown(f'Window: {last_win["window_start"]} → {last_win["val_end"]} · OOS: {last_win["test_start"]} → {last_win["test_end"]}')

            # Table
            metric_df = pd.DataFrame([
                ["ANN RETURN", f"{metrics.get('ann_return', 0)*100:.1f}%"],
                ["ANN VOL", f"{metrics.get('ann_vol', 0)*100:.1f}%"],
                ["SHARPE", f"{metrics.get('sharpe', 0):.2f}"],
                ["MAX DD (PEAK→TROUGH)", f"{metrics.get('max_dd', 0)*100:.1f}%"],
                ["HIT RATE", f"{metrics.get('hit_rate', 0)*100:.1f}%"],
            ], columns=["Metric", "Value"])
            st.table(metric_df)

        st.markdown('</div>', unsafe_allow_html=True)


with tab_fi:
    st.subheader("FI / Commodities")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Global Training")
        global_data = results.get("fi", {}).get("global", {})
        display_card("fi", "Global", global_data)
    with col2:
        st.markdown("### Shrinking Window")
        shrinking_data = results.get("fi", {}).get("shrinking", {})
        display_card("fi", "Shrinking Window", shrinking_data)

with tab_eq:
    st.subheader("Equity")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Global Training")
        global_data = results.get("equity", {}).get("global", {})
        display_card("equity", "Global", global_data)
    with col2:
        st.markdown("### Shrinking Window")
        shrinking_data = results.get("equity", {}).get("shrinking", {})
        display_card("equity", "Shrinking Window", shrinking_data)
