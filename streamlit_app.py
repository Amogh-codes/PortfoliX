"""Streamlit GUI for portopt.

Run from the project root (the folder that contains `portopt/`):
    pip install streamlit
    streamlit run app.py

This file only collects inputs and shows results. All the maths lives in
the portopt package.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portopt.data import download_prices
from portopt.frontier import compute_frontier, plot_frontier
from portopt.optimiser import allocate
from portopt.returns import estimate_mu_cov, simple_returns
from portopt.universe import DEFAULT_UK_TICKERS, DEFAULT_US_TICKERS

STRATEGIES = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]

st.set_page_config(page_title="Portfolio Optimiser", layout="wide")
st.title("Portfolio Optimiser")
st.caption("Research tool: weights are estimated on past data (in-sample). Not financial advice.")


# ---------- Cached helpers (so the app doesn't re-download on every click) ----------
@st.cache_data(show_spinner="Downloading prices...")
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return download_prices(list(tickers), start=start, end=end)


# ---------- Sidebar: inputs ----------
with st.sidebar:
    st.header("Settings")
    market = st.selectbox("Market", ["US", "UK"])
    defaults = DEFAULT_US_TICKERS if market == "US" else DEFAULT_UK_TICKERS
    ticker_text = st.text_input("Tickers (comma separated)", ", ".join(defaults))
    start = st.date_input("Start date", pd.Timestamp("2015-01-01"))
    end = st.date_input("End date", pd.Timestamp("2019-12-31"))
    risk_free = st.number_input("Risk-free rate (annual, e.g. 0.02 = 2%)", value=0.0, step=0.005, format="%.3f")
    strategy = st.selectbox("Strategy", STRATEGIES, index=2)
    run = st.button("Run", type="primary")

if not run:
    st.info("Pick your settings in the sidebar and press **Run**.")
    st.stop()

# ---------- Run the pipeline ----------
tickers = tuple(t.strip() for t in ticker_text.split(",") if t.strip())

try:
    prices = load_prices(tickers, str(start), str(end))
    returns = simple_returns(prices)
    mu, cov = estimate_mu_cov(returns)
    weights = allocate(strategy, mu, cov, risk_free=risk_free)
    frontier = compute_frontier(mu, cov, risk_free=risk_free)
except Exception as exc:  # show errors in the UI instead of a crash
    st.error(f"Something went wrong: {exc}")
    st.stop()

# ---------- Results ----------
tab_weights, tab_frontier, tab_data, tab_backtest = st.tabs(
    ["Weights", "Efficient frontier", "Data", "Backtest"]
)

with tab_weights:
    st.subheader(f"Weights: {strategy}")
    col1, col2 = st.columns(2)
    col1.dataframe(weights.rename("weight").to_frame().style.format("{:.2%}"))
    col2.bar_chart(weights)

    vol = float((weights.values @ cov.values @ weights.values) ** 0.5)
    ret = float(weights @ mu)
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected return (annual)", f"{ret:.2%}")
    m2.metric("Volatility (annual)", f"{vol:.2%}")
    m3.metric("Sharpe", f"{(ret - risk_free) / vol:.2f}" if vol > 0 else "n/a")

with tab_frontier:
    st.pyplot(plot_frontier(frontier))

with tab_data:
    st.write(f"{len(prices)} price rows, {len(returns)} return rows")
    st.line_chart(prices / prices.iloc[0])  # all stocks start at 1.0 so they are comparable
    st.dataframe(prices.tail(20))

with tab_backtest:
    st.info("Coming in week 2: rebalancing, costs, and a train/test split.")