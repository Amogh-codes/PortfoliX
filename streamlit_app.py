"""Streamlit GUI for portopt.
 
Run from the project root (the folder that contains `portopt/`):
    pip install streamlit yfinance
    streamlit run streamlit_app.py
 
Two separate windows of time:
- OPTIMISATION data (sidebar start/end): used to fit the weights.
- BACKTEST (Backtest tab): starts on or after the optimisation end date and
  tests those fixed weights on data they have never seen.
 
This file only collects inputs and shows results. All the maths lives in
the portopt package.
"""
 
from __future__ import annotations
 
import datetime as dt
 
import pandas as pd
import streamlit as st
 
from portopt.backtest import REBALANCE_RULES, BacktestResult, run_backtest
from portopt.data import download_prices
from portopt.frontier import compute_frontier, plot_frontier
from portopt.metrics import compare, drawdown_series
from portopt.optimiser import allocate
from portopt.returns import estimate_mu_cov, simple_returns
from portopt.universe import DEFAULT_UK_TICKERS, DEFAULT_US_TICKERS
 
STRATEGIES = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]
PERIODS = {"1 month": 1, "3 months": 3, "6 months": 6, "1 year": 12, "2 years": 24, "3 years": 36}
COST_BPS = 10.0  # trading cost per rebalance: 10 bps = 0.10% of the amount traded
MIN_OPT_RETURNS = 60  # need at least this many days of optimisation data
ONE_DAY = pd.Timedelta(days=1)
 
st.set_page_config(page_title="Portfolio Optimiser", layout="wide")
st.title("Portfolio Optimiser")
st.caption("Research tool, not financial advice. Past performance does not predict future results.")
 
 
def iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")
 
 
# ---------- Cached helpers (so the app doesn't redo slow work on every click) ----------
@st.cache_data(show_spinner="Downloading prices...")
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return download_prices(list(tickers), start=start, end=end)
 
 
@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(
    bt_prices: pd.DataFrame,
    weights: pd.Series,
    rebalance: str,
    cost_bps: float,
) -> BacktestResult:
    return run_backtest(
        bt_prices,
        test_start=bt_prices.index[0],
        fixed_weights=weights,
        rebalance=rebalance,
        cost_bps=cost_bps,
    )
 
 
# ---------- Sidebar: optimisation data ----------
with st.sidebar:
    st.header("Optimisation data")
    market = st.selectbox("Market", ["US", "UK"])
    defaults = DEFAULT_US_TICKERS if market == "US" else DEFAULT_UK_TICKERS
    ticker_text = st.text_input("Tickers (comma separated)", ", ".join(defaults))
    start = st.date_input("Start date", dt.date(2015, 1, 1), min_value=dt.date(1990, 1, 1))
    end = st.date_input(
        "End date",
        dt.date(2019, 12, 31),
        min_value=dt.date(1990, 1, 1),
        max_value=dt.date.today(),
        help="Weights are fitted on data up to and including this date. Backtests can start no earlier.",
    )
    risk_free = st.number_input(
        "Risk-free rate (annual, e.g. 0.02 = 2%)", value=0.0, step=0.005, format="%.3f"
    )
    strategy = st.selectbox("Strategy (for Weights tab)", STRATEGIES, index=2)
    run = st.button("Run", type="primary")
 
# Remember that Run was pressed, so changing a setting later re-runs the page
# instead of sending you back to the "press Run" message.
if run:
    st.session_state["ran"] = True
if not st.session_state.get("ran"):
    st.info("Pick your settings in the sidebar and press **Run**.")
    st.stop()
 
# ---------- Fit on the optimisation data ----------
tickers = tuple(t.strip() for t in ticker_text.split(",") if t.strip())
opt_start, opt_end = pd.Timestamp(start), pd.Timestamp(end)
 
if not opt_start < opt_end:
    st.error("Start date must be before the end date.")
    st.stop()
 
try:
    prices = load_prices(tickers, iso(opt_start), iso(opt_end + ONE_DAY))  # data source end is exclusive
    opt_prices = prices.loc[prices.index <= opt_end]  # never anything after the end date
    returns = simple_returns(opt_prices)
    if len(returns) < MIN_OPT_RETURNS:
        raise ValueError(f"Only {len(returns)} days of optimisation data; need at least {MIN_OPT_RETURNS}.")
    mu, cov = estimate_mu_cov(returns)
    weights = allocate(strategy, mu, cov, risk_free=risk_free)
    frontier = compute_frontier(mu, cov, risk_free=risk_free)
except Exception as exc:  # show errors in the UI instead of a crash
    st.error(f"Something went wrong: {exc}")
    st.stop()
 
st.info(
    f"**Optimisation data:** {opt_prices.index[0]:%Y-%m-%d} to {opt_prices.index[-1]:%Y-%m-%d} "
    f"({len(opt_prices)} days). Weights and the frontier use only this. "
    f"Backtests can start on or after {opt_end:%Y-%m-%d}."
)
 
# ---------- Results ----------
tab_weights, tab_frontier, tab_data, tab_backtest = st.tabs(
    ["Weights", "Efficient frontier", "Data", "Backtest"]
)
 
with tab_weights:
    st.subheader(f"Weights: {strategy}")
    st.caption("In-sample: fitted on the optimisation data, so it looks better than real life. Judge it in the Backtest tab.")
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
    st.write(f"{len(opt_prices)} optimisation price rows, {len(returns)} return rows")
    st.line_chart(opt_prices / opt_prices.iloc[0])  # all stocks start at 1.0 so they are comparable
    st.dataframe(opt_prices.tail(20))
 
with tab_backtest:
    st.subheader("Backtest")
    st.caption(
        f"Each strategy's weights are fitted on the optimisation data (up to {opt_end:%Y-%m-%d}) and "
        f"held fixed. You buy at the close of the start date; the first return is earned the next trading day. "
        f"Trading cost: {COST_BPS:.0f} bps per trade, including the first purchase."
    )
 
    st.markdown("**Strategies**")
    toggle_cols = st.columns(len(STRATEGIES))
    selected = [
        s for s, col in zip(STRATEGIES, toggle_cols) if col.toggle(s, value=True, key=f"bt_{s}")
    ]
 
    c1, c2, c3 = st.columns(3)
    bt_start = c1.date_input(
        "Backtest start date",
        value=end,
        min_value=end,  # cannot go back before the end of the optimisation data
        max_value=dt.date.today(),
        help="Earliest allowed date = the end date of the optimisation data.",
    )
    period_label = c2.selectbox("Period", list(PERIODS), index=3)
    rebalance = c3.selectbox(
        "Rebalance", list(REBALANCE_RULES), index=3, help="How often to trade back to the target weights."
    )
 
    if not selected:
        st.warning("Switch on at least one strategy.")
        st.stop()
 
    bt_from = pd.Timestamp(bt_start)
    if bt_from < opt_end:  # safety net; the date picker already blocks this
        st.error(f"Backtest start cannot be before the optimisation end date ({opt_end:%Y-%m-%d}).")
        st.stop()
    bt_to = bt_from + pd.DateOffset(months=PERIODS[period_label])
 
    try:
        bt_all = load_prices(tickers, iso(bt_from), iso(bt_to + ONE_DAY))
        bt_prices = bt_all.loc[(bt_all.index >= bt_from) & (bt_all.index <= bt_to)]
        if len(bt_prices) < 3:
            raise ValueError("Not enough price data in that window. Try an earlier start or a longer period.")
        fitted = {s: allocate(s, mu, cov, risk_free=risk_free) for s in selected}
        results = {s: cached_backtest(bt_prices, w, rebalance, COST_BPS) for s, w in fitted.items()}
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()
 
    first, last = bt_prices.index[0], bt_prices.index[-1]
    st.caption(f"Backtest window: {first:%Y-%m-%d} to {last:%Y-%m-%d} ({len(bt_prices) - 1} trading days of returns).")
    if last < bt_to - pd.Timedelta(days=7):
        st.warning(f"Price data only goes up to {last:%Y-%m-%d}, so this covers less than the period you picked.")
    if PERIODS[period_label] < 6:
        st.caption("Short periods give very noisy annualised numbers. Treat them with care.")
 
    curves = {name: res.wealth for name, res in results.items()}
 
    st.markdown("**Growth of 1 unit of money**")
    st.line_chart(pd.DataFrame(curves))
 
    st.markdown("**Performance**")
    table = compare(curves, risk_free=risk_free).rename(
        columns={
            "total_return": "Total return",
            "annual_return": "Annual return",
            "annual_volatility": "Volatility",
            "sharpe": "Sharpe",
            "max_drawdown": "Max drawdown",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Total return": "{:.1%}",
                "Annual return": "{:.1%}",
                "Volatility": "{:.1%}",
                "Sharpe": "{:.2f}",
                "Max drawdown": "{:.1%}",
            }
        )
    )
 
    st.markdown("**Drawdown (how far below the previous peak)**")
    st.line_chart(pd.DataFrame({name: drawdown_series(w) for name, w in curves.items()}))
 
    st.markdown("**Weights held and trading costs**")
    st.dataframe(pd.DataFrame(fitted).style.format("{:.1%}"))
    activity = pd.DataFrame(
        {
            name: {
                "Rebalances": len(res.weights),
                "Avg turnover per rebalance": float(res.turnover.mean()),
                "Total costs (% of starting wealth)": res.total_costs,
            }
            for name, res in results.items()
        }
    ).T
    st.dataframe(
        activity.style.format(
            {
                "Rebalances": "{:.0f}",
                "Avg turnover per rebalance": "{:.0%}",
                "Total costs (% of starting wealth)": "{:.2%}",
            }
        )
    )
 
