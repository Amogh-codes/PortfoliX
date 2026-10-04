"""Streamlit GUI for portopt.

Run from the project root (the folder that contains `portopt/`):
    pip install streamlit
    streamlit run streamlit_app.py

This file only collects inputs and shows results. All the maths lives in
the portopt package.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portopt.backtest import REBALANCE_RULES, BacktestResult, run_backtest
from portopt.benchmarks import equal_weight_backtest, index_backtest
from portopt.data import download_prices
from portopt.frontier import compute_frontier, plot_frontier
from portopt.metrics import compare, drawdown_series
from portopt.optimiser import allocate
from portopt.returns import estimate_mu_cov, simple_returns
from portopt.universe import BENCHMARKS, DEFAULT_UK_TICKERS, DEFAULT_US_TICKERS

STRATEGIES = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]
BACKTEST_STRATEGIES = ["min_variance", "max_sharpe", "risk_parity"]

st.set_page_config(page_title="Portfolio Optimiser", layout="wide")
st.title("Portfolio Optimiser")
st.caption("Research tool, not financial advice. Past performance does not predict future results.")


# ---------- Cached helpers (so the app doesn't redo slow work on every click) ----------
@st.cache_data(show_spinner="Downloading prices...")
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return download_prices(list(tickers), start=start, end=end)


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(
    prices: pd.DataFrame,
    strategy: str,
    test_start: str,
    lookback: int,
    rebalance: str,
    cost_bps: float,
    risk_free: float,
) -> BacktestResult:
    return run_backtest(
        prices,
        strategy,
        test_start=test_start,
        lookback=lookback,
        rebalance=rebalance,
        cost_bps=cost_bps,
        risk_free=risk_free,
    )


@st.cache_data(show_spinner=False)
def cached_equal_weight(
    prices: pd.DataFrame,
    test_start: str,
    lookback: int,
    rebalance: str,
    cost_bps: float,
) -> BacktestResult:
    return equal_weight_backtest(
        prices,
        test_start=test_start,
        lookback=lookback,
        rebalance=rebalance,
        cost_bps=cost_bps,
    )


# ---------- Sidebar: inputs ----------
with st.sidebar:
    st.header("Data and strategy")
    market = st.selectbox("Market", ["US", "UK"])
    defaults = DEFAULT_US_TICKERS if market == "US" else DEFAULT_UK_TICKERS
    ticker_text = st.text_input("Tickers (comma separated)", ", ".join(defaults))
    start = st.date_input("Start date", pd.Timestamp("2015-01-01"))
    end = st.date_input("End date", pd.Timestamp("2019-12-31"))
    risk_free = st.number_input(
        "Risk-free rate (annual, e.g. 0.02 = 2%)", value=0.0, step=0.005, format="%.3f"
    )
    strategy = st.selectbox("Strategy", STRATEGIES, index=2)

    st.header("Backtest")
    test_start = st.date_input("Test start date", pd.Timestamp("2019-01-01"))
    rebalance = st.selectbox("Rebalance", list(REBALANCE_RULES), index=3)
    cost_bps = st.number_input("Trading cost (basis points)", value=10.0, min_value=0.0, step=1.0)
    lookback = st.number_input("Lookback (trading days)", value=252, min_value=60, max_value=2000, step=21)
    bt_strategies = st.multiselect("Strategies to backtest", BACKTEST_STRATEGIES, default=BACKTEST_STRATEGIES)
    show_index = st.checkbox(f"Compare with index ({BENCHMARKS[market]})", value=False)

    run = st.button("Run", type="primary")

# Remember that Run was pressed, so changing a setting later re-runs the page
# instead of sending you back to the "press Run" message.
if run:
    st.session_state["ran"] = True
if not st.session_state.get("ran"):
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
    st.caption("In-sample: estimated on the whole date range above, so this looks better than real life.")
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
    st.subheader("Backtest: train on the past, test on data the strategy has not seen")
    st.caption(
        "At each rebalance, weights are estimated from the lookback window BEFORE that day only. "
        "Trading costs are charged on every rebalance, including the first purchase."
    )

    if not bt_strategies:
        st.warning("Pick at least one strategy to backtest in the sidebar.")
        st.stop()
    if not (pd.Timestamp(start) < pd.Timestamp(test_start) < pd.Timestamp(end)):
        st.error("Test start date must be between the start and end dates.")
        st.stop()

    ts, lb, cost = str(test_start), int(lookback), float(cost_bps)
    try:
        results = {
            s: cached_backtest(prices, s, ts, lb, rebalance, cost, risk_free) for s in bt_strategies
        }
        bench = cached_equal_weight(prices, ts, lb, rebalance, cost)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

    all_results = {**results, "equal_weight (benchmark)": bench}
    curves = {name: res.wealth for name, res in all_results.items()}

    if show_index:
        ticker = BENCHMARKS[market]
        try:
            index_prices = load_prices((ticker,), str(start), str(end))
            curves[f"index ({ticker})"] = index_backtest(index_prices, bench.wealth.index, cost_bps=cost)
        except Exception as exc:
            st.warning(f"Could not load index {ticker}: {exc}")

    st.markdown("**Growth of 1 unit of money**")
    st.line_chart(pd.DataFrame(curves))

    st.markdown("**Performance (test period only)**")
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

    st.markdown("**Trading activity**")
    activity = pd.DataFrame(
        {
            name: {
                "Rebalances": len(res.weights),
                "Avg turnover per rebalance": float(res.turnover.mean()),
                "Total costs (% of starting wealth)": res.total_costs,
            }
            for name, res in all_results.items()
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

    st.markdown("**Weights over time**")
    chosen = st.selectbox("Show weights for", list(all_results))
    chosen_weights = all_results[chosen].weights
    st.area_chart(chosen_weights)
    with st.expander("Weights table"):
        st.dataframe(chosen_weights.style.format("{:.1%}"))