"""Benchmarks to beat: equal-weight and an index fund.

Same rebalance calendar and cost model as the optimiser so the comparison
is fair. If a fancy strategy cannot beat "just split money equally" or
"just buy the index", it is not adding value.
"""

from __future__ import annotations

import pandas as pd

from portopt.backtest import BacktestResult, run_backtest


def equal_weight_backtest(
    prices: pd.DataFrame,
    test_start: str | pd.Timestamp = "2019-01-01",
    lookback: int | None = 252,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
    initial_wealth: float = 1.0,
    min_history: int = 60,
) -> BacktestResult:
    """1/n in every stock, rebalanced back to 1/n on the same schedule.

    Runs through the same engine as the optimiser, so dates, rebalance days
    and costs match exactly. Use the same arguments you gave run_backtest.
    """
    return run_backtest(
        prices,
        strategy="equal_weight",
        test_start=test_start,
        lookback=lookback,
        rebalance=rebalance,
        cost_bps=cost_bps,
        initial_wealth=initial_wealth,
        min_history=min_history,
    )


def index_backtest(
    index_prices: pd.Series | pd.DataFrame,
    dates: pd.DatetimeIndex,
    cost_bps: float = 10.0,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """Buy the index once on dates[0] and hold (SPY for US, ISF.L for UK).

    dates: the wealth index of a strategy, e.g. `result.wealth.index`, so
           both curves start and end on the same days. If the index misses a
           date (different holidays), the last known price is used.
    Pays the same one-off cost as a strategy buying its first portfolio.
    """
    if isinstance(index_prices, pd.DataFrame):
        if index_prices.shape[1] != 1:
            raise ValueError("index_prices must be a single price column.")
        index_prices = index_prices.iloc[:, 0]
    if len(dates) < 2:
        raise ValueError("Need at least two dates.")

    px = index_prices.dropna().sort_index().astype(float)
    px = px.reindex(px.index.union(dates)).ffill().loc[dates]
    if px.isna().any():
        raise ValueError("Index prices do not cover the start of the date range.")
    if (px <= 0).any():
        raise ValueError("Index prices must be strictly positive.")

    wealth = initial_wealth * px / px.iloc[0]
    wealth.iloc[1:] = wealth.iloc[1:] * (1.0 - cost_bps / 10_000.0)  # one-off buying cost
    wealth.name = "index"
    return wealth