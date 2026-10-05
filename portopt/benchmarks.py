"""Benchmark strategies for comparison.

Use the same rebalance calendar and cost model as the optimiser.
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
    """Backtest an equal-weight portfolio on the same schedule."""
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
    """Buy the index on the first date and hold."""
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