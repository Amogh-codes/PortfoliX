"""Backtest engine: test a strategy on data it has NOT seen.

How it works (one trading day at a time, from test_start onwards):

1. On a rebalance day, estimate mu and cov from the returns BEFORE that day
   only (no peeking at the future), ask the optimiser for target weights,
   and pay transaction costs on whatever has to be traded.
2. Apply that day's returns:  wealth_new = wealth * (1 + r)  (simple returns,
   same convention as returns.py).
3. Weights drift with prices until the next rebalance day.

Costs: cost = wealth * turnover * cost_bps / 10,000, where turnover is the
sum of |target weight - current weight|. Buying the first portfolio from
cash counts as turnover = 1, so every strategy pays to get started.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portopt.optimiser import Strategy, allocate
from portopt.returns import estimate_mu_cov, simple_returns

# Friendly name -> pandas period code (None = only buy once at the start).
REBALANCE_RULES: dict[str, str | None] = {
    "never": None,
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


@dataclass
class BacktestResult:
    """Everything a backtest produces."""

    wealth: pd.Series  # portfolio value; first point = initial_wealth, before any return
    weights: pd.DataFrame  # target weights chosen on each rebalance date
    turnover: pd.Series  # sum |target - current| on each rebalance date
    costs: pd.Series  # money paid in costs on each rebalance date

    @property
    def total_costs(self) -> float:
        return float(self.costs.sum())


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy = "max_sharpe",
    test_start: str | pd.Timestamp = "2019-01-01",
    lookback: int | None = 252,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
    risk_free: float = 0.0,
    initial_wealth: float = 1.0,
    min_history: int = 60,
) -> BacktestResult:
    """Walk-forward backtest of one strategy.

    prices:         date-indexed table of adjusted closes (columns = tickers).
    strategy:       "equal_weight", "min_variance", "max_sharpe" or "risk_parity".
    test_start:     first date whose returns are earned by the strategy.
                    Everything before it is training data only.
    lookback:       how many past daily returns to estimate mu/cov from.
                    None = use everything before the rebalance date.
    rebalance:      "never", "daily", "weekly", "monthly", "quarterly", "yearly".
    cost_bps:       trading cost in basis points (10 = 0.10%) of traded value.
    risk_free:      annual rate used by max_sharpe (same units as mu).
    min_history:    minimum number of past returns needed to estimate mu/cov.
    """
    if rebalance not in REBALANCE_RULES:
        raise ValueError(f"Unknown rebalance '{rebalance}'. Choose from {list(REBALANCE_RULES)}.")
    if lookback is not None and lookback < min_history:
        raise ValueError("lookback must be at least min_history.")
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative.")
    if initial_wealth <= 0:
        raise ValueError("initial_wealth must be positive.")

    rets = simple_returns(prices)
    i0 = int(rets.index.searchsorted(pd.Timestamp(test_start)))
    if i0 >= len(rets):
        raise ValueError("test_start is after the last date in the data.")
    if i0 < min_history:
        raise ValueError(
            f"Only {i0} returns before test_start; need at least {min_history} to estimate mu/cov."
        )

    test_dates = rets.index[i0:]
    is_rebalance = _rebalance_flags(test_dates, REBALANCE_RULES[rebalance])
    cost_rate = cost_bps / 10_000.0

    n_assets = rets.shape[1]
    wealth = float(initial_wealth)
    current = np.zeros(n_assets)  # start in cash: every weight is 0

    wealth_path = [wealth]
    weight_rows: list[pd.Series] = []
    turnover_rows: list[float] = []
    cost_rows: list[float] = []
    rebalance_dates: list[pd.Timestamp] = []

    for step, date in enumerate(test_dates):
        i = i0 + step

        if is_rebalance[step]:
            start = 0 if lookback is None else i - lookback
            window = rets.iloc[max(0, start) : i]  # rows strictly BEFORE today
            mu, cov = estimate_mu_cov(window)
            target = allocate(strategy, mu, cov, risk_free=risk_free)

            target_arr = target.reindex(rets.columns).to_numpy(dtype=float)
            turnover = float(np.abs(target_arr - current).sum())
            cost = wealth * turnover * cost_rate

            wealth -= cost
            current = target_arr

            weight_rows.append(target)
            turnover_rows.append(turnover)
            cost_rows.append(cost)
            rebalance_dates.append(date)

        growth = 1.0 + rets.iloc[i].to_numpy(dtype=float)
        holdings = current * growth  # what each slice of the portfolio is worth now
        port_growth = float(holdings.sum())
        wealth *= port_growth
        current = holdings / port_growth  # weights drift with prices
        wealth_path.append(wealth)

    start_label = rets.index[i0 - 1]
    wealth_series = pd.Series(
        wealth_path,
        index=pd.DatetimeIndex([start_label, *test_dates]),
        name="wealth",
    )
    return BacktestResult(
        wealth=wealth_series,
        weights=pd.DataFrame(weight_rows, index=pd.DatetimeIndex(rebalance_dates)),
        turnover=pd.Series(turnover_rows, index=rebalance_dates, name="turnover"),
        costs=pd.Series(cost_rows, index=rebalance_dates, name="costs"),
    )


def _rebalance_flags(dates: pd.DatetimeIndex, freq: str | None) -> np.ndarray:
    """True on the first trading day of each new period (and always on day one)."""
    flags = np.zeros(len(dates), dtype=bool)
    flags[0] = True
    if freq is not None and len(dates) > 1:
        periods = dates.to_period(freq)
        flags[1:] = np.asarray(periods[1:] != periods[:-1])
    return flags