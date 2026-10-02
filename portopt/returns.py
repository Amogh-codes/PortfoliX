"""Turn prices into simple returns, then into μ and Σ.

The optimiser does not read prices. It reads:
- mu: expected return per asset (vector)
- cov: covariance of returns (matrix)

We use *simple* returns r_t = P_t / P_{t-1} - 1 so that a backtest can
update wealth as wealth * (1 + r). Keep that convention everywhere.

Annualisation (252 trading days) is only a unit conversion so we can talk
about "12% volatility" instead of "0.75% per day".
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns. First row is NaN (no previous price) and is dropped."""
    rets = prices.pct_change().dropna(how="all")
    rets = rets.dropna(how="any")
    if rets.empty:
        raise ValueError("Need at least two price rows to compute returns.")
    return rets


def estimate_mu_cov(
    returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    annualise: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Sample mean and covariance.

    These are noisy estimates. That is why we later test weights on a
    *different* date range than the one used here.
    """
    if returns.shape[0] < 2:
        raise ValueError("Need at least two return observations.")
    mu = returns.mean()
    cov = returns.cov()
    if annualise:
        mu = mu * periods_per_year
        cov = cov * periods_per_year
    return mu, cov
