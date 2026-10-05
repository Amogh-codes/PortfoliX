"""Turn prices into simple returns, then estimate μ and Σ.

The optimiser uses:
- mu: expected return per asset
- cov: covariance of returns

Use simple returns so the backtest can update wealth as wealth * (1 + r).

Annualisation converts daily estimates into yearly units.
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily simple returns and drop incomplete rows."""
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
    """Estimate mean returns and covariance from historical returns.

    These estimates are noisy, so weights should be tested on a different
    date range.
    """
    if returns.shape[0] < 2:
        raise ValueError("Need at least two return observations.")
    mu = returns.mean()
    cov = returns.cov()
    if annualise:
        mu = mu * periods_per_year
        cov = cov * periods_per_year
    return mu, cov
