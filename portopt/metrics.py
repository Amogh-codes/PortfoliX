"""Performance metrics from a wealth series.

A wealth series is a pandas Series of portfolio values over time. The
functions here take wealth in and return a metric, which makes them easy
to test with simple example data.

Conventions:
- Simple returns: r_t = W_t / W_{t-1} - 1
- 252 trading days per year
- Sharpe uses mean daily return * 252 as the annual return
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
 
TRADING_DAYS = 252
 
 
def _check(wealth: pd.Series) -> None:
    if len(wealth) < 2:
        raise ValueError("Need at least two wealth observations.")
    if (wealth <= 0).any():
        raise ValueError("Wealth must be strictly positive.")
 
 
def wealth_returns(wealth: pd.Series) -> pd.Series:
    """Period-by-period simple returns of a wealth series."""
    _check(wealth)
    return wealth.pct_change().dropna()
 
 
def total_return(wealth: pd.Series) -> float:
    """Final wealth / starting wealth - 1 (0.10 means +10%)."""
    _check(wealth)
    return float(wealth.iloc[-1] / wealth.iloc[0] - 1.0)
 
 
def annualised_return(wealth: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate (CAGR)."""
    _check(wealth)
    years = (len(wealth) - 1) / periods_per_year
    return float((wealth.iloc[-1] / wealth.iloc[0]) ** (1.0 / years) - 1.0)
 
 
def annualised_volatility(wealth: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard deviation of period returns, scaled to a year."""
    rets = wealth_returns(wealth)
    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))
 
 
def sharpe_ratio(
    wealth: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """(annual mean return - risk_free) / annual volatility.
 
    risk_free is an annual rate (0.02 = 2%). Returns NaN if volatility is 0.
    """
    rets = wealth_returns(wealth)
    vol = rets.std(ddof=1) * np.sqrt(periods_per_year)
    if vol < 1e-12:
        return float("nan")
    return float((rets.mean() * periods_per_year - risk_free) / vol)
 
 
def drawdown_series(wealth: pd.Series) -> pd.Series:
    """How far below its previous peak wealth is at each date (0 or negative)."""
    _check(wealth)
    return wealth / wealth.cummax() - 1.0
 
 
def max_drawdown(wealth: pd.Series) -> float:
    """Worst peak-to-trough fall, as a negative number (-0.25 = -25%)."""
    return float(drawdown_series(wealth).min())
 
 
def summarise(
    wealth: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """All headline numbers in one Series."""
    return pd.Series(
        {
            "total_return": total_return(wealth),
            "annual_return": annualised_return(wealth, periods_per_year),
            "annual_volatility": annualised_volatility(wealth, periods_per_year),
            "sharpe": sharpe_ratio(wealth, risk_free, periods_per_year),
            "max_drawdown": max_drawdown(wealth),
        },
        name="metrics",
    )
 
 
def compare(
    curves: dict[str, pd.Series],
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Side-by-side table: one row per named wealth curve."""
    rows = {name: summarise(w, risk_free, periods_per_year) for name, w in curves.items()}
    return pd.DataFrame(rows).T
