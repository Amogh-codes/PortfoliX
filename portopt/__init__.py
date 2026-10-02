"""Portfolio optimisation and backtesting toolkit.

This package is a research tool: estimate weights on one window of data,
then evaluate them on another. It is not a trading bot.
"""

from portopt.universe import (
    BENCHMARKS,
    DEFAULT_UK_TICKERS,
    DEFAULT_US_TICKERS,
)

__all__ = [
    "BENCHMARKS",
    "DEFAULT_UK_TICKERS",
    "DEFAULT_US_TICKERS",
]
