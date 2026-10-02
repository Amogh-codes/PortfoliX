"""Default stock lists and index tickers.

Keep the universe small (8–15 liquid names). Mean-variance with 500 stocks
produces unstable weights; a demo that you can explain is more useful.
"""

from __future__ import annotations

DEFAULT_US_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JNJ",
    "JPM",
    "XOM",
    "NVDA",
)

DEFAULT_UK_TICKERS: tuple[str, ...] = (
    "AZN.L",
    "HSBA.L",
    "SHEL.L",
    "ULVR.L",
    "VOD.L",
    "BP.L",
    "REL.L",
    "LLOY.L",
)

# Yahoo Finance tickers for a cheap, liquid index proxy.
BENCHMARKS: dict[str, str] = {
    "US": "SPY",  # S&P 500 ETF
    "UK": "ISF.L",  # iShares Core FTSE 100
}
