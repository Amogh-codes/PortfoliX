"""Download and cache adjusted close prices.

Adjusted close includes splits and dividends. Using raw close would treat a
2-for-1 split as a 50% crash, which would poison returns and the optimiser.

Tests should load a CSV fixture instead of hitting Yahoo Finance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def prices_from_csv(path: str | Path) -> pd.DataFrame:
    """Load a date-indexed price table (columns = tickers)."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return _clean_prices(frame)


def download_prices(
    tickers: list[str] | tuple[str, ...],
    start: str,
    end: str,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance, with an optional CSV cache.

    Cache key is tickers + dates, so changing the universe fetches again.
    """
    tickers = _normalise_tickers(tickers)
    if not tickers:
        raise ValueError("Need at least one ticker.")

    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_path(cache_dir, tickers, start, end)
        if cache_path.exists():
            return prices_from_csv(cache_path)

    import yfinance as yf  # imported lazily so unit tests need no yfinance I/O

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )
    prices = _extract_close(raw, tickers)
    prices = _clean_prices(prices)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)

    return prices


def _normalise_tickers(tickers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for ticker in tickers:
        cleaned = ticker.strip().upper()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return tuple(seen)


def _cache_path(cache_dir: str | Path, tickers: tuple[str, ...], start: str, end: str) -> Path:
    slug = "_".join(tickers) + f"_{start}_{end}"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in slug)
    return Path(cache_dir) / f"{safe}.csv"


def _extract_close(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """yfinance returns a MultiIndex when several tickers are requested."""
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in set(level0):
            close = raw["Close"].copy()
        elif "Adj Close" in set(level0):
            close = raw["Adj Close"].copy()
        else:
            raise ValueError(f"No Close column in download. Columns: {raw.columns}")
    else:
        # Single ticker: columns are Open/High/Low/Close/...
        if "Close" in raw.columns:
            close = raw[["Close"]].copy()
            close.columns = [tickers[0]]
        else:
            close = raw.copy()
    return close


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows/cols, sort dates, and require a shared calendar.

    Inner-join style dropna means we only keep days where every name has a
    price. That avoids mixing US and UK holidays in one covariance matrix.
    """
    frame = prices.copy()
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    frame = frame.sort_index()
    frame = frame.replace(0, pd.NA)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    frame = frame.dropna(axis=0, how="any")
    if frame.empty:
        raise ValueError("Price table is empty after cleaning.")
    if (frame <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    return frame.astype(float)
