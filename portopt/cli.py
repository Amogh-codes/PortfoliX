"""Command-line tool to download prices, estimate μ and Σ, and plot the frontier.

Example:
    python -m portopt.cli --start 2015-01-01 --end 2019-12-31 --out frontier.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from portopt.data import download_prices
from portopt.frontier import compute_frontier, plot_frontier
from portopt.optimiser import allocate
from portopt.returns import estimate_mu_cov, simple_returns
from portopt.universe import DEFAULT_UK_TICKERS, DEFAULT_US_TICKERS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate a mean-variance frontier (in-sample).")
    parser.add_argument("--market", choices=("US", "UK"), default="US")
    parser.add_argument("--tickers", nargs="*", help="Override default tickers.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2019-12-31", help="In-sample end date (not the backtest).")
    parser.add_argument("--out", default="frontier.png")
    parser.add_argument("--risk-free", type=float, default=0.0)
    args = parser.parse_args(argv)

    tickers = args.tickers or list(
        DEFAULT_US_TICKERS if args.market == "US" else DEFAULT_UK_TICKERS
    )
    prices = download_prices(tickers, start=args.start, end=args.end)
    returns = simple_returns(prices)
    mu, cov = estimate_mu_cov(returns)

    print(f"Assets: {list(prices.columns)}")
    print(f"Days: {len(prices)} prices, {len(returns)} returns")
    print("Max-Sharpe weights (in-sample):")
    print(allocate("max_sharpe", mu, cov, risk_free=args.risk_free).round(4).to_string())
    print("Min-variance weights (in-sample):")
    print(allocate("min_variance", mu, cov).round(4).to_string())

    frontier = compute_frontier(mu, cov, risk_free=args.risk_free)
    fig = plot_frontier(frontier)
    out = Path(args.out)
    fig.savefig(out, dpi=140)
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
