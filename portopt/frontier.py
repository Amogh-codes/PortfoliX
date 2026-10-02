"""In-sample efficient frontier.

Each point is: minimise variance subject to a target expected return,
long-only, fully invested. The curve lives in *in-sample* space — it is
not a forecast of future returns. Label that clearly on any plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portopt.optimiser import max_sharpe, min_variance, min_variance_for_target


def compute_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 25,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Return columns: target_return, volatility, expected_return, plus weights."""
    gmv = min_variance(mu, cov)
    gmv_ret = float(gmv @ mu)
    max_asset_ret = float(mu.max())
    targets = np.linspace(gmv_ret, max_asset_ret, n_points)

    rows = []
    for target in targets:
        weights = min_variance_for_target(mu, cov, float(target))
        if weights is None:
            continue
        vol = float(np.sqrt(weights.values @ cov.values @ weights.values))
        exp_ret = float(weights @ mu)
        row = {
            "target_return": float(target),
            "volatility": vol,
            "expected_return": exp_ret,
        }
        for name, value in weights.items():
            row[f"w_{name}"] = float(value)
        rows.append(row)

    if not rows:
        raise RuntimeError("Could not compute any frontier points.")

    frontier = pd.DataFrame(rows)

    sharpe_w = max_sharpe(mu, cov, risk_free=risk_free)
    markers = pd.DataFrame(
        [
            {
                "label": "min_variance",
                "volatility": float(np.sqrt(gmv.values @ cov.values @ gmv.values)),
                "expected_return": gmv_ret,
            },
            {
                "label": "max_sharpe",
                "volatility": float(np.sqrt(sharpe_w.values @ cov.values @ sharpe_w.values)),
                "expected_return": float(sharpe_w @ mu),
            },
        ]
    )
    frontier.attrs["markers"] = markers
    frontier.attrs["min_variance_weights"] = gmv
    frontier.attrs["max_sharpe_weights"] = sharpe_w
    return frontier


def plot_frontier(frontier: pd.DataFrame, title: str = "Efficient frontier (in-sample)"):
    """Matplotlib figure. Caller is responsible for show/save."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        frontier["volatility"],
        frontier["expected_return"],
        color="steelblue",
        label="Frontier",
    )
    markers = frontier.attrs.get("markers")
    if markers is not None:
        for _, row in markers.iterrows():
            ax.scatter(row["volatility"], row["expected_return"], zorder=3)
            ax.annotate(
                row["label"],
                (row["volatility"], row["expected_return"]),
                textcoords="offset points",
                xytext=(6, 6),
            )
    ax.set_xlabel("Volatility (annualised)")
    ax.set_ylabel("Expected return (annualised)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
