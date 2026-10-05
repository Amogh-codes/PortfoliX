"""Long-only portfolio weights using SciPy SLSQP.

All strategies return weights that sum to 1 and are non-negative.

If the solver fails, fall back to equal weights.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

Strategy = Literal["equal_weight", "min_variance", "max_sharpe", "risk_parity"]


def equal_weight(mu: pd.Series) -> pd.Series:
    n = len(mu)
    if n == 0:
        raise ValueError("mu is empty.")
    return pd.Series(np.ones(n) / n, index=mu.index, name="weight")


def min_variance(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Lowest-volatility fully invested long-only portfolio."""
    _validate(mu, cov)
    n = len(mu)
    return _solve(
        mu,
        objective=lambda w: float(w @ cov.values @ w),
        n=n,
        fallback=equal_weight(mu),
    )


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    """Maximise the Sharpe ratio. risk_free must use the same units as mu."""
    _validate(mu, cov)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov.values @ w))
        if vol < 1e-12:
            return 0.0
        ret = float(w @ mu.values)
        return -(ret - risk_free) / vol

    return _solve(mu, objective=neg_sharpe, n=len(mu), fallback=equal_weight(mu))


def min_variance_for_target(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_return: float,
) -> pd.Series | None:
    """Minimise variance for a target return. Used to trace the frontier."""
    _validate(mu, cov)
    n = len(mu)
    extra = {"type": "eq", "fun": lambda w: float(w @ mu.values) - target_return}
    weights = _solve(
        mu,
        objective=lambda w: float(w @ cov.values @ w),
        n=n,
        extra_constraints=[extra],
        fallback=None,
    )
    return weights


def risk_parity(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Equal risk contribution across assets.

    Risk contribution is w_i * (Σw)_i. mu is unused; the signature matches the other strategies.
    """
    _validate(mu, cov)
    n = len(mu)
    sigma = cov.values

    def objective(w: np.ndarray) -> float:
        rc = w * (sigma @ w)
        target = rc.sum() / n
        return float(np.sum((rc - target) ** 2))

    return _solve(mu, objective=objective, n=n, fallback=equal_weight(mu))


def allocate(strategy: Strategy, mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    if strategy == "equal_weight":
        return equal_weight(mu)
    if strategy == "min_variance":
        return min_variance(mu, cov)
    if strategy == "max_sharpe":
        return max_sharpe(mu, cov, risk_free=risk_free)
    if strategy == "risk_parity":
        return risk_parity(mu, cov)
    raise ValueError(f"Unknown strategy: {strategy}")


def _validate(mu: pd.Series, cov: pd.DataFrame) -> None:
    if list(mu.index) != list(cov.index) or list(mu.index) != list(cov.columns):
        raise ValueError("mu and cov must share the same asset labels.")
    if mu.empty:
        raise ValueError("Need at least one asset.")


def _solve(
    mu: pd.Series,
    objective,
    n: int,
    extra_constraints: list | None = None,
    fallback: pd.Series | None = None,
) -> pd.Series | None:
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-9},
    )
    if not result.success:
        return fallback
    weights = np.clip(result.x, 0.0, 1.0)
    total = weights.sum()
    if total <= 0:
        return fallback
    weights = weights / total
    return pd.Series(weights, index=mu.index, name="weight")
