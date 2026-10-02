import numpy as np
import pandas as pd

from portopt.optimiser import allocate, max_sharpe, min_variance, risk_parity


def _identical_assets(n: int = 3) -> tuple[pd.Series, pd.DataFrame]:
    names = [f"S{i}" for i in range(n)]
    mu = pd.Series([0.08] * n, index=names)
    cov = pd.DataFrame(np.eye(n) * 0.04, index=names, columns=names)
    return mu, cov


def test_min_variance_identical_assets_equal_weights():
    mu, cov = _identical_assets()
    weights = min_variance(mu, cov)
    np.testing.assert_allclose(weights.values, np.full(3, 1 / 3), atol=1e-5)
    assert abs(weights.sum() - 1) < 1e-8
    assert (weights >= -1e-10).all()


def test_max_sharpe_identical_assets_equal_weights():
    mu, cov = _identical_assets()
    weights = max_sharpe(mu, cov, risk_free=0.0)
    np.testing.assert_allclose(weights.values, np.full(3, 1 / 3), atol=1e-5)


def test_risk_parity_identical_assets_equal_weights():
    mu, cov = _identical_assets()
    weights = risk_parity(mu, cov)
    np.testing.assert_allclose(weights.values, np.full(3, 1 / 3), atol=1e-4)


def test_allocate_unknown_strategy():
    mu, cov = _identical_assets()
    try:
        allocate("not_a_strategy", mu, cov)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_min_variance_prefers_the_quiet_asset():
    names = ["NOISY", "QUIET"]
    mu = pd.Series([0.10, 0.10], index=names)
    cov = pd.DataFrame([[0.09, 0.0], [0.0, 0.01]], index=names, columns=names)
    weights = min_variance(mu, cov)
    assert weights["QUIET"] > weights["NOISY"]
