import pandas as pd

from portopt.data import prices_from_csv
from portopt.returns import estimate_mu_cov, simple_returns
from tests.conftest import FIXTURES


def test_simple_returns_match_pct_change():
    prices = prices_from_csv(FIXTURES / "prices.csv")
    rets = simple_returns(prices)
    expected = prices.pct_change().dropna()
    pd.testing.assert_frame_equal(rets, expected)
    assert len(rets) == len(prices) - 1


def test_mu_cov_shapes_and_annualisation():
    prices = prices_from_csv(FIXTURES / "prices.csv")
    rets = simple_returns(prices)
    mu, cov = estimate_mu_cov(rets, periods_per_year=252, annualise=True)
    mu_d, cov_d = estimate_mu_cov(rets, annualise=False)
    assert list(mu.index) == list(rets.columns)
    assert cov.shape == (3, 3)
    pd.testing.assert_series_equal(mu, mu_d * 252)
    pd.testing.assert_frame_equal(cov, cov_d * 252)
