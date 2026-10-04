import numpy as np
import pandas as pd
import pytest

from portopt.metrics import (
    annualised_return,
    annualised_volatility,
    compare,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    summarise,
    total_return,
)


def test_total_return():
    wealth = pd.Series([1.0, 1.1, 1.21])
    assert total_return(wealth) == pytest.approx(0.21)


def test_max_drawdown_known_answer():
    # peak 2.0 then falls to 1.0 -> -50%; later new high 3.0 has no drawdown
    wealth = pd.Series([1.0, 2.0, 1.0, 3.0])
    assert max_drawdown(wealth) == pytest.approx(-0.5)
    assert drawdown_series(wealth).iloc[-1] == 0.0


def test_max_drawdown_only_rising_is_zero():
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2])) == 0.0


def test_annualised_return_one_year_of_steady_growth():
    wealth = pd.Series(1.01 ** np.arange(253))  # 252 daily steps of +1%
    assert annualised_return(wealth) == pytest.approx(1.01**252 - 1)


def test_volatility_matches_numpy():
    wealth = pd.Series([1.0, 1.02, 1.00, 1.03, 1.01])
    rets = wealth.pct_change().dropna().to_numpy()
    assert annualised_volatility(wealth) == pytest.approx(rets.std(ddof=1) * np.sqrt(252))


def test_sharpe_is_nan_when_no_risk_and_positive_when_rising_noisily():
    assert np.isnan(sharpe_ratio(pd.Series(1.01 ** np.arange(10))))
    rng = np.random.default_rng(0)
    wealth = pd.Series(np.cumprod(1 + rng.normal(0.001, 0.01, 500)))
    assert sharpe_ratio(wealth) > 0


def test_summarise_and_compare_shapes():
    w = pd.Series([1.0, 1.05, 1.02, 1.08])
    s = summarise(w)
    assert list(s.index) == [
        "total_return", "annual_return", "annual_volatility", "sharpe", "max_drawdown",
    ]
    table = compare({"A": w, "B": w * 2})
    assert list(table.index) == ["A", "B"]


def test_bad_input_raises():
    with pytest.raises(ValueError):
        total_return(pd.Series([1.0]))
    with pytest.raises(ValueError):
        total_return(pd.Series([1.0, -1.0]))
