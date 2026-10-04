import numpy as np
import pandas as pd
import pytest

from portopt.backtest import run_backtest
from portopt.benchmarks import equal_weight_backtest, index_backtest


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2017-01-02", periods=600)
    rets = rng.normal([0.0005, 0.0003, 0.0007], [0.01, 0.015, 0.02], size=(600, 3))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=["A", "B", "C"])


def test_wealth_starts_at_initial_and_covers_test_period(prices):
    res = run_backtest(prices, "min_variance", test_start="2018-06-01", initial_wealth=100.0)
    assert res.wealth.iloc[0] == 100.0
    assert res.wealth.index[1] >= pd.Timestamp("2018-06-01")
    assert res.wealth.index[-1] == prices.index[-1]
    assert (res.wealth > 0).all()


def test_weights_are_valid_and_one_row_per_rebalance(prices):
    res = run_backtest(prices, "max_sharpe", test_start="2018-06-01", rebalance="monthly")
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)
    assert (res.weights >= -1e-9).all().all()
    assert len(res.weights) == len(res.turnover) == len(res.costs)
    assert len(res.weights) > 5


def test_equal_weight_daily_rebalance_zero_cost_matches_hand_calc(prices):
    res = run_backtest(prices, "equal_weight", test_start="2018-06-01", rebalance="daily", cost_bps=0)
    daily = prices.pct_change().loc[res.wealth.index[1:]].mean(axis=1)
    expected = res.wealth.iloc[0] * (1 + daily).cumprod()
    np.testing.assert_allclose(res.wealth.iloc[1:].to_numpy(), expected.to_numpy(), rtol=1e-10)


def test_never_rebalance_zero_cost_is_buy_and_hold(prices):
    res = run_backtest(prices, "equal_weight", test_start="2018-06-01", rebalance="never", cost_bps=0)
    ratio = prices.loc[res.wealth.index] / prices.loc[res.wealth.index[0]]
    np.testing.assert_allclose(res.wealth.to_numpy(), ratio.mean(axis=1).to_numpy(), rtol=1e-10)
    assert len(res.weights) == 1


def test_costs_reduce_final_wealth_and_first_buy_costs_cost_rate(prices):
    free = run_backtest(prices, "equal_weight", test_start="2018-06-01", cost_bps=0)
    paid = run_backtest(prices, "equal_weight", test_start="2018-06-01", cost_bps=50)
    assert paid.wealth.iloc[-1] < free.wealth.iloc[-1]
    assert paid.turnover.iloc[0] == pytest.approx(1.0)
    assert paid.costs.iloc[0] == pytest.approx(0.005)  # 50 bps of wealth 1.0


def test_no_lookahead_future_prices_do_not_change_past_wealth(prices):
    cut = prices.index[-40]
    changed = prices.copy()
    changed.loc[cut:, "A"] *= 3.0  # shock only the future
    a = run_backtest(prices, "max_sharpe", test_start="2018-06-01")
    b = run_backtest(changed, "max_sharpe", test_start="2018-06-01")
    before = a.wealth.index < cut
    np.testing.assert_allclose(a.wealth[before].to_numpy(), b.wealth[before].to_numpy(), rtol=1e-10)


def test_bad_arguments_raise(prices):
    with pytest.raises(ValueError):
        run_backtest(prices, "min_variance", rebalance="hourly")
    with pytest.raises(ValueError):
        run_backtest(prices, "min_variance", test_start="2017-01-10")  # too little history
    with pytest.raises(ValueError):
        run_backtest(prices, "min_variance", test_start="2030-01-01")


def test_equal_weight_benchmark_matches_engine(prices):
    kwargs = dict(test_start="2018-06-01", rebalance="quarterly", cost_bps=10)
    bench = equal_weight_backtest(prices, **kwargs)
    direct = run_backtest(prices, "equal_weight", **kwargs)
    pd.testing.assert_series_equal(bench.wealth, direct.wealth)


def test_index_backtest_buy_and_hold(prices):
    res = run_backtest(prices, "equal_weight", test_start="2018-06-01")
    index = prices["A"].rename("SPY")
    free = index_backtest(index, res.wealth.index, cost_bps=0)
    np.testing.assert_allclose(
        free.to_numpy(), (index.loc[res.wealth.index] / index.loc[res.wealth.index[0]]).to_numpy()
    )
    paid = index_backtest(index, res.wealth.index, cost_bps=100)
    assert paid.iloc[0] == 1.0
    assert paid.iloc[-1] == pytest.approx(free.iloc[-1] * 0.99)


def test_index_backtest_uses_last_price_on_missing_days(prices):
    res = run_backtest(prices, "equal_weight", test_start="2018-06-01")
    index = prices["A"].drop(res.wealth.index[5])  # index market closed that day
    out = index_backtest(index, res.wealth.index, cost_bps=0)
    assert out.iloc[5] == pytest.approx(out.iloc[4])
