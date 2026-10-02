from pathlib import Path

import pandas as pd
import pytest

from portopt.data import prices_from_csv
from tests.conftest import FIXTURES


def test_prices_from_csv_loads_dates_and_tickers():
    prices = prices_from_csv(FIXTURES / "prices.csv")
    assert list(prices.columns) == ["AAA", "BBB", "CCC"]
    assert prices.index[0] == pd.Timestamp("2018-01-02")
    assert (prices > 0).all().all()
    assert len(prices) == 10


def test_prices_from_csv_rejects_empty(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text("date,AAA\n")
    with pytest.raises(ValueError):
        prices_from_csv(empty)
