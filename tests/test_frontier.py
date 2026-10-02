import numpy as np
import pandas as pd

from portopt.frontier import compute_frontier


def test_frontier_volatility_rises_with_target_return():
    names = ["A", "B"]
    mu = pd.Series([0.05, 0.15], index=names)
    cov = pd.DataFrame([[0.02, 0.0], [0.0, 0.08]], index=names, columns=names)
    frontier = compute_frontier(mu, cov, n_points=8)
    assert len(frontier) >= 3
    # Higher expected return should not come with *lower* vol on a well-behaved frontier.
    vols = frontier.sort_values("expected_return")["volatility"].values
    assert vols[-1] >= vols[0] - 1e-9
    assert {"min_variance", "max_sharpe"} <= set(frontier.attrs["markers"]["label"])
    np.testing.assert_allclose(
        frontier[[c for c in frontier.columns if c.startswith("w_")]].sum(axis=1).values,
        1.0,
        atol=1e-6,
    )
