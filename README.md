# Portfolio optimiser (week 1)

Research tool, not a trading bot. You estimate portfolio weights on **historical returns**, then (from week 2) you **test those weights on later dates** the optimiser did not see.

This week: download prices → simple returns → mean `μ` and covariance `Σ` → long-only weights → **in-sample** efficient frontier. HI MY NANE IS AMIDJ this is a change

## What each folder is

| Path | Job |
|------|-----|
| `portopt/data.py` | Adjusted close prices + disk cache |
| `portopt/returns.py` | `r = P_t/P_{t-1}-1`, then `μ` and `Σ` |
| `portopt/optimiser.py` | SciPy: min-variance, max-Sharpe, risk-parity |
| `portopt/frontier.py` | Sweep target returns, plot risk vs return |
| `portopt/cli.py` | One command to produce `frontier.png` |
| `tests/` | Maths checks that **do not** call Yahoo |
| `portopt/backtest.py` | Empty on purpose (week 2) |
| `app/streamlit_app.py` | Empty on purpose (week 4) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the frontier (needs internet the first time)

Default US names, **2015–2019** as the in-sample window. That end date is deliberate: week 2 will use 2020 onward as the test window.

```bash
python3 -m portopt.cli --market US --start 2015-01-01 --end 2019-12-31 --out frontier.png
```

UK:

```bash
python3 -m portopt.cli --market UK --start 2015-01-01 --end 2019-12-31 --out frontier_uk.png
```

## Tests (no network)

```bash
pytest
```

## How to talk about this

- The frontier is **in-sample**. It is the best risk–return tradeoff *on the estimation window*, not a promise about the future.
- Do not say the optimiser “beats the market”. Week 2–3 will compare it honestly to equal-weight and an index, **including costs**.
