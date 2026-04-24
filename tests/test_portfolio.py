import numpy as np
import pandas as pd
import pytest

from QUANTAXIS.backtest.portfolio import (
    PortfolioConfig,
    compute_portfolio_risk,
    optimize_portfolio,
)


def _make_returns(n: int = 252, n_assets: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    symbols = [f"STOCK_{i}" for i in range(n_assets)]
    data = {}
    for sym in symbols:
        data[sym] = rng.normal(0.001, 0.02, n)
    return pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def test_optimize_mvo_equal():
    returns = _make_returns(100, 3)
    weights = optimize_portfolio(returns, PortfolioConfig(method="mvo", max_weight=0.5))
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(0 <= w <= 0.5 for w in weights.values())


def test_optimize_risk_parity():
    returns = _make_returns(100, 4)
    weights = optimize_portfolio(returns, PortfolioConfig(method="risk_parity", max_weight=0.5))
    assert len(weights) == 4
    assert abs(sum(weights.values()) - 1.0) < 0.1


def test_optimize_hrp():
    returns = _make_returns(100, 4)
    weights = optimize_portfolio(returns, PortfolioConfig(method="hrp", max_weight=0.5))
    assert len(weights) == 4
    assert abs(sum(weights.values()) - 1.0) < 0.1


def test_optimize_black_litterman():
    returns = _make_returns(100, 3)
    views = {"STOCK_0": 0.02, "STOCK_1": -0.01}
    weights = optimize_portfolio(
        returns,
        PortfolioConfig(method="black_litterman", max_weight=0.5),
        views=views,
    )
    assert len(weights) == 3


def test_optimize_empty_returns():
    weights = optimize_portfolio(pd.DataFrame(), PortfolioConfig())
    assert weights == {}


def test_optimize_single_asset():
    returns = pd.DataFrame({"A": np.random.randn(50)})
    weights = optimize_portfolio(returns, PortfolioConfig())
    assert weights == {"A": 1.0}


def test_optimize_with_group_constraint():
    returns = _make_returns(100, 4)
    group_map = {"STOCK_0": "tech", "STOCK_1": "tech", "STOCK_2": "finance", "STOCK_3": "finance"}
    weights = optimize_portfolio(
        returns,
        PortfolioConfig(method="risk_parity", group_constraints={"tech": 0.3}),
        group_map=group_map,
    )
    assert sum(weights[s] for s in weights if group_map.get(s, "") == "tech") <= 0.3 + 0.01


def test_compute_portfolio_risk():
    returns = _make_returns(100, 3).iloc[:, :2]
    weights = {"STOCK_0": 0.6, "STOCK_1": 0.4}
    risk = compute_portfolio_risk(weights, returns)
    assert "volatility" in risk
    assert "var_95" in risk
    assert "diversification_ratio" in risk
    assert risk["volatility"] >= 0
