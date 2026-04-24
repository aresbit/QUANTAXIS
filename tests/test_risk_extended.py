"""Extended tests for risk module: stress testing, factor exposure, etc."""

import numpy as np
import pandas as pd
import pytest

from QUANTAXIS.backtest.risk import (
    correlation_risk_matrix,
    factor_exposure_analysis,
    monte_carlo_var,
    sensitivity_analysis,
    stress_test_scenarios,
)


def _make_returns(n: int = 252) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.02, n), index=pd.date_range("2020-01-01", periods=n, freq="D"))


def test_stress_test_scenarios():
    returns = _make_returns(500)
    scenarios = stress_test_scenarios(returns)
    assert isinstance(scenarios, dict)
    for name, data in scenarios.items():
        assert "cumulative_return" in data
        assert "max_drawdown" in data
        assert "volatility" in data


def test_monte_carlo_var():
    returns = _make_returns(200)
    mc = monte_carlo_var(returns, n_simulations=1000, horizon=21)
    assert "mc_var_95" in mc
    assert "mc_cvar_95" in mc
    assert mc["mc_var_95"] >= 0


def test_monte_carlo_var_short_series():
    returns = pd.Series(np.random.randn(5))
    mc = monte_carlo_var(returns)
    assert mc["mc_var_95"] == 0.0


def test_factor_exposure():
    rng = np.random.default_rng(42)
    n = 200
    portfolio_returns = pd.Series(rng.normal(0.001, 0.02, n))
    factor_returns = pd.DataFrame({
        "market": rng.normal(0.0005, 0.01, n),
        "size": rng.normal(0.0002, 0.008, n),
        "value": rng.normal(0.0001, 0.006, n),
    })
    result = factor_exposure_analysis(portfolio_returns, factor_returns)
    assert "exposures" in result
    assert "r_squared" in result
    assert "adj_r_squared" in result
    assert result["r_squared"] >= 0


def test_factor_exposure_insufficient_data():
    portfolio_returns = pd.Series(np.random.randn(3))
    factor_returns = pd.DataFrame({"market": np.random.randn(3)})
    result = factor_exposure_analysis(portfolio_returns, factor_returns)
    assert result == {}


def test_correlation_risk_matrix():
    rng = np.random.default_rng(42)
    n_assets = 5
    n_periods = 100
    data = {f"asset_{i}": rng.normal(0.001, 0.02, n_periods) for i in range(n_assets)}
    returns = pd.DataFrame(data)
    result = correlation_risk_matrix(returns, lookback=100)
    assert "avg_correlation" in result
    assert "condition_number" in result
    assert result["n_assets"] == 5


def test_correlation_risk_matrix_single_asset():
    returns = pd.DataFrame({"A": np.random.randn(100)})
    result = correlation_risk_matrix(returns)
    assert result["avg_correlation"] == 0.0


def test_sensitivity_analysis():
    rng = np.random.default_rng(42)
    n = 200
    strat_returns = pd.Series(rng.normal(0.001, 0.02, n), index=pd.date_range("2020-01-01", periods=n, freq="D"))
    bench_returns = pd.Series(rng.normal(0.0005, 0.015, n), index=pd.date_range("2020-01-01", periods=n, freq="D"))
    result = sensitivity_analysis(strat_returns, benchmark_returns=bench_returns)
    assert "beta" in result
    assert "alpha" in result
    assert "up_capture" in result
    assert "down_capture" in result


def test_sensitivity_no_benchmark():
    returns = _make_returns(100)
    result = sensitivity_analysis(returns)
    assert result["beta"] == 0.0
