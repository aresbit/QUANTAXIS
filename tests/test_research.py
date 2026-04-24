import numpy as np
import pandas as pd
import pytest

from QUANTAXIS.backtest.research import (
    Experiment,
    compute_normal_ic,
    compute_rank_ic,
    factor_ic_analysis,
    grid_search,
    log_experiment,
    set_experiment_dir,
    walk_forward,
)


def _make_factor_data(n: int = 100) -> tuple[pd.Series, pd.Series]:
    date_idx = pd.date_range("2020-01-01", periods=n, freq="D")
    factor = pd.Series(np.random.randn(n), index=date_idx, name="factor")
    forward = pd.Series(factor.values * 0.5 + np.random.randn(n) * 0.1, index=date_idx, name="fwd_return")
    return factor, forward


def test_rank_ic():
    factor, fwd = _make_factor_data(50)
    ic = compute_rank_ic(factor, fwd)
    assert isinstance(ic, float)


def test_normal_ic():
    factor, fwd = _make_factor_data(50)
    ic = compute_normal_ic(factor, fwd)
    assert isinstance(ic, float)


def test_rank_ic_too_few():
    factor = pd.Series([1.0, 2.0])
    fwd = pd.Series([0.01, -0.02])
    ic = compute_rank_ic(factor, fwd)
    assert ic == 0.0


def test_factor_ic_analysis():
    factor, fwd = _make_factor_data(200)
    report = factor_ic_analysis(factor, fwd, factor_name="test")
    assert report.factor == "test"
    assert isinstance(report.rank_ic_mean, float)
    assert len(report.ic_decay) == 5


def test_factor_ic_analysis_short_series():
    factor = pd.Series(np.random.randn(10))
    fwd = pd.Series(np.random.randn(10))
    report = factor_ic_analysis(factor, fwd)
    assert report.factor == "factor"


def test_grid_search(tmp_path):
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 10.0
    closes = [base + i * 0.02 for i in range(n)]
    df = pd.DataFrame({
        "datetime": dates,
        "open": [c - 0.01 for c in closes],
        "high": [c + 0.02 for c in closes],
        "low": [c - 0.02 for c in closes],
        "close": closes,
        "volume": [1000 + i * 10 for i in range(n)],
    })

    from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig
    from QUANTAXIS.backtest.engine import run_backtest

    def _backtest_fn(data, strategy, **kwargs):
        return run_backtest(data, strategy, initial_cash=100000, **kwargs)

    param_grid = {"buy_threshold": [0.0, 0.02], "min_holding_bars": [2, 5]}
    result = grid_search(
        df, RecursiveQTransformerStrategy, param_grid, _backtest_fn,
        scoring="sharpe",
    )
    assert "buy_threshold" in result.best_params
    assert len(result.all_results) == 4
    assert result.best_score >= 0


def test_walk_forward():
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 10.0
    closes = [base + i * 0.02 for i in range(n)]
    df = pd.DataFrame({
        "datetime": dates,
        "open": [c - 0.01 for c in closes],
        "high": [c + 0.02 for c in closes],
        "low": [c - 0.02 for c in closes],
        "close": closes,
        "volume": [1000 + i * 10 for i in range(n)],
    })

    from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig
    from QUANTAXIS.backtest.engine import run_backtest

    def _builder(train_data):
        return RecursiveQTransformerStrategy(StrategyConfig(buy_threshold=-0.01, hold_threshold=-0.02))

    result = walk_forward(
        df, _builder, run_backtest,
        window=200, step=50, min_train=100,
        initial_cash=100000,
    )
    assert len(result.folds) >= 1
    assert isinstance(result.oos_sharpe_mean, float)


def test_experiment_tracking(tmp_path):
    set_experiment_dir(tmp_path / "experiments")
    exp = log_experiment(
        name="test_run",
        parameters={"lr": 0.01},
        metrics={"sharpe": 1.5},
        tags=["test"],
    )
    assert exp.name == "test_run"
    assert exp.metrics["sharpe"] == 1.5

    from QUANTAXIS.backtest.research import list_experiments
    exps = list_experiments()
    assert len(exps) >= 1
    assert any(e.name == "test_run" for e in exps)
