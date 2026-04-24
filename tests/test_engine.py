import pandas as pd
import pytest

from QUANTAXIS.backtest.engine import run_backtest
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig


def _make_ohlcv(n: int = 100, trend: str = "flat") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 10.0
    if trend == "up":
        closes = [base + i * 0.05 for i in range(n)]
    elif trend == "down":
        closes = [base - i * 0.05 for i in range(n)]
    else:
        closes = [base + (i % 10) * 0.01 for i in range(n)]
    data = {
        "datetime": dates,
        "open": [c - 0.01 for c in closes],
        "high": [c + 0.02 for c in closes],
        "low": [c - 0.02 for c in closes],
        "close": closes,
        "volume": [1000 + i * 10 for i in range(n)],
    }
    return pd.DataFrame(data)


def test_run_backtest_basic():
    df = _make_ohlcv(50)
    strategy = RecursiveQTransformerStrategy(StrategyConfig(trade_size=100, buy_threshold=0.0))
    result = run_backtest(df, strategy, initial_cash=100_000)
    assert result.bars == 50
    assert result.final_equity > 0
    assert isinstance(result.equity_curve, list)


def test_run_backtest_missing_columns():
    df = pd.DataFrame({"datetime": ["2020-01-01"], "close": [10.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest(df, RecursiveQTransformerStrategy())


def test_run_backtest_portfolio_size_validation():
    df = _make_ohlcv(10)
    with pytest.raises(ValueError, match="portfolio_size must be at least 1"):
        run_backtest(df, RecursiveQTransformerStrategy(), portfolio_size=0)


def test_run_backtest_with_slippage():
    df = _make_ohlcv(50, trend="up")
    strategy = RecursiveQTransformerStrategy(StrategyConfig(trade_size=100, buy_threshold=-1.0))
    result = run_backtest(df, strategy, initial_cash=100_000, slippage_model="fixed", slippage_value=0.01)
    assert result.bars == 50
    assert result.rejected >= 0
