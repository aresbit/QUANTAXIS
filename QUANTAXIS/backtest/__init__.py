"""Local backtesting utilities for strategy research."""

from QUANTAXIS.backtest.data import fetch_ashare_daily, load_ohlcv_csv
from QUANTAXIS.backtest.engine import BacktestResult, run_backtest
from QUANTAXIS.backtest.plot import save_backtest_figure
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig

__all__ = [
    "BacktestResult",
    "RecursiveQTransformerStrategy",
    "StrategyConfig",
    "fetch_ashare_daily",
    "load_ohlcv_csv",
    "run_backtest",
    "save_backtest_figure",
]
