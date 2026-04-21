"""Local backtesting utilities for strategy research."""

from QUANTAXIS.backtest.data import fetch_ashare_bars, fetch_ashare_daily, fetch_ashare_portfolio_bars, load_multi_ohlcv_csv, load_ohlcv_csv
from QUANTAXIS.backtest.engine import BacktestResult, run_backtest
from QUANTAXIS.backtest.plot import save_backtest_figure
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig

__all__ = [
    "BacktestResult",
    "RecursiveQTransformerStrategy",
    "StrategyConfig",
    "fetch_ashare_bars",
    "fetch_ashare_daily",
    "fetch_ashare_portfolio_bars",
    "load_multi_ohlcv_csv",
    "load_ohlcv_csv",
    "run_backtest",
    "save_backtest_figure",
]
