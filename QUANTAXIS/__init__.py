"""Minimal QUANTAXIS build focused on China A-share execution."""

from QUANTAXIS.ashare.account import AccountSnapshot, Position
from QUANTAXIS.ashare.broker import EasyTraderBroker, ExecutionReport, Order, PaperBroker
from QUANTAXIS.ashare.quotes import PytdxQuoteClient, Quote
from QUANTAXIS.ashare.runner import run_once_from_config
from QUANTAXIS.backtest.data import fetch_ashare_daily, load_ohlcv_csv
from QUANTAXIS.backtest.engine import BacktestResult, run_backtest
from QUANTAXIS.backtest.plot import save_backtest_figure
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig

__all__ = [
    "AccountSnapshot",
    "BacktestResult",
    "EasyTraderBroker",
    "ExecutionReport",
    "Order",
    "PaperBroker",
    "Position",
    "PytdxQuoteClient",
    "Quote",
    "RecursiveQTransformerStrategy",
    "StrategyConfig",
    "fetch_ashare_daily",
    "load_ohlcv_csv",
    "run_backtest",
    "run_once_from_config",
    "save_backtest_figure",
]

__author__ = "yutiansut"
__version__ = "3.0.0"
