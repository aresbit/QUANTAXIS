"""QUANTAXIS package entrypoint with lightweight + legacy-compatible exports."""

from __future__ import annotations

from importlib import import_module

from QUANTAXIS.ashare.account import AccountSnapshot, Position
from QUANTAXIS.ashare.broker import EasyTraderBroker, ExecutionReport, Order, PaperBroker
from QUANTAXIS.ashare.quotes import PytdxQuoteClient, Quote
from QUANTAXIS.ashare.runner import run_once_from_config
from QUANTAXIS.backtest.data import fetch_ashare_daily, load_ohlcv_csv
from QUANTAXIS.backtest.engine import BacktestResult, run_backtest
from QUANTAXIS.backtest.plot import save_backtest_figure
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig

__author__ = "yutiansut"
__version__ = "3.0.0"

_DIRECT_EXPORTS = {
    "AccountSnapshot": AccountSnapshot,
    "BacktestResult": BacktestResult,
    "EasyTraderBroker": EasyTraderBroker,
    "ExecutionReport": ExecutionReport,
    "Order": Order,
    "PaperBroker": PaperBroker,
    "Position": Position,
    "PytdxQuoteClient": PytdxQuoteClient,
    "Quote": Quote,
    "RecursiveQTransformerStrategy": RecursiveQTransformerStrategy,
    "StrategyConfig": StrategyConfig,
    "fetch_ashare_daily": fetch_ashare_daily,
    "load_ohlcv_csv": load_ohlcv_csv,
    "run_backtest": run_backtest,
    "run_once_from_config": run_once_from_config,
    "save_backtest_figure": save_backtest_figure,
}

_LEGACY_ATTRS = {
    "QA_fetch_get_stock_day": ("QUANTAXIS.QAFetch", "QA_fetch_get_stock_day"),
    "QA_fetch_get_stock_min": ("QUANTAXIS.QAFetch", "QA_fetch_get_stock_min"),
    "QA_fetch_get_stock_list": ("QUANTAXIS.QAFetch", "QA_fetch_get_stock_list"),
    "QA_fetch_get_stock_block": ("QUANTAXIS.QAFetch", "QA_fetch_get_stock_block"),
    "QA_fetch_get_index_day": ("QUANTAXIS.QAFetch", "QA_fetch_get_index_day"),
    "QA_fetch_get_index_min": ("QUANTAXIS.QAFetch", "QA_fetch_get_index_min"),
    "QA_fetch_get_trade_date": ("QUANTAXIS.QAFetch", "QA_fetch_get_trade_date"),
    "QA_quotation": ("QUANTAXIS.QAFetch.Fetcher", "QA_quotation"),
    "QA_DataStruct_Stock_day": ("QUANTAXIS.QAData", "QA_DataStruct_Stock_day"),
    "QA_DataStruct_Stock_min": ("QUANTAXIS.QAData", "QA_DataStruct_Stock_min"),
    "QA_DataStruct_Index_day": ("QUANTAXIS.QAData", "QA_DataStruct_Index_day"),
    "QA_DataStruct_Index_min": ("QUANTAXIS.QAData", "QA_DataStruct_Index_min"),
    "QA_DataStruct_Indicators": ("QUANTAXIS.QAData", "QA_DataStruct_Indicators"),
    "QA_data_min_resample": ("QUANTAXIS.QAData", "QA_data_min_resample"),
    "QA_data_day_resample": ("QUANTAXIS.QAData", "QA_data_day_resample"),
    "QA_data_stock_to_fq": ("QUANTAXIS.QAData", "QA_data_stock_to_fq"),
    "QA_SU_save_strategy": ("QUANTAXIS.QASU.save_strategy", "QA_SU_save_strategy"),
    "QA_SU_save_stock_day": ("QUANTAXIS.QASU.main", "QA_SU_save_stock_day"),
    "QA_SU_save_stock_min": ("QUANTAXIS.QASU.main", "QA_SU_save_stock_min"),
    "QA_SU_save_index_day": ("QUANTAXIS.QASU.main", "QA_SU_save_index_day"),
    "QA_SU_save_index_min": ("QUANTAXIS.QASU.main", "QA_SU_save_index_min"),
    "QA_Setting": ("QUANTAXIS.QAUtil", "QA_Setting"),
    "DATABASE": ("QUANTAXIS.QAUtil", "DATABASE"),
    "MARKET_TYPE": ("QUANTAXIS.QAUtil", "MARKET_TYPE"),
    "ORDER_DIRECTION": ("QUANTAXIS.QAUtil", "ORDER_DIRECTION"),
    "ORDER_MODEL": ("QUANTAXIS.QAUtil", "ORDER_MODEL"),
    "FREQUENCE": ("QUANTAXIS.QAUtil", "FREQUENCE"),
    "QA_util_log_info": ("QUANTAXIS.QAUtil", "QA_util_log_info"),
    "QA_util_code_tolist": ("QUANTAXIS.QAUtil", "QA_util_code_tolist"),
    "QA_util_code_tostr": ("QUANTAXIS.QAUtil", "QA_util_code_tostr"),
    "QA_util_date_today": ("QUANTAXIS.QAUtil", "QA_util_date_today"),
    "QA_util_get_next_trade_date": ("QUANTAXIS.QAUtil", "QA_util_get_next_trade_date"),
}

__all__ = sorted(list(_DIRECT_EXPORTS.keys()) + list(_LEGACY_ATTRS.keys()))


def __getattr__(name: str):
    if name in _DIRECT_EXPORTS:
        return _DIRECT_EXPORTS[name]
    if name in _LEGACY_ATTRS:
        module_name, attr_name = _LEGACY_ATTRS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'QUANTAXIS' has no attribute {name!r}")
