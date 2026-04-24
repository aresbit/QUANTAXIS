"""Local backtesting utilities for strategy research."""

from QUANTAXIS.backtest.data import fetch_ashare_bars, fetch_ashare_daily, fetch_ashare_portfolio_bars, load_multi_ohlcv_csv, load_ohlcv_csv
from QUANTAXIS.backtest.engine import BacktestResult, ImpactConfig, run_backtest
from QUANTAXIS.backtest.market_rules import MarketContext, build_market_contexts, can_trade, infer_market_segment
from QUANTAXIS.backtest.plot import save_backtest_figure
from QUANTAXIS.backtest.portfolio import (
    PortfolioConfig,
    compute_portfolio_risk,
    optimize_portfolio,
)
from QUANTAXIS.backtest.research import (
    FactorICReport,
    GridSearchResult,
    WalkForwardResult,
    factor_ic_analysis,
    grid_search,
    walk_forward,
)
from QUANTAXIS.backtest.risk import (
    RiskChecker,
    RiskConfig,
    RiskState,
    compute_beta,
    compute_cvar,
    compute_sortino,
    compute_var,
    correlation_risk_matrix,
    factor_exposure_analysis,
    generate_risk_report,
    monte_carlo_var,
    sensitivity_analysis,
    stress_test_scenarios,
)
from QUANTAXIS.backtest.strategy import RecursiveQTransformerStrategy, StrategyConfig

__all__ = [
    "BacktestResult",
    "FactorICReport",
    "GridSearchResult",
    "ImpactConfig",
    "MarketContext",
    "PortfolioConfig",
    "RecursiveQTransformerStrategy",
    "RiskChecker",
    "RiskConfig",
    "RiskState",
    "StrategyConfig",
    "WalkForwardResult",
    "build_market_contexts",
    "can_trade",
    "compute_beta",
    "compute_cvar",
    "compute_portfolio_risk",
    "compute_sortino",
    "compute_var",
    "correlation_risk_matrix",
    "factor_exposure_analysis",
    "factor_ic_analysis",
    "fetch_ashare_bars",
    "fetch_ashare_daily",
    "fetch_ashare_portfolio_bars",
    "generate_risk_report",
    "grid_search",
    "infer_market_segment",
    "load_multi_ohlcv_csv",
    "load_ohlcv_csv",
    "monte_carlo_var",
    "optimize_portfolio",
    "run_backtest",
    "save_backtest_figure",
    "sensitivity_analysis",
    "stress_test_scenarios",
    "walk_forward",
]
